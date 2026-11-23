"""Stage 4: one table of all evaluated runs (reports/runs/*.json) -> reports/results.md, plus the
fine-tuning runs (reports/training/*.json) and a figure of accuracy against training questions."""
import json

from matplotlib.figure import Figure

from src.evaluate import RUNS
from src.params import REPORTS
from src.train import TRAINING

OUTCOMES = [("unknown_column", "unknown column"), ("syntax", "syntax error"), ("wrong_select", "wrong SELECT"),
            ("wrong_where", "wrong WHERE"), ("wrong_both", "both wrong")]


def pct(value) -> str:
    return "-" if value is None else f"{value:.1%}"


def fine_tuning(runs: list[dict]) -> list[str]:
    """Table of the LoRA runs (settings, cost, accuracy) and the figure of accuracy against data size."""
    trained = {t["run"]: t for t in (json.loads(p.read_text()) for p in sorted(TRAINING.glob("*.json")))}
    scored = {r["name"]: r for r in runs}
    rows = [(trained[name], scored[name]) for name in trained if name in scored]
    if not rows:
        return []
    rows.sort(key=lambda tr: (tr[0]["settings"]["model"], questions(tr[0]), tr[0]["settings"]["rank"]))
    lines = ["", "## Fine-tuning runs", "",
             "Same settings unless listed (see params.yaml): QLoRA on all layers, batch 2 with 2 steps of gradient "
             "accumulation, peak learning rate 5e-5, one pass over the questions shown. One run per setting: with "
             "500 questions, an accuracy near 80 % has a 95 % interval of about ±3.5 points, so smaller differences "
             "are not reliable. Training times depend on what else the laptop was doing.", "",
             "| Run | Model | Training questions | Rank | Trainable parameters | Training time | Peak memory "
             "| Validation loss | Execution accuracy |",
             "|---|---|---|---|---|---|---|---|---|"]
    for t, r in rows:
        s = t["settings"]
        lines.append(f"| {t['run']} | {s['model'].split('/')[-1]} | {questions(t):,} | {s['rank']} | "
                     f"{t['trainable_parameters'] / 1e6:.1f} M | {t['minutes']:.0f} min | {t['peak_memory_gb']:.1f} GB | "
                     f"{t['final_val_loss']:.3f} | **{pct(r['execution_accuracy'])}** |")
    figure(rows, scored)
    lines += ["", "![Execution accuracy against training questions](figures/accuracy_vs_questions.png)"]
    return lines


def questions(training: dict) -> int:
    return training["settings"]["iters"] * training["settings"]["batch_size"]


def figure(rows: list[tuple[dict, dict]], scored: dict) -> None:
    """Execution accuracy against training questions, per base model, rank 8 only; 0 = zero-shot."""
    fig = Figure(figsize=(6, 3.6))
    ax = fig.subplots()
    for i, model in enumerate(sorted({t["settings"]["model"] for t, _ in rows})):
        points = sorted((questions(t), r["execution_accuracy"]) for t, r in rows
                        if t["settings"]["model"] == model and t["settings"]["rank"] == 8)
        zero_shot = [r for r in scored.values() if r["model"] == model and not r["adapter"]]
        if zero_shot:
            points.insert(0, (0, zero_shot[0]["execution_accuracy"]))
        label = model.split("/")[-1].replace("-Instruct-4bit", "")
        ax.plot([p[0] for p in points], [100 * p[1] for p in points], "o-", label=label)
        for x, y in points:
            ax.annotate(f"{100 * y:.1f}", (x, 100 * y), textcoords="offset points", xytext=(0, -13 if i == 0 else 6),
                        ha="center", fontsize=8)  # labels of the first line below it, so that they do not overlap
    ax.set(xlabel="Training questions (0 = zero-shot)", ylabel="Execution accuracy (%)", ylim=(0, 100),
           title="Accuracy on 500 test questions")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS / "figures" / "accuracy_vs_questions.png", dpi=150)


def main() -> None:
    runs = sorted((json.loads(p.read_text()) for p in RUNS.glob("*.json")), key=lambda r: r["execution_accuracy"])
    n = runs[0]["questions"] if runs else 0
    lines = ["# Results on the evaluation set", "",
             f"{n} questions from the WikiSQL test split (tables never seen in training). Execution accuracy: the "
             "query returns the same result as the true query. Runs: the query executes without error.", "",
             "| Run | Model | Execution accuracy | Exact match | Runs | No aggregation | Aggregation | 2+ conditions "
             "| s/question |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        two_plus = [r["by_conditions"][k] for k in ("2", "3+")]
        n_two = sum(x["n"] for x in two_plus)
        acc_two = sum(x["accuracy"] * x["n"] for x in two_plus if x["accuracy"] is not None) / n_two if n_two else None
        lines.append(f"| {r['name']} | {r['model'].split('/')[-1]}{' + LoRA' if r['adapter'] else ''} | "
                     f"**{pct(r['execution_accuracy'])}** | {pct(r['exact_match'])} | {pct(r['runs'])} | "
                     f"{pct(r['by_aggregation']['none']['accuracy'])} | {pct(r['by_aggregation']['aggregate']['accuracy'])} | "
                     f"{pct(acc_two)} | {r['seconds_per_question']} |")
    lines += ["", "## Where the wrong answers come from (share of all questions)", "",
              "| Run | " + " | ".join(label for _, label in OUTCOMES) + " | other |", "|---" * (len(OUTCOMES) + 2) + "|"]
    for r in runs:
        counts = r["outcomes"]
        listed = sum(counts.get(k, 0) for k, _ in OUTCOMES) + counts.get("correct", 0)
        lines.append(f"| {r['name']} | " + " | ".join(pct(counts.get(k, 0) / r["questions"]) for k, _ in OUTCOMES) +
                     f" | {pct((r['questions'] - listed) / r['questions'])} |")
    lines += fine_tuning(runs)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
