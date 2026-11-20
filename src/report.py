"""Stage 4: one table of all evaluated runs (reports/runs/*.json) -> reports/results.md."""
import json

from src.evaluate import RUNS
from src.params import REPORTS

OUTCOMES = [("unknown_column", "unknown column"), ("syntax", "syntax error"), ("wrong_select", "wrong SELECT"),
            ("wrong_where", "wrong WHERE"), ("wrong_both", "both wrong")]


def pct(value) -> str:
    return "-" if value is None else f"{value:.1%}"


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
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
