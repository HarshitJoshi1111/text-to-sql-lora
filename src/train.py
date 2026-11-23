"""Stage 5: LoRA fine-tuning with mlx-lm on Apple silicon.

    python -m src.train --run qwen2.5-0.5b-lora              train a run from params.yaml
    python -m src.train --run qwen2.5-0.5b-lora --evaluate   and score it on the evaluation set
    python -m src.train --run A B C --evaluate                several runs one after another

The base model stays frozen (and 4-bit quantised, which makes this QLoRA); only small low-rank
matrices added to every linear layer are trained, about 1 % of the parameters. The loss is computed
on the SQL answer only; examples longer than max_seq_length are left out (see fitting_examples), and
training stops if the loss turns NaN. The adapter (a few MB) is saved in adapters/<run>/; the loss curves go to
reports/training/<run>.json and reports/figures/loss_<run>.png.
"""
import argparse
import json
import math
import time
import types
from pathlib import Path

from matplotlib.figure import Figure

from src.params import PROCESSED, REPORTS, ROOT, load_params

ADAPTERS = ROOT / "adapters"
TRAINING = REPORTS / "training"
LORA_DATA = PROCESSED / "lora"  # train / valid without over-long examples


def run_settings(params: dict, run: str) -> dict:
    cfg = params["train"]
    return {**cfg["defaults"], **cfg["runs"][run]}


def fitting_examples(tokenizer, max_tokens: int) -> tuple[Path, dict]:
    """Copy of train/valid without the examples longer than max_tokens.

    mlx-lm cuts longer examples at max_tokens. With the loss on the answer only, an example whose
    prompt alone is that long keeps no answer token, and since batches hold examples of similar
    length, a batch of only such examples has a loss of 0 / 0: the first run turned NaN this way.
    """
    out, counts = LORA_DATA, {}
    out.mkdir(parents=True, exist_ok=True)
    for name in ("train.jsonl", "valid.jsonl"):
        kept = dropped = 0
        with open(PROCESSED / name, encoding="utf-8") as f, open(out / name, "w", encoding="utf-8") as g:
            for line in f:
                text = tokenizer.apply_chat_template(json.loads(line)["messages"], tokenize=False)
                if len(tokenizer.encode(text, add_special_tokens=False)) <= max_tokens:
                    g.write(line)
                    kept += 1
                else:
                    dropped += 1
        counts[name] = {"kept": kept, "dropped": dropped}
    return out, counts


def lora_args(run: str, s: dict) -> types.SimpleNamespace:
    """mlx-lm's own defaults, overridden by the settings of this run."""
    from mlx_lm.lora import CONFIG_DEFAULTS

    lr, iters, warmup = s["learning_rate"], s["iters"], s["warmup"]
    updates = iters // s["grad_accumulation"]  # the learning-rate schedule counts optimiser updates
    args = {**CONFIG_DEFAULTS,
            "model": s["model"], "train": True, "test": False, "data": str(LORA_DATA),
            "adapter_path": str(ADAPTERS / run), "fine_tune_type": "lora",
            "batch_size": s["batch_size"], "iters": iters, "learning_rate": lr, "num_layers": s["num_layers"],
            "lora_parameters": {"rank": s["rank"], "scale": s["scale"], "dropout": s["dropout"]},
            "lr_schedule": {"name": "cosine_decay", "warmup": warmup, "warmup_init": 0.0,
                            "arguments": [lr, updates - warmup, lr / 10]},
            "grad_accumulation_steps": s["grad_accumulation"], "grad_checkpoint": s["grad_checkpoint"],
            "mask_prompt": s["mask_prompt"], "max_seq_length": s["max_seq_length"],
            "steps_per_eval": s["steps_per_eval"], "val_batches": s["val_batches"], "save_every": s["save_every"],
            "seed": s["seed"], "report_to": None}
    return types.SimpleNamespace(**args)


def figure(curve: dict, path: Path, title: str) -> None:
    fig = Figure(figsize=(6, 3.6))
    ax = fig.subplots()
    ax.plot([r["iteration"] for r in curve["train"]], [r["train_loss"] for r in curve["train"]], label="training", alpha=0.7)
    ax.plot([r["iteration"] for r in curve["val"]], [r["val_loss"] for r in curve["val"]], "o-", label="validation")
    ax.set(xlabel="Iteration (batch of 2 questions)", ylabel="Loss (SQL tokens only)", title=title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def train(run: str, params: dict) -> dict:
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm.lora import train_model
    from mlx_lm.tuner.callbacks import TrainingCallback
    from mlx_lm.tuner.datasets import load_dataset
    from mlx_lm.utils import get_total_parameters, load

    class Recorder(TrainingCallback):
        def __init__(self):
            self.train, self.val = [], []

        def on_train_loss_report(self, info: dict):
            self.train.append({k: round(float(v), 5) for k, v in info.items()})
            if math.isnan(info["train_loss"]):  # stop instead of training on a broken model
                raise FloatingPointError(f"training loss is NaN at iteration {info['iteration']}")

        def on_val_loss_report(self, info: dict):
            self.val.append({k: round(float(v), 5) for k, v in info.items()})
            print(f"{run}: iteration {int(info['iteration'])}, validation loss {info['val_loss']:.4f}", flush=True)

    settings = run_settings(params, run)
    args = lora_args(run, settings)
    np.random.seed(args.seed)
    model, tokenizer = load(args.model)
    if settings.get("dtype"):  # the 4-bit models keep scales and norms in float16, whose range can overflow
        import mlx.core as mx

        model.set_dtype(getattr(mx, settings["dtype"]))  # only floating-point weights; the 4-bit ones stay
    _, filtered = fitting_examples(tokenizer, args.max_seq_length)
    print(f"{run}: examples longer than {args.max_seq_length} tokens left out: {filtered}", flush=True)
    train_set, valid_set, _ = load_dataset(args, tokenizer)
    recorder, start = Recorder(), time.time()
    train_model(args, model, train_set, valid_set, recorder)
    minutes = (time.time() - start) / 60
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    total = get_total_parameters(model)  # counts 4-bit weights unpacked, as mlx-lm reports them

    summary = {"run": run, "settings": settings, "examples": filtered, "minutes": round(minutes, 1),
               "trainable_parameters": int(trainable),
               "share_trainable": round(trainable / total, 4),
               "peak_memory_gb": max((r["peak_memory"] for r in recorder.train), default=None),
               "tokens_per_second": round(float(np.mean([r["tokens_per_second"] for r in recorder.train])), 1),
               "final_val_loss": recorder.val[-1]["val_loss"] if recorder.val else None,
               "curve": {"train": recorder.train, "val": recorder.val}}
    TRAINING.mkdir(parents=True, exist_ok=True)
    (TRAINING / f"{run}.json").write_text(json.dumps(summary, indent=2) + "\n")
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    figure(summary["curve"], REPORTS / "figures" / f"loss_{run}.png", f"{run}: {settings['iters']} iterations")
    print(f"{run}: {minutes:.0f} min, {trainable / 1e6:.1f} M trainable parameters ({trainable / total:.1%}), "
          f"validation loss {summary['final_val_loss']}, peak memory {summary['peak_memory_gb']} GB")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, nargs="+", help="run name(s) from params.yaml")
    parser.add_argument("--evaluate", action="store_true", help="score the adapter on the evaluation set afterwards")
    args = parser.parse_args()
    params = load_params()
    for run in args.run:
        train(run, params)
        if args.evaluate:
            from src.evaluate import evaluate

            adapter = str((ADAPTERS / run).relative_to(ROOT))  # relative, so the report holds no local path
            evaluate(run, run_settings(params, run)["model"], adapter, params)


if __name__ == "__main__":
    main()
