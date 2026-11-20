"""Stage 2: training, validation and evaluation sets.

- train.jsonl: every WikiSQL training question as a chat (system, user, assistant = true query),
  the format mlx-lm fine-tunes on
- valid.jsonl: questions from WikiSQL dev in the same format, for the validation loss
- eval.jsonl: a fixed random sample of WikiSQL test questions with the true query, for every model

Questions whose true query returns nothing (8 % of WikiSQL, mostly values spelled differently in the
question annotation than in the table) are left out: a model could not be checked on them, and an
empty answer would count as correct.
"""
import json
import random

from src.params import PROCESSED, RAW, REPORTS, load_params
from src.prompts import messages
from src.wikisql import SQLError, execute, load_questions, load_tables, to_sql


def usable(split: str) -> tuple[list[dict], dict, dict]:
    """(questions with a true query that returns something, tables, counts)."""
    tables = load_tables(RAW / f"{split}.tables.jsonl")
    kept, counts = [], {"questions": 0, "empty_result": 0, "error": 0}
    for i, q in enumerate(load_questions(RAW / f"{split}.jsonl")):
        counts["questions"] += 1
        table = tables[q["table_id"]]
        sql = to_sql(q["sql"], table)
        try:
            result = execute(sql, table)
        except SQLError:
            counts["error"] += 1
            continue
        if not result or result == [(None,)]:
            counts["empty_result"] += 1
            continue
        kept.append({"id": f"{split}-{i}", "table_id": q["table_id"], "question": q["question"], "sql": sql,
                     "agg": q["sql"]["agg"], "conds": len(q["sql"]["conds"])})
    counts["kept"] = len(kept)
    return kept, tables, counts


def write_jsonl(rows: list[dict], name: str) -> None:
    with open(PROCESSED / name, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    cfg = load_params()["prepare"]
    rng = random.Random(cfg["seed"])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split, name in (("train", "train.jsonl"), ("dev", "valid.jsonl"), ("test", "eval.jsonl")):
        questions, tables, summary[split] = usable(split)
        if split == "dev":
            questions = rng.sample(questions, cfg["valid_size"])
        if split == "test":
            write_jsonl(rng.sample(questions, cfg["eval_size"]), name)
            continue
        write_jsonl([{"messages": messages(tables[q["table_id"]], q["question"], cfg["example_rows"],
                                           cfg["max_cell_chars"], answer=q["sql"])} for q in questions], name)
        print(f"{split}: {summary[split]}")
    print(f"test: {summary['test']}")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "data.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Data", "", "Questions per WikiSQL split, and how many are used (true query returns a result).", "",
             "| Split | Questions | True query returns nothing | Used | Used for |", "|---|---|---|---|---|"]
    use = {"train": "fine-tuning", "dev": f"validation loss ({cfg['valid_size']} sampled)",
           "test": f"evaluation ({cfg['eval_size']} sampled)"}
    for split, c in summary.items():
        lines.append(f"| {split} | {c['questions']:,} | {c['empty_result'] + c['error']:,} | {c['kept']:,} | {use[split]} |")
    (REPORTS / "data.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
