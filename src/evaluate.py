"""Stage 3: run a model on the fixed evaluation set and score it.

    python -m src.evaluate --all                         every zero-shot model in params.yaml
    python -m src.evaluate --name qwen2.5-0.5b           one of them
    python -m src.evaluate --name lora --model Qwen/Qwen2.5-0.5B-Instruct --adapter adapters/lora
    python -m src.evaluate --rescore                     score saved predictions again (no model)

Main metric: execution accuracy, the share of questions where the generated query returns the same
result as the true one (the same answer can be written in different ways, so the text is not compared).
Exact match of the normalised query is reported too. Failures are split into queries that do not
run (unknown column, syntax) and queries that run but answer something else (wrong column in SELECT,
wrong aggregation, wrong condition).
"""
import argparse
import json
import re
import time

from src.params import PREDICTIONS, PROCESSED, RAW, REPORTS, load_params
from src.prompts import messages
from src.sql_text import normalize
from src.wikisql import SQLError, Table, execute, load_tables, same_result

RUNS = REPORTS / "runs"
CLAUSE = re.compile(r"^select (.*?) from t(?: where (.*))?$")


def read_eval() -> list[dict]:
    with open(PROCESSED / "eval.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def diagnose(pred: str, gold: str) -> str:
    """Which part of a wrong query differs from the true one: select, where or both (other: unusual shape)."""
    p, g = CLAUSE.match(normalize(pred)), CLAUSE.match(normalize(gold))
    if not p or not g:
        return "other"
    select_differs, where_differs = p.group(1) != g.group(1), (p.group(2) or "") != (g.group(2) or "")
    return {(True, False): "select", (False, True): "where", (True, True): "both"}.get((select_differs, where_differs), "same_text")


def score(item: dict, sql: str, table: Table) -> dict:
    gold = execute(item["sql"], table)
    if not sql:
        return {"correct": False, "status": "no_sql"}
    try:
        result = execute(sql, table)
    except SQLError as error:
        message = str(error)
        kind = "unknown_column" if "no such column" in message else "syntax" if "syntax" in message else "other_error"
        return {"correct": False, "status": kind, "error": message[:200]}
    if same_result(result, gold):
        return {"correct": True, "status": "correct"}
    return {"correct": False, "status": "wrong_" + diagnose(sql, item["sql"])}


def accuracy(records: list[dict], keep=lambda r: True) -> dict:
    subset = [r for r in records if keep(r)]
    return {"n": len(subset), "accuracy": round(sum(r["correct"] for r in subset) / len(subset), 4) if subset else None}


def summarize(name: str, model: str, adapter: str | None, records: list[dict], seconds_per_question: float) -> dict:
    statuses = {}
    for r in records:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    return {
        "name": name, "model": model, "adapter": adapter, "questions": len(records),
        "execution_accuracy": accuracy(records)["accuracy"],
        "exact_match": round(sum(r["exact"] for r in records) / len(records), 4),
        "runs": round(sum(not r["status"].endswith(("error", "column", "syntax", "no_sql")) for r in records) / len(records), 4),
        "outcomes": dict(sorted(statuses.items(), key=lambda kv: -kv[1])),
        "by_aggregation": {"none": accuracy(records, lambda r: r["agg"] == 0),
                           "aggregate": accuracy(records, lambda r: r["agg"] > 0)},
        "by_conditions": {"0-1": accuracy(records, lambda r: r["conds"] <= 1), "2": accuracy(records, lambda r: r["conds"] == 2),
                          "3+": accuracy(records, lambda r: r["conds"] >= 3)},
        "seconds_per_question": seconds_per_question,
    }


def save(result: dict, records: list[dict]) -> None:
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    with open(PREDICTIONS / f"{result['name']}.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{result['name']}.json").write_text(json.dumps(result, indent=2) + "\n")


def rescore() -> None:
    """Score saved predictions again (after a change to the scoring code), without running any model."""
    tables = load_tables(RAW / "test.tables.jsonl")
    for path in sorted(RUNS.glob("*.json")):
        old = json.loads(path.read_text())
        with open(PREDICTIONS / f"{old['name']}.jsonl", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        records = [{**{k: v for k, v in r.items() if k not in ("correct", "status", "error")},
                    "exact": normalize(r["pred"]) == normalize(r["sql"]), **score(r, r["pred"], tables[r["table_id"]])}
                   for r in records]
        result = summarize(old["name"], old["model"], old["adapter"], records, old["seconds_per_question"])
        save(result, records)
        print(f"{result['name']}: execution accuracy {result['execution_accuracy']}, exact match {result['exact_match']}")


def evaluate(name: str, model: str, adapter: str | None, params: dict, limit: int | None = None) -> dict:
    from src.generate import Generator

    cfg, prep = params["evaluate"], params["prepare"]
    items = read_eval()[:limit]
    tables = load_tables(RAW / "test.tables.jsonl")
    generator = Generator(model, adapter, cfg["max_tokens"])
    records, start = [], time.time()
    for i in range(0, len(items), cfg["batch_size"]):
        batch = items[i : i + cfg["batch_size"]]
        chats = [messages(tables[q["table_id"]], q["question"], prep["example_rows"], prep["max_cell_chars"]) for q in batch]
        for q, (raw, sql) in zip(batch, generator.sql(chats), strict=True):
            records.append({**q, "raw": raw, "pred": sql, "exact": normalize(sql) == normalize(q["sql"]),
                            **score(q, sql, tables[q["table_id"]])})
        print(f"{name}: {len(records)}/{len(items)}, accuracy so far {sum(r['correct'] for r in records) / len(records):.3f}",
              flush=True)
    seconds = time.time() - start
    result = summarize(name, model, adapter, records, round(seconds / len(records), 3))
    save(result, records)
    print(json.dumps({k: result[k] for k in ("name", "execution_accuracy", "exact_match", "runs", "outcomes")}))
    return result


def main() -> None:
    params = load_params()
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="every model in params.yaml (zero-shot)")
    parser.add_argument("--name")
    parser.add_argument("--model", help="Hugging Face id; default: the model of --name in params.yaml")
    parser.add_argument("--adapter", help="LoRA adapter directory")
    parser.add_argument("--limit", type=int, help="only the first N questions (quick check)")
    parser.add_argument("--rescore", action="store_true", help="score the saved predictions again, no model")
    args = parser.parse_args()
    if args.rescore:
        rescore()
        return
    models = params["evaluate"]["models"]
    jobs = models.items() if args.all else [(args.name, args.model or models[args.name])]
    for name, model in jobs:
        evaluate(name, model, args.adapter, params, args.limit)


if __name__ == "__main__":
    main()
