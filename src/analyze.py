"""Stage 6: why the wrong answers are wrong.

    python -m src.analyze      reports/errors.md from the saved predictions of every run

Both queries are read back into the WikiSQL shape (aggregation, selected column, conditions) and
compared part by part. A wrong answer can have several problems, so the shares in a row can add up
to more than the share of wrong answers. Condition values compare case-insensitively, as SQLite does
for text columns here, and numbers compare as numbers ('25' = 25).
"""
import json
import re
from dataclasses import dataclass

from src.evaluate import RUNS
from src.params import PREDICTIONS, RAW, REPORTS
from src.wikisql import Table, load_tables

NAME = r'"(?:[^"]|"")*"|`[^`]*`|[A-Za-z_][A-Za-z0-9_]*'
VALUE = r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|-?[0-9][0-9.,]*%?|[A-Za-z_][A-Za-z0-9_]*"
SELECT = re.compile(rf"^select\s+(?:(max|min|count|sum|avg)\s*\(\s*({NAME})\s*\)|({NAME}))\s+from\s+t\b\s*(.*)$", re.I | re.S)
CONDITION = re.compile(rf"^({NAME})\s*(=|>|<|>=|<=|!=|<>|like)\s*({VALUE})$", re.I | re.S)
AND = re.compile(r"\s+and\s+", re.I)
NOTES = "<!-- notes -->"

# (key, label): problems with the selected part, then with the WHERE part
PROBLEMS = [
    ("aggregation", "wrong aggregation"),
    ("select_column", "wrong SELECT column"),
    ("condition_column", "condition on a wrong column"),
    ("missing_condition", "condition missing"),
    ("extra_condition", "extra condition"),
    ("operator", "wrong operator"),
    ("value_other_span", "value: other words of the question"),
    ("value_changed", "value: changed or invented"),
]


@dataclass(frozen=True)
class Query:
    aggregation: str  # "" for none
    column: str
    conditions: tuple[tuple[str, str, object], ...]  # (column, operator, value)


def unquote_name(name: str) -> str:
    if name[0] in '"`':
        name = name[1:-1].replace('""', '"')
    return name.lower()


def plain_value(value: str):
    """Comparable form of a condition value: lower-case text, or a float for numbers."""
    if value[0] in "'\"":
        value = value[1:-1].replace(value[0] * 2, value[0])
    text = value.strip().lower()
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return text


def split_conditions(where: str) -> list[str]:
    """Split at AND outside quotes."""
    parts, start, quote = [], 0, None
    for i, char in enumerate(where):
        if quote:
            quote = None if char == quote else quote
        elif char in "'\"`":
            quote = char
        elif match := AND.match(where, i):
            parts.append(where[start:i])
            start = match.end()
    return [p.strip() for p in parts + [where[start:]] if p.strip()]


def parse(sql: str) -> Query | None:
    """The WikiSQL parts of a query, or None if it has another shape (joins, ORDER BY, OR, ...)."""
    match = SELECT.match(sql.strip().rstrip(";").strip())
    if not match:
        return None
    aggregation, inner, plain, rest = match.groups()
    rest = rest.strip()
    conditions = []
    if rest:
        if not rest.lower().startswith("where "):
            return None
        for part in split_conditions(rest[6:]):
            cond = CONDITION.match(part)
            if not cond:
                return None
            column, operator, value = cond.groups()
            conditions.append((unquote_name(column), operator.lower(), plain_value(value)))
    return Query((aggregation or "").upper(), unquote_name(inner or plain), tuple(conditions))


def in_question(value, question: str) -> bool:
    if isinstance(value, float):
        number = f"{value:g}"
        return re.search(rf"(?<![0-9.]){re.escape(number)}(?![0-9])", question.replace(",", "")) is not None
    return value in question.lower()


def problems(pred: Query, gold: Query, question: str) -> set[str]:
    found = set()
    if pred.aggregation != gold.aggregation:
        found.add("aggregation")
    if pred.column != gold.column:
        found.add("select_column")
    left_pred, left_gold = list(pred.conditions), []
    for cond in gold.conditions:  # 1. conditions that are right
        if cond in left_pred:
            left_pred.remove(cond)
        else:
            left_gold.append(cond)
    for column, operator, value in list(left_gold):  # 2. right column, wrong operator or value
        other = next((c for c in left_pred if c[0] == column), None)
        if other is None:
            continue
        left_pred.remove(other)
        left_gold.remove((column, operator, value))
        if other[1] != operator:
            found.add("operator")
        elif other[2] != value:
            found.add("value_other_span" if in_question(other[2], question) else "value_changed")
    for column, operator, value in list(left_gold):  # 3. right value, wrong column
        other = next((c for c in left_pred if c[2] == value), None)
        if other is not None:
            left_pred.remove(other)
            left_gold.remove((column, operator, value))
            found.add("condition_column")
    if left_gold:
        found.add("missing_condition")
    if left_pred:
        found.add("extra_condition")
    return found


def column_values(table: Table, column: str) -> set:
    """The cells of a column in the comparable form of plain_value (empty if there is no such column)."""
    names = [c.lower() for c in table.columns]
    if column not in names:
        return set()
    i = names.index(column)
    return {plain_value(str(row[i])) if str(row[i]) else "" for row in table.rows}


def table_checks(pred: Query, gold: Query, table: Table) -> set[str]:
    """Would a look at the table cells have helped? Only '=' conditions, the ones that name a cell.

    column_by_value: a condition is on a wrong column, and its value occurs in the true column only.
    value_not_in_table: a generated value occurs nowhere in the table.
    """
    found = set()
    for column, operator, value in gold.conditions:
        if operator != "=" or (column, operator, value) in pred.conditions:
            continue
        for other, other_operator, other_value in pred.conditions:
            if (other_operator == "=" and other_value == value and other != column
                    and value in column_values(table, column) and value not in column_values(table, other)):
                found.add("column_by_value")
    every_value = set().union(*(column_values(table, c.lower()) for c in table.columns))
    for cond in pred.conditions:
        if cond[1] == "=" and cond not in gold.conditions and cond[2] not in every_value:
            found.add("value_not_in_table")
    return found


def breakdown(records: list[dict], tables: dict[str, Table]) -> dict:
    """Counts per problem over the wrong answers that run; queries that do not run are counted apart."""
    counts = {key: 0 for key, _ in PROBLEMS}
    counts.update(wrong=0, not_run=0, other_shape=0, column_by_value=0, value_not_in_table=0)
    for r in records:
        if r["correct"]:
            continue
        counts["wrong"] += 1
        if not r["status"].startswith("wrong_"):
            counts["not_run"] += 1
            continue
        pred, gold = parse(r["pred"]), parse(r["sql"])
        if pred is None or gold is None:
            counts["other_shape"] += 1
            continue
        found = problems(pred, gold, r["question"])
        checks = table_checks(pred, gold, tables[r["table_id"]])
        if "condition_column" not in found:
            checks.discard("column_by_value")
        for key in found | checks:
            counts[key] += 1
    return counts


def pct(part: int, whole: int) -> str:
    return f"{part / whole:.1%}" if whole else "-"


def main() -> None:
    runs = sorted((json.loads(p.read_text()) for p in RUNS.glob("*.json")), key=lambda r: -r["execution_accuracy"])
    lines = ["# Why the wrong answers are wrong", "",
             "Every wrong answer that runs is compared with the true query part by part (`python -m src.analyze`). "
             "Shares are of all evaluation questions; one answer can have several problems.", "",
             "| Run | Wrong | Does not run | Other query shape | " + " | ".join(label for _, label in PROBLEMS) + " |",
             "|---" * (len(PROBLEMS) + 4) + "|"]
    tables, details = load_tables(RAW / "test.tables.jsonl"), {}
    for run in runs:
        with open(PREDICTIONS / f"{run['name']}.jsonl", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        counts = details[run["name"]] = breakdown(records, tables)
        n = len(records)
        shares = [pct(counts[key], n) for key in ("wrong", "not_run", "other_shape")]
        lines.append(f"| {run['name']} | " + " | ".join(shares) + " | "
                     + " | ".join(pct(counts[key], n) for key, _ in PROBLEMS) + " |")
    lines += ["", "Value problems are split by whether the generated value is still a piece of the question "
              "(the model picked the wrong words) or not (it changed a name, a spelling or a number). "
              "Other query shape: runs, but not in the WikiSQL form (ORDER BY, LIKE, OR, ...).", "",
              "## Would the table cells have helped?", "",
              "The prompt shows the column names and 3 example rows, not the whole table. For '=' conditions:", "",
              "| Run | Condition on a wrong column, value only in the true column | Generated value not in the table |",
              "|---|---|---|"]
    for name, counts in details.items():
        lines.append(f"| {name} | {counts['column_by_value']} of {counts['condition_column']} answers | "
                     f"{counts['value_not_in_table']} of {counts['wrong'] - counts['not_run'] - counts['other_shape']} "
                     "wrong answers that run |")
    (REPORTS / "errors.json").write_text(json.dumps(details, indent=2) + "\n")
    path = REPORTS / "errors.md"
    old = path.read_text() if path.exists() else ""
    notes = old[old.index(NOTES):] if NOTES in old else ""  # written by hand below the table: kept
    path.write_text("\n".join(lines) + "\n" + ("\n" + notes if notes else ""))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
