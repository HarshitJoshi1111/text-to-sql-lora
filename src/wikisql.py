"""WikiSQL: questions about single Wikipedia tables, each with the SQL query that answers it.

The dataset stores a query as a small logical form (selected column, aggregation, conditions);
here it is turned into a real SQLite query on the table's real column names, and every table is
loaded into an in-memory SQLite database so that both the true and a generated query can be run.
Text columns compare case-insensitively (COLLATE NOCASE), as the official WikiSQL evaluation does.
"""
import json
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

AGGREGATIONS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
OPERATORS = ["=", ">", "<"]
TABLE = "t"  # every question is about one table; the prompt calls it t


@dataclass(frozen=True)
class Table:
    id: str
    columns: tuple[str, ...]  # unique, non-empty names
    types: tuple[str, ...]  # "text" or "real"
    rows: tuple[tuple, ...]


def column_names(header: list[str]) -> tuple[str, ...]:
    """Make header names usable as SQL columns: empty names get a placeholder, duplicates a number."""
    names, seen = [], {}
    for i, name in enumerate(header):
        name = " ".join(name.split()) or f"column {i + 1}"
        key = name.lower()
        seen[key] = seen.get(key, 0) + 1
        names.append(name if seen[key] == 1 else f"{name} ({seen[key]})")
    return tuple(names)


def quote_name(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def quote_value(value, column_type: str) -> str:
    if column_type == "real" and isinstance(value, int | float):
        return repr(value) if isinstance(value, float) else str(value)
    if column_type == "real":
        try:
            number = float(str(value).replace(",", ""))
            return str(int(number)) if number.is_integer() else repr(number)
        except ValueError:
            pass
    return "'" + str(value).replace("'", "''") + "'"


def load_tables(path: Path) -> dict[str, Table]:
    tables = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            tables[t["id"]] = Table(t["id"], column_names(t["header"]), tuple(t["types"]), tuple(map(tuple, t["rows"])))
    return tables


def load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def create_statement(table: Table) -> str:
    columns = ",\n  ".join(f"{quote_name(c)} {'REAL' if ty == 'real' else 'TEXT'}"
                            for c, ty in zip(table.columns, table.types, strict=True))
    return f"CREATE TABLE {TABLE} (\n  {columns}\n);"


def to_sql(logical_form: dict, table: Table) -> str:
    """WikiSQL logical form -> SQLite query, e.g. SELECT COUNT("School") FROM t WHERE "No." = '3'."""
    column = quote_name(table.columns[logical_form["sel"]])
    aggregation = AGGREGATIONS[logical_form["agg"]]
    select = f"{aggregation}({column})" if aggregation else column
    sql = f"SELECT {select} FROM {TABLE}"
    conditions = [f"{quote_name(table.columns[c])} {OPERATORS[op]} {quote_value(v, table.types[c])}"
                  for c, op, v in logical_form["conds"]]
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    return sql


@lru_cache(maxsize=256)
def connection(table: Table) -> sqlite3.Connection:
    """In-memory database holding one table; text columns compare case-insensitively."""
    db = sqlite3.connect(":memory:", check_same_thread=False)
    columns = ", ".join(f"{quote_name(c)} {'REAL' if ty == 'real' else 'TEXT COLLATE NOCASE'}"
                        for c, ty in zip(table.columns, table.types, strict=True))
    db.execute(f"CREATE TABLE {TABLE} ({columns})")
    db.executemany(f"INSERT INTO {TABLE} VALUES ({', '.join('?' * len(table.columns))})", table.rows)
    return db


class SQLError(Exception):
    pass


def quoted_names(sql: str) -> list[str]:
    """Double-quoted names in a query, skipping 'text values' (which may themselves contain quotes)."""
    names, i = [], 0
    while i < len(sql):
        quote = sql[i]
        if quote in "'\"":
            j, part = i + 1, []
            while j < len(sql):
                if sql[j] == quote and sql[j + 1 : j + 2] == quote:  # doubled quote = escaped
                    part.append(quote)
                    j += 2
                elif sql[j] == quote:
                    break
                else:
                    part.append(sql[j])
                    j += 1
            if quote == '"':
                names.append("".join(part))
            i = j + 1
        else:
            i += 1
    return names


def check_names(sql: str, table: Table) -> None:
    """SQLite treats a double-quoted name that is not a column as a text literal ("Team" -> 'Team'), so a
    wrong column name would silently return that text. Such names are rejected here as unknown columns."""
    known = {c.lower() for c in table.columns} | {TABLE}
    for name in quoted_names(sql):
        if name.lower() not in known:
            raise SQLError(f"no such column: {name}")


def execute(sql: str, table: Table, max_rows: int = 1000) -> list[tuple]:
    """Run a query; any SQLite error (syntax, unknown column, ...) is raised as SQLError."""
    check_names(sql, table)
    try:
        cursor = connection(table).execute(sql)
        return cursor.fetchmany(max_rows)
    except (sqlite3.Error, sqlite3.Warning, OverflowError) as error:
        raise SQLError(str(error)) from error


def same_result(a: list[tuple], b: list[tuple]) -> bool:
    """Compare results as multisets of rows; numbers compared after rounding, text case-insensitively."""
    def norm(row):
        return tuple(round(float(v), 4) if isinstance(v, int | float) else v.lower() if isinstance(v, str) else v for v in row)
    return sorted(map(norm, a), key=repr) == sorted(map(norm, b), key=repr)
