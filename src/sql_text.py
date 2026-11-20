"""Getting the SQL query out of a model's answer, and normalising queries for exact-match comparison."""
import re

FENCE = re.compile(r"```(?:sql|sqlite)?\s*(.*?)(?:```|$)", re.S | re.I)


def extract_sql(text: str) -> str:
    """First SQL statement in the answer: code fences and a leading 'SQL:' removed, cut at the first ';'."""
    match = FENCE.search(text)
    sql = match.group(1) if match else text
    sql = re.sub(r"^\s*(sql|sqlite|query)\s*:\s*", "", sql.strip(), flags=re.I)
    start = re.search(r"\b(select|with)\b", sql, re.I)
    if start:
        sql = sql[start.start():]
    sql = sql.split(";")[0]
    return " ".join(sql.split())


SIMPLE_NAME = re.compile(r'"([A-Za-z_][A-Za-z0-9_]*)"')


def normalize(sql: str) -> str:
    """For exact-match accuracy: lower case, one space between tokens, `name` written as "name", and quotes
    dropped around simple names (Player = "Player"); names with spaces or symbols keep their quotes, since
    without them the query would not run."""
    sql = SIMPLE_NAME.sub(r"\1", re.sub(r"`([^`]*)`", r'"\1"', sql)).rstrip(";").lower()
    sql = re.sub(r"\s*([(),=<>])\s*", r" \1 ", sql)
    return " ".join(sql.split())
