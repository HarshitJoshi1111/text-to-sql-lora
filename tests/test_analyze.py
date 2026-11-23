from src.analyze import Query, parse, problems, split_conditions, table_checks
from tests.test_wikisql import TABLE

CLUB = '"Club ""A"""'  # a column name with quotes in it
QUESTION = "Which player wears number 42 and plays for Cincinnati?"
GOLD = parse(f"SELECT \"Player\" FROM t WHERE \"No.\" = '42' AND {CLUB} = 'Cincinnati'")


def found(where: str, select: str = '"Player"') -> set[str]:
    return problems(parse(f"SELECT {select} FROM t WHERE {where}"), GOLD, QUESTION)


def test_parse_reads_the_wikisql_parts():
    assert GOLD == Query("", "player", (("no.", "=", 42.0), ('club "a"', "=", "cincinnati")))
    assert parse("SELECT count(Player) FROM t") == Query("COUNT", "player", ())
    assert parse("SELECT Player FROM t ORDER BY 1") is None  # not the WikiSQL shape
    assert split_conditions("a = 'x and y' AND b > 2") == ["a = 'x and y'", "b > 2"]


def test_problems_name_each_wrong_part():
    assert found(f"{CLUB} = 'cincinnati' AND \"No.\" = 42") == set()
    assert found(f"\"No.\" = 42 AND {CLUB} = 'Cincinnati'", 'COUNT("No.")') == {"aggregation", "select_column"}
    assert found(f"\"Player\" = '42' AND {CLUB} = 'Cincinnati'") == {"condition_column"}
    assert found('"No." = 42') == {"missing_condition"}
    assert found(f"\"No.\" > 42 AND {CLUB} = 'Cincinnati'") == {"operator"}
    assert found(f"\"No.\" = 42 AND {CLUB} = 'plays'") == {"value_other_span"}
    assert found(f"\"No.\" = 42 AND {CLUB} = 'Cincinatti'") == {"value_changed"}


def test_table_checks_look_at_the_cells():
    pred = parse(f"SELECT \"Player\" FROM t WHERE \"Player\" = '42' AND {CLUB} = 'Cincinatti'")
    assert table_checks(pred, GOLD, TABLE) == {"column_by_value", "value_not_in_table"}
