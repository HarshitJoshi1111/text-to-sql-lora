import pytest

from src.wikisql import SQLError, Table, column_names, create_statement, execute, quote_value, same_result, to_sql

TABLE = Table("1-1", ("Player", "No.", "Height (m)", 'Club "A"'), ("text", "text", "real", "text"),
              (("Antonio Lang", "21", 2.03, "Duke"), ("Kyle Lowry", "3", 1.83, "Villanova"),
               ("Art Long", "42", 2.06, "Cincinnati"), ("John Long", "25", 1.96, "Detroit")))


def test_column_names_are_unique_and_non_empty():
    assert column_names(["Score", "", "score", "Score", " Notes "]) == ("Score", "column 2", "score (2)", "Score (3)", "Notes")


def test_values_are_quoted_by_column_type():
    assert quote_value("3", "text") == "'3'"
    assert quote_value("O'Neal", "text") == "'O''Neal'"
    assert quote_value("1,200", "real") == "1200"
    assert quote_value(2.5, "real") == "2.5"
    assert quote_value("n/a", "real") == "'n/a'"


def test_logical_form_to_sql():
    sql = to_sql({"sel": 0, "agg": 3, "conds": [[1, 0, "3"], [2, 1, 1.9]]}, TABLE)
    assert sql == """SELECT COUNT("Player") FROM t WHERE "No." = '3' AND "Height (m)" > 1.9"""
    assert to_sql({"sel": 3, "agg": 0, "conds": []}, TABLE) == 'SELECT "Club ""A""" FROM t'


def test_execution_is_case_insensitive_for_text():
    assert execute("""SELECT "No." FROM t WHERE "Player" = 'art long'""", TABLE) == [("42",)]
    assert execute("""SELECT MAX("Height (m)") FROM t WHERE "Player" LIKE '%long%'""", TABLE) == [(2.06,)]


def test_numbers_stored_as_text_still_match():
    assert execute("""SELECT "Player" FROM t WHERE "No." = 3""", TABLE) == [("Kyle Lowry",)]


def test_errors_are_raised_as_sql_error():
    with pytest.raises(SQLError, match="no such column"):
        execute('SELECT "Team" FROM t', TABLE)
    with pytest.raises(SQLError):
        execute("SELEC Player FROM t", TABLE)


def test_results_compare_as_multisets():
    assert same_result([("a",), ("B",)], [("b",), ("A",)])
    assert same_result([(3,)], [(3.0,)])
    assert not same_result([("a",)], [("a",), ("a",)])


def test_create_statement_names_the_table_t():
    assert create_statement(TABLE).startswith('CREATE TABLE t (\n  "Player" TEXT,\n  "No." TEXT,\n  "Height (m)" REAL')


def test_double_quotes_inside_text_values_are_not_names():
    table = Table("1-2", ("Title", "Director"), ("text", "text"), (('"Ambush"', "Lee"), ('"Don\'t Stop"', "Kim")))
    assert execute("""SELECT "Director" FROM t WHERE "Title" = '"Ambush"'""", table) == [("Lee",)]
    assert execute("""SELECT "Director" FROM t WHERE "Title" = '"Don''t Stop"'""", table) == [("Kim",)]


def test_quoted_names_handle_both_quote_types():
    from src.wikisql import quoted_names

    assert quoted_names("""SELECT "Men's singles" FROM t WHERE "Women's" = 'O''Neal "x"' AND "a""b" = 1""") == [
        "Men's singles", "Women's", 'a"b']
