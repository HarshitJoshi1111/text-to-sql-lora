from src.evaluate import diagnose, score
from src.prompts import messages
from tests.test_wikisql import TABLE

ITEM = {"sql": """SELECT "Player" FROM t WHERE "No." = '42'"""}


def test_score_outcomes():
    assert score(ITEM, """SELECT Player FROM t WHERE "No." = 42""", TABLE)["status"] == "correct"
    assert score(ITEM, """SELECT "Club" FROM t""", TABLE)["status"] == "unknown_column"
    assert score(ITEM, "SELECT FROM", TABLE)["status"] == "syntax"
    assert score(ITEM, """SELECT "Player" FROM t WHERE "No." = '21'""", TABLE)["status"] == "wrong_where"
    assert score(ITEM, "", TABLE)["status"] == "no_sql"


def test_diagnose_parts():
    gold = """SELECT "Player" FROM t WHERE "No." = '42'"""
    assert diagnose("""SELECT COUNT("Player") FROM t WHERE "No." = '42'""", gold) == "select"
    assert diagnose("""SELECT COUNT("Player") FROM t""", gold) == "both"
    assert diagnose("SELECT a FROM t ORDER BY a LIMIT 1", gold) == "other"  # not the WikiSQL shape


def test_prompt_contains_schema_rows_and_question():
    chat = messages(TABLE, "Who wears number 42?", n_rows=2, max_chars=8)
    user = chat[1]["content"]
    assert 'CREATE TABLE t (' in user and "Question: Who wears number 42?" in user
    assert "Anton..." in user and "Art Long" not in user  # 2 rows, long cells cut
    assert len(messages(TABLE, "q", answer="SELECT 1")) == 3
