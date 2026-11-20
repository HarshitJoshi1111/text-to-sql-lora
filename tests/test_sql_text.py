from src.sql_text import extract_sql, normalize


def test_extract_from_code_fence_and_prose():
    assert extract_sql("```sql\nSELECT a\nFROM t;\n```\nThis query ...") == "SELECT a FROM t"
    assert extract_sql('SQL: SELECT "No." FROM t WHERE x = 1; -- done') == 'SELECT "No." FROM t WHERE x = 1'
    assert extract_sql("Here is the query: SELECT COUNT(*) FROM t") == "SELECT COUNT(*) FROM t"
    assert extract_sql("I cannot answer that.") == "I cannot answer that."


def test_normalize_ignores_quotes_case_and_spacing():
    a = normalize('SELECT COUNT( "Player" ) FROM t WHERE "No." = \'3\';')
    assert a == normalize("""select count(player) from t where "no."='3'""")


def test_normalize_keeps_quotes_that_are_needed():
    assert normalize('SELECT "Laps Led" FROM t') != normalize("SELECT Laps Led FROM t")
    assert normalize("SELECT `Score` FROM t") == normalize('SELECT "Score" FROM t') == normalize("SELECT Score FROM t")
