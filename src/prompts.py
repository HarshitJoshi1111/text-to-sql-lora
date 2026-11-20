"""The prompt: table schema, a few example rows and the question; the answer is one SQL query.

The example rows matter: WikiSQL questions often spell values differently from the table
("butler cc (ks)" vs "Butler CC (KS)") or refer to numbers stored as text, and the rows show the model
how the values look.
"""
from src.wikisql import Table, create_statement

SYSTEM = ("You translate questions into SQLite queries. You get the schema of a table named t, a few of "
          "its rows and a question. Answer with one SQL query that answers the question, nothing else.")


def example_rows(table: Table, n: int, max_chars: int) -> str:
    def cell(value) -> str:
        text = " ".join(str(value).split())
        return text if len(text) <= max_chars else text[: max_chars - 3] + "..."
    lines = [" | ".join(table.columns)] + [" | ".join(cell(v) for v in row) for row in table.rows[:n]]
    return "\n".join(lines)


def messages(table: Table, question: str, n_rows: int = 3, max_chars: int = 40, answer: str | None = None) -> list[dict]:
    """Chat messages for one question; with `answer` (the true query) it is a training example."""
    user = (f"{create_statement(table)}\n\n/* {min(n_rows, len(table.rows))} example rows:\n"
            f"{example_rows(table, n_rows, max_chars)}\n*/\n\nQuestion: {question}")
    chat = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    if answer is not None:
        chat.append({"role": "assistant", "content": answer})
    return chat
