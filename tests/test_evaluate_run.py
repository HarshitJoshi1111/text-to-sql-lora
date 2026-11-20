import json

import pytest

import src.evaluate as ev
import src.generate as gen


class EchoGenerator:
    """Answers every second question with the true query and the others with a wrong column."""

    def __init__(self, *args, **kwargs):
        self.calls = 0

    def sql(self, chats):
        out = []
        for chat in chats:
            question = chat[1]["content"].split("Question: ")[1]
            self.calls += 1
            out.append(("raw", ANSWERS[question]))
        return out


ANSWERS = {}


@pytest.mark.skipif(not (ev.PROCESSED / "eval.jsonl").exists(), reason="needs make data")
def test_evaluate_writes_metrics(tmp_path, monkeypatch):
    items = [json.loads(line) for line in open(ev.PROCESSED / "eval.jsonl", encoding="utf-8")][:20]
    for i, q in enumerate(items):
        ANSWERS[q["question"]] = q["sql"] if i % 2 == 0 else 'SELECT "No such column" FROM t'
    monkeypatch.setattr(gen, "Generator", EchoGenerator)
    monkeypatch.setattr(ev, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(ev, "PREDICTIONS", tmp_path / "pred")
    params = ev.load_params()
    result = ev.evaluate("fake", "none", None, params, limit=20)
    assert result["execution_accuracy"] == 0.5 and result["exact_match"] == 0.5
    assert result["outcomes"] == {"correct": 10, "unknown_column": 10}
    assert (tmp_path / "runs" / "fake.json").exists() and len((tmp_path / "pred" / "fake.jsonl").read_text().splitlines()) == 20
