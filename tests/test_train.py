import pytest

from src.params import load_params
from src.train import run_settings


def test_run_settings_merge_defaults_and_run():
    params = load_params()
    s = run_settings(params, "qwen2.5-0.5b-lora")
    assert s["model"].endswith("Qwen2.5-0.5B-Instruct-4bit")
    assert s["mask_prompt"] is True and s["iters"] == params["train"]["defaults"]["iters"]
    with pytest.raises(KeyError):
        run_settings(params, "no-such-run")


class CharTokenizer:
    def apply_chat_template(self, chat, tokenize=False):
        return " ".join(m["content"] for m in chat)

    def encode(self, text, add_special_tokens=False):
        return list(text)


def test_long_examples_are_left_out(tmp_path, monkeypatch):
    import json

    import src.train as tr

    monkeypatch.setattr(tr, "PROCESSED", tmp_path)
    monkeypatch.setattr(tr, "LORA_DATA", tmp_path / "lora")
    rows = [{"messages": [{"role": "user", "content": "x" * n}, {"role": "assistant", "content": "y"}]} for n in (5, 50, 8)]
    for name in ("train.jsonl", "valid.jsonl"):
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    out, counts = tr.fitting_examples(CharTokenizer(), max_tokens=20)
    assert counts["train.jsonl"] == {"kept": 2, "dropped": 1}
    assert len((out / "train.jsonl").read_text().splitlines()) == 2
