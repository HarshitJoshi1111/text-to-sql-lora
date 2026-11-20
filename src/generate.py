"""Model answers with mlx-lm on Apple silicon: greedy decoding, prompts processed in batches."""
from src.sql_text import extract_sql


class Generator:
    def __init__(self, model: str, adapter: str | None = None, max_tokens: int = 128):
        from mlx_lm import load

        self.model, self.tokenizer = load(model, adapter_path=adapter)
        self.max_tokens = max_tokens

    def prompt_tokens(self, chat: list[dict]) -> list[int]:
        text = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def answer(self, chats: list[list[dict]]) -> list[str]:
        """Raw answers (greedy) for a batch of chats."""
        from mlx_lm import batch_generate

        response = batch_generate(self.model, self.tokenizer, [self.prompt_tokens(c) for c in chats],
                                  max_tokens=self.max_tokens)
        return response.texts

    def sql(self, chats: list[list[dict]]) -> list[tuple[str, str]]:
        return [(text, extract_sql(text)) for text in self.answer(chats)]
