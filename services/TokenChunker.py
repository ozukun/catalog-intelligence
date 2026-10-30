
class TokenChunker:

    def __init__(self, tokenizer, max_tokens=50, overlap=30):

        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.overlap = overlap

        if max_tokens <= overlap or overlap < 0:
            raise ValueError("Invalid chunk settings")

    def split(self, text):

        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False
        )

        step = self.max_tokens - self.overlap

        for i in range(0, len(tokens), step):

            chunk_tokens = tokens[i:i + self.max_tokens]

            chunk_text = self.tokenizer.decode(
                chunk_tokens,
                skip_special_tokens=True
            )

            yield {
                "text": chunk_text,
                "token_count": len(chunk_tokens)
            }

            if i + self.max_tokens >= len(tokens):
                break