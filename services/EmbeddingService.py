from sentence_transformers import SentenceTransformer

class EmbeddingService:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, text):
        return self.model.encode(text)


if __name__ == "__main__":
    embedding_service = EmbeddingService()

    text = "This is a sample text for embedding."
    embedding = embedding_service.embed(text)

