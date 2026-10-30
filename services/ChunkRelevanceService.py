
from services.EmbeddingService import EmbeddingService
from services.extraction_schema_loader import schema_loader


class ChunkRelevanceService:

    def __init__(self):

        self.embedding_service = EmbeddingService()

        loader = schema_loader()
        schema = loader.load_schema(
            "context/extraction_schema.json"
        )

        terms = schema["attributes"]["product_name"]["semantic_terms"]

        self.term_embeddings = {}

        for term in terms:
            embedding = self.embedding_service.embed(term)
            self.term_embeddings[term] = embedding


if __name__ == "__main__":

    service = ChunkRelevanceService()

    for term, embedding in service.term_embeddings.items():
        print(term, embedding.shape)
