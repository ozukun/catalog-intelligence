
from services.EmbeddingService import EmbeddingService
from services.extraction_schema_loader import schema_loader
from services.CosineSimilarityService import CosineSimilarityService

class ChunkRelevanceService:

    def __init__(self):

        self.embedding_service = EmbeddingService()
        self.similarity_service = CosineSimilarityService()

        loader = schema_loader()
        schema = loader.load_schema(
            "context/extraction_schema.json"
        )

        terms = schema["attributes"]["product_name"]["semantic_terms"]

        self.term_embeddings = {}

        for term in terms:
            embedding = self.embedding_service.embed(term)
            self.term_embeddings[term] = embedding



    def compare(self, chunk_embedding):

        scores = {}

        for term, term_embedding in self.term_embeddings.items():

            similarity = self.similarity_service.compare(
                chunk_embedding,
                term_embedding
            )

            scores[term] = similarity

        max_score = max(scores.values())

        average_score = sum(scores.values()) / len(scores)

        return {
            "scores": scores,
            "max_score": max_score,
            "average_score": average_score
        }


if __name__ == "__main__":

    service = ChunkRelevanceService()

    text = """
Baltic Beverages presents its 2026 product catalog featuring a wide range
of ready-to-drink beverages. The product assortment includes Nordic Brew
Oat Latte, a premium oat-based coffee drink available in 250 ml aluminum
cans under supplier SKU BD-OATLATTE-250. Another product in the portfolio
is Crystal Berry Sparkle, a refreshing carbonated berry beverage supplied
in 500 ml PET bottles. The commercial product names, item descriptions,
brand information, article numbers, and product specifications are
provided to support wholesale ordering and product identification.
"""

    chunk_embedding = service.embedding_service.embed(text)

    result = service.compare(chunk_embedding)

    for term, score in result["scores"].items():
        print(term, round(score, 3))

    print("Max Score:", round(result["max_score"], 3))
    print("Average Score:", round(result["average_score"], 3))

