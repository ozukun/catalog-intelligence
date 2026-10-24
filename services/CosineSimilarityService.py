
import numpy as np


class CosineSimilarityService:

    def compare(self, vector1, vector2):

        dot_product = np.dot(vector1, vector2)

        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        similarity = dot_product / (magnitude1 * magnitude2)

        return float(similarity)

if __name__ == "__main__":

    from services.EmbeddingService import EmbeddingService

    service = CosineSimilarityService()
    embedding_service = EmbeddingService()

    text1 = "Supplier SKU code is BD-OATLATTE-250."
    text2 = "The beverage comes in 250 milliliter cans."
    text3 = "Employees must register their parking spaces."

    vector1 = embedding_service.embed(text1)
    vector2 = embedding_service.embed(text2)
    vector3 = embedding_service.embed(text3)

    print("Similar texts:", service.compare(vector1, vector2))
    print("Different texts:", service.compare(vector1, vector3))


    from services.extraction_schema_loader import schema_loader

    loader = schema_loader()
    schema = loader.load_schema("context/extraction_schema.json")

    for attribute_name, definition in schema["attributes"].items():

        attribute_text = (
            attribute_name.replace("_", " ")
            + " "
            + definition["description"]
            + " "
            + " ".join(definition["semantic_terms"])
        )

        attribute_embedding = embedding_service.embed(attribute_text)

        similarity = service.compare(vector1, attribute_embedding)

        print(attribute_name, round(similarity, 3))