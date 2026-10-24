
import pymupdf
from pathlib import Path

from services.PageDetector import PageDetector
from services.TextExtractor import TextExtractor
from services.EmbeddingService import EmbeddingService
from services.TokenChunker import TokenChunker

class Orchestrator:

    def __init__(self):

        self.detector = PageDetector()
        self.text_extractor = TextExtractor()
        self.embedding_service = EmbeddingService()
        self.chunker = TokenChunker(
        tokenizer=self.embedding_service.model.tokenizer,
        max_tokens=200,
        overlap=30
        )

    def process(self, pdflocation):

        folder = Path(pdflocation)

        for pdffile in folder.glob("*.pdf"):

            with pymupdf.open(pdffile) as pdf:

                for page_number, page in enumerate(pdf, start=1):

                    # Step 1: Detect page type
                    page_type = self.detector.detect(page)

                    # Step 2: Process TEXT pages
                    if page_type == "TEXT":

                        text = self.text_extractor.extract(page)

                        if not text:
                            continue

                        # Split page into token-based chunks
                        for chunk_number, chunk in enumerate(self.chunker.split(text), start=1):

                            embedding = self.embedding_service.embed(
                                chunk["text"]
                            )

                            yield {
                                "document": pdffile.name,
                                "page": page_number,
                                "chunk": chunk_number,
                                "token_count": chunk["token_count"],
                                "text": chunk["text"],
                                "embedding": embedding
                            }


if __name__ == "__main__":

    orchestrator = Orchestrator()

    path = "Data/Source/Beverages/unstructured/SUP-003/"

    for result in orchestrator.process(path):

        print("Document:", result["document"])
        print("Page:", result["page"])
        print("Embedding:", result["embedding"].shape)
