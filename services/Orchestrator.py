
import pymupdf
from pathlib import Path

from services.PageDetector import PageDetector
from services.TextExtractor import TextExtractor
from services.EmbeddingService import EmbeddingService


class Orchestrator:

    def __init__(self):

        self.detector = PageDetector()
        self.text_extractor = TextExtractor()
        self.embedding_service = EmbeddingService()

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

                        # Step 3: Generate embedding
                        embedding = self.embedding_service.embed(text)

                        yield {
                            "document": pdffile.name,
                            "page": page_number,
                            "content_type": page_type,
                            "text": text,
                            "embedding": embedding
                        }


if __name__ == "__main__":

    orchestrator = Orchestrator()

    path = "Data/Source/Beverages/unstructured/SUP-003/"

    for result in orchestrator.process(path):

        print("Document:", result["document"])
        print("Page:", result["page"])
        print("Embedding:", result["embedding"].shape)
