
import json
import pymupdf
from pathlib import Path

from services.PageDetector import PageDetector
from services.TextExtractor import TextExtractor
from services.PageClassificationService import PageClassificationService
from services.TextExtractionService import TextExtractionService


class Orchestrator:

    def __init__(self):

        self.detector = PageDetector()
        self.text_extractor = TextExtractor()
        self.classifier = PageClassificationService()
        self.extractor = TextExtractionService()


    # STEP 1: PDF Reading + Page Classification + Routing
    def process(self, pdflocation):

        folder = Path(pdflocation)

        for pdffile in sorted(folder.glob("*.pdf")):

            pages = []
            text_pages = []

            with pymupdf.open(pdffile) as pdf:

                for page_number, page in enumerate(pdf, start=1):

                    page_type = self.detector.detect(page)
                    text = ""

                    if page_type in ("TEXT", "MIXED"):
                        text = self.text_extractor.extract(page)

                    page_data = {
                        "document": pdffile.name,
                        "pdf_path": str(pdffile.resolve()),
                        "page": page_number,
                        "content_type": page_type,
                        "text": text
                    }

                    pages.append(page_data)

                    # Only TEXT pages go to mini model
                    if page_type == "TEXT":

                        text_pages.append({
                            "page": page_number,
                            "content_type": page_type,
                            "text": text
                        })

            # Mini model classification
            page_flags = {}

            if text_pages:

                decisions = self.classifier.classify(
                    pdffile.name,
                    text_pages
                )

                for decision in decisions:
                    page_flags[decision["page"]] = decision["flag"]

            # Routing decisions
            for page_data in pages:

                page_type = page_data["content_type"]

                if page_type in ("IMAGE", "MIXED"):

                    flag = "KEEP"
                    route = "LARGE_MODEL"

                elif page_type == "TEXT":

                    flag = page_flags.get(
                        page_data["page"],
                        "REVIEW"
                    )

                    if flag not in ("KEEP", "REVIEW", "DROP"):
                        flag = "REVIEW"

                    if flag == "DROP":
                        route = "SKIP"
                    else:
                        route = "LARGE_MODEL"

                else:

                    flag = "DROP"
                    route = "SKIP"

                yield {
                    **page_data,
                    "flag": flag,
                    "route": route
                }


    # STEP 2: Large Model Structured Extraction
    def run(self, pdflocation):

        current_pdf = None
        document_pages = []

        for page in self.process(pdflocation):

            pdf_path = page["pdf_path"]

            # Previous PDF finished
            if current_pdf is not None and pdf_path != current_pdf:

                result = self.extractor.extract_text(document_pages)

                if result is not None:
                    yield result

                document_pages = []

            current_pdf = pdf_path
            document_pages.append(page)

        # Process the last PDF
        if document_pages:

            result = self.extractor.extract_text(document_pages)

            if result is not None:
                yield result


if __name__ == "__main__":

    orchestrator = Orchestrator()

    path = "Data/Source/Beverages/unstructured/SUP-003/"

    for result in orchestrator.run(path):

        print(json.dumps(
            result,
            indent=4,
            ensure_ascii=False
        ))
