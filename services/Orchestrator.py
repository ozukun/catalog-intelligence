
from pathlib import Path
import pymupdf

from services.PageDetector import PageDetector
from services.TextExtractor import TextExtractor
from services.PageClassificationService import PageClassificationService


class Orchestrator:

    def __init__(self):
        self.detector = PageDetector()
        self.text_extractor = TextExtractor()
        self.classifier = PageClassificationService()

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

                    # Only TEXT pages go to the mini model
                    if page_type == "TEXT":
                        text_pages.append({
                            "page": page_number,
                            "content_type": "TEXT",
                            "text": text
                        })

            # Mini classification: one call per PDF
            page_flags = {}

            if text_pages:

                decisions = self.classifier.classify(
                    pdffile.name,
                    text_pages
                )

                for decision in decisions:
                    page_flags[decision["page"]] = decision["flag"]

            # Decide the final route for every page
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


if __name__ == "__main__":

    orchestrator = Orchestrator()

    path = "Data/Source/Beverages/unstructured/SUP-003/"

    for result in orchestrator.process(path):

        print(
            result["document"],
            "page", result["page"],
            result["content_type"],
            "->", result["flag"],
            "->", result["route"]
        )
