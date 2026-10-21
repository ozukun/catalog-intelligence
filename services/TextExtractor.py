
import pymupdf
from pathlib import Path
from services.PageDetector import PageDetector


class TextExtractor:

    def extract(self, page: pymupdf.Page) -> str:

        text = page.get_text("text", sort=True)

        return text.strip()


if __name__ == "__main__":

    pdflocation = "Data/Source/Beverages/unstructured/SUP-003/"

    detector = PageDetector()
    extractor = TextExtractor()

    folder = Path(pdflocation)

    for pdffile in folder.glob("*.pdf"):

        with pymupdf.open(pdffile) as pdf:

            for page_number, page in enumerate(pdf, start=1):

                page_type = detector.detect(page)

                if page_type == "TEXT" or page_type == "MIXED":

                    text = extractor.extract(page)

                    if not text:
                        continue

                    result = {
                        "document": pdffile.name,
                        "page": page_number,
                        "content_type": page_type,
                        "text": text
                    }

                    print(result)
