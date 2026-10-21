
import pymupdf
from pathlib import Path


class PageDetector:

    def detect(self, page):

        blocks = page.get_text("blocks",flags=pymupdf.TEXTFLAGS_BLOCKS | pymupdf.TEXT_PRESERVE_IMAGES)

        has_text = False
        has_image = False

        for block in blocks:

            block_type = block[6] 

            if block_type == 0:      # means Text Block
                if block[4].strip():
                    has_text = True

            elif block_type == 1:   # means Image Block
                has_image = True

        if has_text and has_image:
            return "MIXED"

        elif has_text:
            return "TEXT"

        elif has_image:
            return "IMAGE"

        else:
            return "EMPTY"

    def scan(self, pdflocation):

        folder = Path(pdflocation)

        for pdffile in folder.glob("*.pdf"):

            with pymupdf.open(pdffile) as pdf:

                for page_number, page in enumerate(pdf, start=1):

                    page_type = self.detect(page)

                    yield {
                        "document": pdffile.name,
                        "page": page_number,
                        "content_type": page_type
                    }


if __name__ == "__main__":

    detector = PageDetector()

    path = "Data/Source/Beverages/unstructured/SUP-003/"

    for result in detector.scan(path):
        print(result)
