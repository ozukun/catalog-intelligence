import pymupdf
from pydantic import BaseModel
from pathlib import Path
import re

class PDFParser:


    def extract_text(self, pdflocation: str) -> str:
        Path1 = Path(pdflocation)
        for pdffile in Path1.glob("*.pdf"):
            print(pdffile)
            for page in pymupdf.open(str(pdffile)):
                print(page.get_text("text"))



    def extract_chunks(self, pdflocation: str):

        for pdffile in Path(pdflocation).glob("*.pdf"):

            with pymupdf.open(str(pdffile)) as pdf:

                for page_number, page in enumerate(pdf, start=1):

                    blocks = page.get_text("blocks", sort=True)

                    for block_number, block in enumerate(blocks):

                        if block[6] != 0:
                            continue

                        paragraphs = re.split(r"\n\s*\n", block[4])

                        for paragraph_number, paragraph in enumerate(paragraphs):

                            text = " ".join(paragraph.split())

                            if not text:
                                continue

                            yield {
                                "document": pdffile.name,
                                "page": page_number,
                                "block": block_number,
                                "paragraph": paragraph_number,
                                "text": text
                            }


if __name__ == "__main__":
    pdflocation = "Data/Source/Beverages/unstructured/SUP-003/"
    Parser = PDFParser()
    #Parser.extract_text(pdflocation)
 

    
    header_list = (Parser.extract_chunks(pdflocation))

    for h in header_list:
        print(h)