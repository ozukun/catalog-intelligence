import pymupdf
from pydantic import BaseModel
from pathlib import Path

class PDFParser:


    def extract_text(self, pdflocation: str) -> str:
        Path1 = Path(pdflocation)
        for pdffile in Path1.glob("*.pdf"):
            print(pdffile)
            for page in pymupdf.open(str(pdffile)):
                print(page.get_text("text"))



if __name__ == "__main__":
    pdflocation = "Data/Source/Beverages/unstructured/SUP-003/"
    Parser = PDFParser()
    Parser.extract_text(pdflocation)
 


    