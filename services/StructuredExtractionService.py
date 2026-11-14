
import json
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


class Product(BaseModel):
    product_name: str
    brand: str | None
    supplier_sku: str | None
    package_size_ml: int | None
    package_type: Literal[
        "glass_bottle", "pet_bottle",
        "aluminum_can", "carton", "other"
    ] | None
    source_page: int
    source_evidence: str


class ExtractionResult(BaseModel):
    products: list[Product]


class StructuredExtractionService:

    def __init__(self):
        load_dotenv()
        root = Path(__file__).resolve().parents[1]
        self.client = OpenAI()
        self.model = os.getenv("EXTRACTION_MODEL", "gpt-4.1")
        self.prompt_path = root / "context" / "product_extraction_prompt.txt"
        schema_path = root / "context" / "extraction_schema.json"

        with schema_path.open(encoding="utf-8") as file:
            self.schema = json.load(file)

    def extract_text(self, pages: list[dict]) -> dict | None:

        # Sadece seçilmiş TEXT sayfalarını al
        selected = []

        for page in pages:
            if (
                page["content_type"] == "TEXT"
                and page["route"] == "LARGE_MODEL"
            ):
                selected.append(page)

        if not selected:
            return None

        # Sayfa metinlerini birleştir
        page_text = ""

        for page in selected:
            page_text += (
                f"\n--- PAGE {page['page']} ---\n"
                f"{page['text']}\n"
            )

        # Extraction schema alanlarını hazırla
        field_lines = []

        for name, definition in self.schema["attributes"].items():
            description = definition["description"]

            if "allowed_values" in definition:
                description += (
                    "; allowed values: "
                    + ", ".join(definition["allowed_values"])
                )

            field_lines.append(f"- {name}: {description}")

        instructions = self.prompt_path.read_text(
            encoding="utf-8"
        )

        instructions = instructions.replace(
            "{{FIELDS}}",
            "\n".join(field_lines)
        )

        # Büyük model çağrısı
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": page_text}
            ],
            text_format=ExtractionResult
        )

        if response.output_parsed is None:
            raise RuntimeError("Structured extraction failed")

        products = []

        for product in response.output_parsed.products:
            products.append(product.model_dump())

        usage = response.usage

        return {
            "document": selected[0]["document"],
            "products": products,
            "usage": {
                "input_tokens": usage.input_tokens if usage else None,
                "output_tokens": usage.output_tokens if usage else None,
                "total_tokens": usage.total_tokens if usage else None
            }
        }

    def extract_mixed(self, pages: list[dict]):
        raise NotImplementedError("MIXED extraction not implemented yet")

    def extract_image(self, pages: list[dict]):
        raise NotImplementedError("IMAGE extraction not implemented yet")
