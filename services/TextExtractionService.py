
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from services.StructuredExtractionModels import ExtractionResult


class TextExtractionService:

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

        selected = []

        for page in pages:
            if (
                page["content_type"] == "TEXT"
                and page["route"] == "LARGE_MODEL"
            ):
                selected.append(page)

        if not selected:
            return None

        page_text = ""

        for page in selected:
            page_text += (
                f"\n--- PAGE {page['page']} ---\n"
                f"{page['text']}\n"
            )

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
