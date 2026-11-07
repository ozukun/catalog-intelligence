"""Classify PDF pages for a schema-specific extraction task using a small model.

The service does NOT extract products. It only labels pages KEEP/REVIEW/DROP.
"""

import json
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


class PageDecision(BaseModel):
    page: int
    flag: Literal["KEEP", "REVIEW", "DROP"]


class DocumentDecision(BaseModel):
    pages: list[PageDecision]


class PageClassificationService:
    def __init__(self, model=None, client=None):
        load_dotenv()
        self.model = model or os.getenv("PAGE_CLASSIFIER_MODEL", "gpt-4.1-mini")
        self.client = client if client is not None else OpenAI()

        root = Path(__file__).resolve().parents[1]
        schema_path = root / "context" / "extraction_schema.json"
        with schema_path.open(encoding="utf-8") as file:
            schema = json.load(file)

        self.entity = schema["entity"]
        self.domain = schema["domain"]
        self.field_descriptions = []
        for name, definition in schema["attributes"].items():
            self.field_descriptions.append(
                f"{name}: {definition['description']}"
            )

    def classify(self, document: str, pages: list[dict]) -> list[dict]:
        """Analyze one document in one model call; return each page with its flag.

        Expected page input: {"page": 1, "content_type": "TEXT", "text": "..."}
        Pages without readable text are marked REVIEW, never silently discarded.
        """
        if not pages:
            return []

        page_payload = []
        has_readable_text = False
        for page in pages:
            text = page["text"].strip()
            if text:
                has_readable_text = True
            page_payload.append({
                "page": page["page"],
                "content_type": page["content_type"],
                "text": text,
            })

        flags = {}
        if has_readable_text:
            instructions = (
                f"You classify PDF pages for {self.domain} {self.entity} data extraction. "
                "Your task is ONLY page selection, not product extraction. "
                "Read the entire document, including cross-page context. "
                "A page is relevant if it could provide evidence for ANY of these fields:\n"
                + "\n".join(self.field_descriptions)
                + "\nKEEP: Contains identifiable product names, product-specific facts, "
                  "or related information whose meaning depends on neighboring pages. "
                  "REVIEW: Uncertain relevance, possible continuation, or not enough readable text. "
                  "DROP: Clearly unrelated to the requested product extraction task, "
                  "including generic administrative/parking information and boilerplate "
                  "without product-specific information. "
                  "If in doubt, choose REVIEW instead of DROP. "
                  "Respond with exactly one page number and flag for EVERY page. "
                  "Do not extract products, SKUs, or attributes."
            )

            response = self.client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": json.dumps({
                        "document": document,
                        "pages": page_payload,
                    }, ensure_ascii=False)},
                ],
                text_format=DocumentDecision,
            )

            if response.output_parsed is None:
                raise RuntimeError(f"Could not classify document: {document}")

            valid_numbers = {page["page"] for page in pages}
            for decision in response.output_parsed.pages:
                if decision.page not in valid_numbers:
                    continue
                if decision.page in flags and flags[decision.page] != decision.flag:
                    flags[decision.page] = "REVIEW"
                else:
                    flags[decision.page] = decision.flag

        results = []
        for page in pages:
            flag = flags.get(page["page"], "REVIEW")
            if not page["text"].strip() or page["content_type"] != "TEXT":
                # The text-only mini model cannot reliably reject image content.
                if flag == "DROP":
                    flag = "REVIEW"

            results.append({
                "document": document,
                "page": page["page"],
                "content_type": page["content_type"],
                "flag": flag,
                "text": page["text"],
            })

        return results
