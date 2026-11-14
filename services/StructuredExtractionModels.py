
from typing import Literal
from pydantic import BaseModel


class Product(BaseModel):
    product_name: str
    brand: str | None
    supplier_sku: str | None
    package_size_ml: int | None
    package_type: Literal[
        "glass_bottle",
        "pet_bottle",
        "aluminum_can",
        "carton",
        "other"
    ] | None
    source_page: int
    source_evidence: str


class ExtractionResult(BaseModel):
    products: list[Product]
