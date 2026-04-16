from datetime import date
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel


class ParsedTransaction(BaseModel):
    """Raw transaction extracted by an issuer parser.

    All fields match the TDD Section 3 data contract.
    """

    issuer: str
    cash_date: date
    description: str
    original_amount: Decimal
    original_currency: str
    fx_rate: Optional[Decimal] = None
    fx_rate_source: Optional[str] = None
    cad_amount: Optional[Decimal] = None
    statement_page: int
