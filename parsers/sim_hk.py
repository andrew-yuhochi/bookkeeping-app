"""SIM Hong Kong credit card statement parser.

Extracts transactions from SIM HK (sim World MasterCard) PDF statements
using pdfplumber text extraction and line-based regex parsing.

Statement layout (observed from real Jan/Feb/Mar 2026 samples):
  - Page 1: Header, address, card number, then transactions
  - Page 2: Usually empty (footer only)
  - Transactions are between the "Trans Date" header and "GRAND TOTAL"

Transaction format (purchases — always foreign currency with inline FX):
  Line 1: MERCHANT_NAME                         (merchant, may be truncated)
  Line 2: DD Mon LOCATION COUNTRY CURRENCY FOREIGN_AMT HKD_AMT
  Line 3: *EXCHANGE RATE:X.XXXXX

Credit format (payments, cashback — single line):
  DD Mon DESCRIPTION AMOUNT CR

Special lines (skipped):
  - "Previous Balance XXXXX"
  - "sim World MasterCard® - 521263..."
  - "GRAND TOTAL"
  - "End of Statement"
"""

import logging
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from io import BytesIO

import pdfplumber

from parsers.base import IssuerParser
from parsers.models import ParsedTransaction

logger = logging.getLogger(__name__)

_MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

# Purchase line: DD Mon LOCATION COUNTRY CURRENCY FOREIGN_AMT HKD_AMT
_PURCHASE_RE = re.compile(
    r"^(\d{1,2})\s+"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+"
    r"(.+?)\s+"
    r"([A-Z]{3})\s+"            # currency code (CAD, JPY, USD, etc.)
    r"([\d,]+\.\d{2})\s+"       # foreign amount
    r"([\d,]+\.\d{2})$"         # HKD amount
)

# Credit line: DD Mon DESCRIPTION AMOUNT CR
_CREDIT_RE = re.compile(
    r"^(\d{1,2})\s+"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+"
    r"(.+?)\s+"
    r"([\d,]+\.\d{2})\s+CR$"
)

# Exchange rate line: *EXCHANGE RATE:X.XXXXX
_FX_RE = re.compile(r"^\*EXCHANGE RATE:([\d.]+)$")

# Statement date: "22 Jan 2026"
_STMT_DATE_RE = re.compile(
    r"(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})"
)

# Lines to skip
_SKIP_PATTERNS = [
    "Previous Balance",
    "sim World MasterCard",
    "sim Credit Card",
    "GRAND TOTAL",
    "End of Statement",
    "Annualised Percentage Rate",
    "Retail Purchase",
    "Cash Advance",
    "Please examine",
    "請即查核",
    "Page ",
]


def _parse_amount(s: str) -> Decimal:
    """Parse a formatted amount string like '1,308.98' into a Decimal."""
    return Decimal(s.replace(",", ""))


def _resolve_year(day: int, month_str: str, statement_year: int, statement_month: int) -> int:
    """Resolve the transaction year from DD Mon format.

    SIM HK statements only show DD Mon (no year). Transactions in the
    statement can span Dec of the previous year through the statement month.
    """
    month_num = _MONTHS[month_str]
    # If the transaction month is much larger than the statement month,
    # it's from the previous year (e.g., Dec transaction on a Jan statement)
    if month_num > statement_month + 2:
        return statement_year - 1
    return statement_year


class SIMHKParser(IssuerParser):
    """Parser for SIM Hong Kong (sim World MasterCard) PDF statements."""

    issuer_name: str = "SIM_HK"

    def detect(self, filename: str, first_page_text: str) -> bool:
        """Detect SIM HK statements by filename or first-page text."""
        fname_lower = filename.lower()
        # Filename heuristics: "sim" in name, or Chinese monthly statement marker
        if "sim" in fname_lower or "月結單" in filename:
            return True
        # Text signature: "sim World MasterCard" or "sim Credit Card"
        if "sim World MasterCard" in first_page_text or "sim Credit Card" in first_page_text:
            return True
        return False

    def parse(self, pdf_bytes: bytes) -> list[ParsedTransaction]:
        """Extract transactions from SIM HK PDF bytes."""
        transactions: list[ParsedTransaction] = []

        try:
            pdf = pdfplumber.open(BytesIO(pdf_bytes))
        except Exception:
            logger.exception("SIM HK parser: failed to open PDF")
            return []

        with pdf:
            # Determine statement date from page 1 for year resolution
            stmt_year, stmt_month = self._extract_statement_date(pdf.pages[0])

            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    page_txns = self._parse_page(page, page_num, stmt_year, stmt_month)
                    transactions.extend(page_txns)
                except Exception:
                    logger.exception(
                        "SIM HK parser failed on page %d", page_num,
                    )
                    # Continue with other pages

        logger.info(
            "SIM HK parser: extracted %d transactions", len(transactions),
        )
        return transactions

    def _extract_statement_date(self, page: "pdfplumber.page.Page") -> tuple[int, int]:
        """Extract statement year and month from page 1 header."""
        text = page.extract_text() or ""
        # Look for "Statement Date" section — the date follows nearby
        # Format: "22 Jan 2026" appearing near "Statement Date"
        for line in text.split("\n"):
            if "Statement Date" in line or "月結單截數日" in line:
                # The date might be on this line or a nearby one
                m = _STMT_DATE_RE.search(line)
                if m:
                    return int(m.group(3)), _MONTHS[m.group(2)]

        # Fallback: find any date pattern on page 1
        m = _STMT_DATE_RE.search(text)
        if m:
            return int(m.group(3)), _MONTHS[m.group(2)]

        # Last resort: use current year
        today = date.today()
        logger.warning("Could not determine SIM HK statement date, using current year")
        return today.year, today.month

    def _parse_page(
        self,
        page: "pdfplumber.page.Page",
        page_num: int,
        stmt_year: int,
        stmt_month: int,
    ) -> list[ParsedTransaction]:
        """Parse transactions from a single page."""
        text = page.extract_text()
        if not text:
            return []

        lines = text.split("\n")
        transactions: list[ParsedTransaction] = []
        pending_merchant: str | None = None
        in_transaction_section = False

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines
            if not line:
                i += 1
                continue

            # Detect start of transaction section
            if "Trans Date" in line or "交易日期" in line:
                in_transaction_section = True
                i += 1
                continue

            # Detect end of transaction section
            if "GRAND TOTAL" in line or "End of Statement" in line:
                in_transaction_section = False
                i += 1
                continue

            if not in_transaction_section:
                i += 1
                continue

            # Skip known non-transaction lines
            if any(pat in line for pat in _SKIP_PATTERNS):
                pending_merchant = None
                i += 1
                continue

            # Try to match a purchase line
            m_purchase = _PURCHASE_RE.match(line)
            if m_purchase:
                day = int(m_purchase.group(1))
                month_str = m_purchase.group(2)
                # group(3) is location+country — we don't need it separately
                currency = m_purchase.group(4)
                foreign_amount = _parse_amount(m_purchase.group(5))
                hkd_amount = _parse_amount(m_purchase.group(6))

                year = _resolve_year(day, month_str, stmt_year, stmt_month)
                cash_date = date(year, _MONTHS[month_str], day)

                # Check for exchange rate on next line
                fx_rate: Decimal | None = None
                if i + 1 < len(lines):
                    m_fx = _FX_RE.match(lines[i + 1].strip())
                    if m_fx:
                        fx_rate = Decimal(m_fx.group(1))
                        i += 1  # consume the FX line

                description = pending_merchant or line
                pending_merchant = None

                transactions.append(ParsedTransaction(
                    issuer="SIM_HK",
                    cash_date=cash_date,
                    description=description,
                    original_amount=foreign_amount,
                    original_currency=currency,
                    fx_rate=fx_rate,
                    fx_rate_source="statement" if fx_rate else None,
                    cad_amount=None,  # resolved later by TransactionNormalizer
                    statement_page=page_num,
                ))

                i += 1
                continue

            # Try to match a credit line (payments, cashback)
            m_credit = _CREDIT_RE.match(line)
            if m_credit:
                day = int(m_credit.group(1))
                month_str = m_credit.group(2)
                description = m_credit.group(3)
                hkd_amount = _parse_amount(m_credit.group(4))

                year = _resolve_year(day, month_str, stmt_year, stmt_month)
                cash_date = date(year, _MONTHS[month_str], day)

                pending_merchant = None

                transactions.append(ParsedTransaction(
                    issuer="SIM_HK",
                    cash_date=cash_date,
                    description=description,
                    original_amount=-hkd_amount,  # negative for credits
                    original_currency="HKD",
                    fx_rate=Decimal("1"),
                    fx_rate_source="statement",
                    cad_amount=None,
                    statement_page=page_num,
                ))

                i += 1
                continue

            # If none of the above matched, this line is likely a merchant name
            # for the next purchase transaction
            pending_merchant = line
            i += 1

        return transactions
