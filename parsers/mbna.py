"""MBNA Canada credit card statement parser.

Extracts transactions from MBNA PDF statements using pdfplumber text extraction
and line-based regex parsing.

Statement layout (observed from real samples):
  - Page 1: Summary (balance, payment info, rewards)
  - Page 2: Legal boilerplate ("Understanding your account")
  - Pages 3+: Transaction details

Transaction line format:
  MM/DD/YY MM/DD/YY DESCRIPTION REFNUM $AMOUNT
  or
  MM/DD/YY MM/DD/YY DESCRIPTION REFNUM -$AMOUNT (credits/refunds)

Sections: PAYMENTS, PURCHASES, PURCHASES(continued)
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

# Regex to match a transaction line:
#   Trans Date  Posting Date  Description...  RefNum  Amount
# Example: 02/21/26 02/23/26 BIGCRAZYSUPERMARKET RICHMOND BC 6746 $27.46
# Example: 01/30/26 02/02/26 OSAKA#005 RICHMOND BC 8020 -$4.31
_TXN_LINE_RE = re.compile(
    r"^(\d{2}/\d{2}/\d{2})\s+"  # Trans date (MM/DD/YY)
    r"(\d{2}/\d{2}/\d{2})\s+"   # Posting date (MM/DD/YY)
    r"(.+?)\s+"                  # Description (greedy but minimal to leave room for ref+amount)
    r"(\d{4})\s+"                # Reference number (4 digits)
    r"(-?\$[\d,]+\.\d{2})\s*$"  # Amount ($X.XX or -$X.XX)
)

# Statement period pattern from page 1:
# "StatementPeriod:" or "Statement Period:" followed by dates
_PERIOD_RE = re.compile(
    r"StatementPeriod:\s*(\d{2}/\d{2}/\d{2})\s*to\s*(\d{2}/\d{2}/\d{2})",
    re.IGNORECASE,
)

# Lines to skip (section headers, totals, summary)
_SKIP_PATTERNS = [
    "Previousstatementbalance",
    "Previous statement balance",
    "PAYMENTS",
    "PURCHASES",
    "Total ",
    "Total$",
    "SubtotalofActivity",
    "Subtotal of Activity",
    "NewBalance",
    "New Balance",
    "continuedonnextpage",
    "continued on next page",
    "Details of your transactions",
    "Page",
    "Trans Posting",
    "Date Date Description",
]


def _parse_date(date_str: str) -> date:
    """Parse MM/DD/YY date string to a date object.

    Assumes 2000s century for YY < 80, 1900s for YY >= 80.
    """
    month, day, year_2d = date_str.split("/")
    year_int = int(year_2d)
    full_year = 2000 + year_int if year_int < 80 else 1900 + year_int
    return date(full_year, int(month), int(day))


def _parse_amount(amount_str: str) -> Decimal:
    """Parse $1,127.95 or -$4.31 to Decimal."""
    cleaned = amount_str.replace("$", "").replace(",", "")
    return Decimal(cleaned)


class MBNAParser(IssuerParser):
    """Parser for MBNA Canada credit card PDF statements."""

    issuer_name: str = "MBNA"

    def detect(self, filename: str, first_page_text: str) -> bool:
        """Detect MBNA statements by filename or first-page text signature."""
        filename_lower = filename.lower()
        if "mbna" in filename_lower:
            return True
        if "mbna" in first_page_text.lower() and "credit card account statement" in first_page_text.lower():
            return True
        return False

    def parse(self, pdf_bytes: bytes) -> list[ParsedTransaction]:
        """Extract transactions from an MBNA PDF statement.

        Skips page 1 (summary) and page 2 (legal). Parses pages 3+
        for transaction lines. Handles partial failures per page.
        """
        transactions: list[ParsedTransaction] = []

        try:
            pdf = pdfplumber.open(BytesIO(pdf_bytes))
        except Exception:
            logger.exception("Failed to open PDF")
            return transactions

        try:
            for page_idx, page in enumerate(pdf.pages):
                page_num = page_idx + 1

                # Skip summary (page 1) and legal (page 2)
                if page_num <= 2:
                    continue

                try:
                    text = page.extract_text()
                    if not text:
                        logger.warning("Page %d: no text extracted", page_num)
                        continue

                    page_txns = self._parse_page_text(text, page_num)
                    transactions.extend(page_txns)

                except Exception:
                    logger.exception(
                        "Page %d: parse error — continuing with remaining pages",
                        page_num,
                    )
        finally:
            pdf.close()

        logger.info(
            "MBNA parser: extracted %d transactions from %d pages",
            len(transactions), len(pdf.pages),
        )
        return transactions

    def _parse_page_text(self, text: str, page_num: int) -> list[ParsedTransaction]:
        """Parse transaction lines from a single page's text."""
        results: list[ParsedTransaction] = []
        in_payment_section = False

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Track section headers
            if line.startswith("PAYMENTS"):
                in_payment_section = True
                continue
            if line.startswith("PURCHASES"):
                in_payment_section = False
                continue

            # Skip known non-transaction lines
            if self._should_skip(line):
                continue

            # Try to match a transaction line
            match = _TXN_LINE_RE.match(line)
            if not match:
                continue

            trans_date_str = match.group(1)
            _posting_date_str = match.group(2)
            description = match.group(3).strip()
            _ref_num = match.group(4)
            amount_str = match.group(5)

            try:
                cash_date = _parse_date(trans_date_str)
                amount = _parse_amount(amount_str)
            except (ValueError, InvalidOperation) as e:
                logger.warning(
                    "Page %d: failed to parse date/amount in line: %s (%s)",
                    page_num, line, e,
                )
                continue

            # Skip payment lines (they appear in the PAYMENTS section)
            if in_payment_section:
                continue

            # Negative amounts are credits/refunds — preserve sign
            txn = ParsedTransaction(
                issuer="MBNA",
                cash_date=cash_date,
                description=description,
                original_amount=amount,
                original_currency="CAD",
                fx_rate=None,
                fx_rate_source=None,
                cad_amount=amount,
                statement_page=page_num,
            )
            results.append(txn)

        return results

    def _should_skip(self, line: str) -> bool:
        """Check if a line is a known non-transaction line."""
        for pattern in _SKIP_PATTERNS:
            if pattern in line:
                return True
        return False
