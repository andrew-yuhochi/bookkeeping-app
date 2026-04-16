"""Wealthsimple statement parser — banking (chequing) and credit card.

Detects statement type from first-page text and branches to the
appropriate extraction path.

Banking layout:
  - Date format: YYYY-MM-DD
  - Columns: DATE | POSTED DATE | DESCRIPTION | AMOUNT (CAD) | BALANCE (CAD)
  - Multi-line descriptions (bill payments wrap)

Credit card layout:
  - Date format: Mon DD (short month + day)
  - Columns: TRANS. DATE | POSTED DATE | TYPE | DETAILS | AMOUNT ($CAD)
  - FX conversion on next line: "3,755.55 HKD • 0.178637 exchange rate"
  - Types: Purchase, Payment, Cash withdrawal, Interest charge
"""

import logging
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from io import BytesIO
from typing import Optional

import pdfplumber

from parsers.base import IssuerParser
from parsers.models import ParsedTransaction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Month name → number lookup
_MONTH_MAP: dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_amount(s: str) -> Decimal:
    """Parse an amount string like '$1,127.95', '–$204.50', '-$4.31'."""
    cleaned = s.replace("$", "").replace(",", "").replace("\u2013", "-").replace("–", "-").strip()
    return Decimal(cleaned)


# ---------------------------------------------------------------------------
# Banking (chequing) parser
# ---------------------------------------------------------------------------

# Banking transaction line: YYYY-MM-DD YYYY-MM-DD Description Amount Balance
# Example: 2026-02-01 2026-02-01 Bank Of Montreal –$204.50 $4,390.14
_BANKING_TXN_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})\s+"          # Transaction date
    r"(\d{4}-\d{2}-\d{2})\s+"           # Posted date
    r"(.+?)\s+"                          # Description
    r"([–\-]?\$[\d,]+\.\d{2})\s+"       # Amount (with en-dash or hyphen)
    r"[–\-]?\$[\d,]+\.\d{2}\s*$"        # Balance (captured but not used)
)


def _parse_banking_pages(pages: list, all_text_pages: list[str]) -> list[ParsedTransaction]:
    """Parse banking (chequing) statement pages."""
    transactions: list[ParsedTransaction] = []
    pending_description: Optional[str] = None
    pending_line_parts: Optional[tuple] = None

    for page_idx, text in enumerate(all_text_pages):
        page_num = page_idx + 1
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip header/footer lines
            if _is_banking_skip_line(line):
                continue

            # Try to match a transaction line
            match = _BANKING_TXN_RE.match(line)
            if match:
                # Flush any pending multi-line description
                if pending_line_parts:
                    txn = _build_banking_txn(pending_line_parts, pending_description or "", page_num)
                    if txn:
                        transactions.append(txn)

                trans_date_str = match.group(1)
                posted_date_str = match.group(2)
                description = match.group(3).strip()
                amount_str = match.group(4)

                pending_line_parts = (trans_date_str, posted_date_str, amount_str)
                pending_description = description
            elif pending_line_parts and not line.startswith("DATE") and not line.startswith("Page "):
                # Continuation of a multi-line description
                if not line.startswith("Date is when") and not line.startswith("Wealthsimple"):
                    pending_description = (pending_description or "") + " " + line

    # Flush last pending
    if pending_line_parts:
        txn = _build_banking_txn(pending_line_parts, pending_description or "", len(all_text_pages))
        if txn:
            transactions.append(txn)

    return transactions


def _build_banking_txn(
    parts: tuple, description: str, page_num: int
) -> Optional[ParsedTransaction]:
    """Build a ParsedTransaction from banking line parts."""
    trans_date_str, posted_date_str, amount_str = parts
    try:
        cash_date = date.fromisoformat(trans_date_str)
        amount = _parse_amount(amount_str)
    except (ValueError, InvalidOperation) as e:
        logger.warning("Failed to parse banking txn: %s (%s)", parts, e)
        return None

    return ParsedTransaction(
        issuer="WEALTHSIMPLE",
        cash_date=cash_date,
        description=description.strip(),
        original_amount=amount,
        original_currency="CAD",
        fx_rate=None,
        fx_rate_source=None,
        cad_amount=amount,
        statement_page=page_num,
    )


def _is_banking_skip_line(line: str) -> bool:
    """Check if a banking line should be skipped."""
    skip_starts = [
        "Chequing monthly", "Joint chequing", "Wealthsimple",
        "Account number:", "Your ", "Activity",
        "DATE POSTED", "Date is when", "Posted date is",
        "Page ", "$",
    ]
    for prefix in skip_starts:
        if line.startswith(prefix):
            return True
    # Name/address lines
    if re.match(r"^[A-Z][a-z]+ [A-Z][a-z]+ [A-Z]+", line):  # "Ho Chi YU"
        return True
    if re.match(r"^\d{4}\s*-\s*\d{4}", line) and "Hazelbridge" in line:
        return True
    if line.startswith("Richmond") or line.startswith("Canada"):
        return True
    return False


# ---------------------------------------------------------------------------
# Credit card parser
# ---------------------------------------------------------------------------

# Credit card transaction line: Mon DD Mon DD Type Description Amount
# Example: Jan 21 Jan 22 Purchase SHOPPERS DRUG MART #22 $5.24
# Example: Jan 30 Jan 30 Payment From chequing account –$672.25
_CC_TXN_RE = re.compile(
    r"^([A-Z][a-z]{2}\s+\d{1,2})\s+"     # Trans date (Mon DD)
    r"([A-Z][a-z]{2}\s+\d{1,2})\s+"       # Posted date (Mon DD)
    r"(Purchase|Payment|Cash withdrawal|Interest charge)\s+"  # Type
    r"(.+?)\s+"                            # Details/description
    r"([–\-]?\$[\d,]+\.\d{2})\s*$"        # Amount
)

# Simpler fallback: some lines might have the amount directly after description
# without clear Type (interest charge lines)
_CC_INTEREST_RE = re.compile(
    r"^([A-Z][a-z]{2}\s+\d{1,2})\s+"
    r"([A-Z][a-z]{2}\s+\d{1,2})\s+"
    r"(Interest charge)\s+"
    r"(.+?)\s+"
    r"([–\-]?\$[\d,]+\.\d{2})\s*$"
)

# FX conversion line: "3,755.55 HKD • 0.178637 exchange rate"
_FX_LINE_RE = re.compile(
    r"^([\d,]+\.?\d*)\s+"                  # Foreign amount
    r"([A-Z]{3})\s+"                       # Currency code
    r"[•·]\s+"                             # Bullet separator
    r"([\d.]+)\s+"                         # Exchange rate
    r"exchange rate\s*$"
)


def _parse_cc_short_date(date_str: str, statement_year: int, statement_end_month: int) -> date:
    """Parse 'Jan 21' to a full date using statement context for year.

    The statement spans two months (e.g., Dec 15 - Jan 14, 2026). Dates
    in the earlier month belong to the prior year if the statement
    crosses a year boundary.
    """
    parts = date_str.strip().split()
    month_name = parts[0].lower()
    day = int(parts[1])
    month = _MONTH_MAP[month_name]

    # If the month is > statement_end_month, it's likely the previous year
    # (e.g., Dec dates in a Dec-Jan statement with year=2026)
    if month > statement_end_month:
        return date(statement_year - 1, month, day)
    return date(statement_year, month, day)


def _extract_cc_year_and_end_month(first_page_text: str) -> tuple[int, int]:
    """Extract statement year and end month from the first page.

    Looks for patterns like "Dec 15 — Jan 14, 2026" or "Jan 15 — Feb 14, 2026".
    """
    # Try "Mon DD — Mon DD, YYYY" pattern
    m = re.search(
        r"([A-Z][a-z]{2})\s+\d{1,2}\s*[—–-]\s*([A-Z][a-z]{2})\s+(\d{1,2}),?\s+(\d{4})",
        first_page_text,
    )
    if m:
        end_month_name = m.group(2).lower()
        year = int(m.group(4))
        end_month = _MONTH_MAP.get(end_month_name, 1)
        return year, end_month

    # Fallback: look for a 4-digit year
    year_match = re.search(r"\b(20\d{2})\b", first_page_text)
    year = int(year_match.group(1)) if year_match else 2026
    return year, 12


def _parse_cc_pages(
    all_text_pages: list[str],
    statement_year: int,
    statement_end_month: int,
) -> list[ParsedTransaction]:
    """Parse credit card statement pages."""
    transactions: list[ParsedTransaction] = []

    for page_idx, text in enumerate(all_text_pages):
        page_num = page_idx + 1
        lines = text.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i].strip()
            i += 1

            if not line:
                continue

            # Skip headers, footers, legal text
            if _is_cc_skip_line(line):
                continue

            # Try to match a transaction line
            match = _CC_TXN_RE.match(line)
            if not match:
                continue

            trans_date_str = match.group(1)
            _posted_date_str = match.group(2)
            txn_type = match.group(3)
            description = match.group(4).strip()
            amount_str = match.group(5)

            try:
                cash_date = _parse_cc_short_date(
                    trans_date_str, statement_year, statement_end_month
                )
                amount = _parse_amount(amount_str)
            except (ValueError, InvalidOperation, KeyError) as e:
                logger.warning("Page %d: failed to parse CC txn: %s (%s)", page_num, line, e)
                continue

            # Check next line for FX info
            fx_amount_foreign: Optional[Decimal] = None
            fx_currency: Optional[str] = None
            fx_rate: Optional[Decimal] = None

            if i < len(lines):
                fx_match = _FX_LINE_RE.match(lines[i].strip())
                if fx_match:
                    try:
                        fx_amount_foreign = Decimal(fx_match.group(1).replace(",", ""))
                        fx_currency = fx_match.group(2)
                        fx_rate = Decimal(fx_match.group(3))
                    except (InvalidOperation, ValueError):
                        pass
                    i += 1  # Skip the FX line

            # Skip payments and interest charges — those aren't purchases
            if txn_type in ("Payment", "Interest charge"):
                continue

            original_currency = fx_currency or "CAD"
            original_amount = fx_amount_foreign if fx_amount_foreign is not None else amount

            txn = ParsedTransaction(
                issuer="WEALTHSIMPLE",
                cash_date=cash_date,
                description=description,
                original_amount=original_amount,
                original_currency=original_currency,
                fx_rate=fx_rate,
                fx_rate_source="statement" if fx_rate is not None else None,
                cad_amount=amount,
                statement_page=page_num,
            )
            transactions.append(txn)

    return transactions


def _is_cc_skip_line(line: str) -> bool:
    """Check if a credit card line should be skipped."""
    skip_starts = [
        "Credit card statement", "Wealthsimple", "4126", "HO CHI",
        "1087", "RICHMOND", "STATEMENT BALANCE", "Statement date",
        "Credit limit", "If you only", "Account summary",
        "Previous balance", "- Payments", "- Other credits",
        "+ Purchases", "+ Fees", "+ Interest", "+ Cash advances",
        "Total ", "Annual interest", "Cash advance interest",
        "New balance", "Page ", "Activity",
        "TRANS. DATE", "Information about",
        "Minimum payment", "How we charge", "Missed payments",
        "Foreign transaction", "Your obligations",
    ]
    for prefix in skip_starts:
        if line.startswith(prefix):
            return True
    # Numbered legal notes
    if re.match(r"^\d\s+", line):
        return True
    return False


# ---------------------------------------------------------------------------
# Main parser class
# ---------------------------------------------------------------------------

class WealthsimpleParser(IssuerParser):
    """Parser for Wealthsimple banking and credit card PDF statements."""

    issuer_name: str = "WEALTHSIMPLE"

    def detect(self, filename: str, first_page_text: str) -> bool:
        """Detect Wealthsimple statements by filename or first-page text."""
        filename_lower = filename.lower()
        if "wealthsimple" in filename_lower or "ws_" in filename_lower:
            return True
        if "wealthsimple" in first_page_text.lower():
            return True
        return False

    def parse(self, pdf_bytes: bytes) -> list[ParsedTransaction]:
        """Extract transactions, branching on banking vs credit card."""
        try:
            pdf = pdfplumber.open(BytesIO(pdf_bytes))
        except Exception:
            logger.exception("Failed to open Wealthsimple PDF")
            return []

        try:
            if not pdf.pages:
                return []

            first_page_text = pdf.pages[0].extract_text() or ""
            all_text_pages = [p.extract_text() or "" for p in pdf.pages]

            is_credit_card = "credit card statement" in first_page_text.lower()
            is_banking = "chequing" in first_page_text.lower()

            if is_credit_card:
                logger.info("Detected Wealthsimple credit card statement")
                year, end_month = _extract_cc_year_and_end_month(first_page_text)
                txns = _parse_cc_pages(all_text_pages, year, end_month)
            elif is_banking:
                logger.info("Detected Wealthsimple banking (chequing) statement")
                txns = _parse_banking_pages(pdf.pages, all_text_pages)
            else:
                logger.warning("Unknown Wealthsimple statement type")
                txns = []

            logger.info(
                "Wealthsimple parser: extracted %d transactions from %d pages",
                len(txns), len(pdf.pages),
            )
            return txns

        except Exception:
            logger.exception("Wealthsimple parser error")
            return []
        finally:
            pdf.close()
