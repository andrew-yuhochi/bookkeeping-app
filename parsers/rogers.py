"""Rogers Bank Mastercard statement parser.

Extracts transactions from Rogers Bank PDF statements using pdfplumber
text extraction and line-based regex parsing.

Statement layout (observed from real samples):
  - Page 1: Summary + first transactions (mixed)
  - Page 2: Legal boilerplate
  - Page 3+: Continued transactions + interest rate chart
  - Remaining pages: blank (headers only)

Transaction line format (no $ sign on amounts):
  MonDD MonDD DESCRIPTION CITY PROV AMOUNT
  e.g.: Feb18 Feb19 COMPASSVENDINGBURNAB BURNABY BC 20.00

Foreign currency format (next line):
  FOREIGNCURRENCY JPY 1,000@0.009160000

Multiple cards may appear separated by "CardNumber XXXXXXXXXXXX####".
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

# Month name → number
_MONTH_MAP: dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Transaction line regex:
#   MonDD MonDD Description Amount
# Dates may or may not have space between month and day.
# Amount has no $ sign, optional minus, optional commas.
_TXN_LINE_RE = re.compile(
    r"^([A-Z][a-z]{2}\s?\d{1,2})\s+"     # Trans date (MonDD or Mon DD)
    r"([A-Z][a-z]{2}\s?\d{1,2})\s+"       # Post date
    r"(.+?)\s+"                            # Description
    r"(-?[\d,]+\.\d{2})\s*$"              # Amount (no $ sign)
)

# Foreign currency line:
#   FOREIGNCURRENCY JPY 1,000@0.009160000
_FX_LINE_RE = re.compile(
    r"^FOREIGNCURRENCY\s+"
    r"([A-Z]{3})\s+"                       # Currency code
    r"([\d,]+\.?\d*)"                      # Foreign amount
    r"@"
    r"([\d.]+)"                            # Exchange rate
)

# Statement period: "Statement Period Feb 18,2026-Mar17,2026"
_PERIOD_RE = re.compile(
    r"Statement\s*Period\s*"
    r"([A-Z][a-z]{2})\s*(\d{1,2})\s*,?\s*(\d{4})\s*-\s*"
    r"([A-Z][a-z]{2})\s*(\d{1,2})\s*,?\s*(\d{4})",
    re.IGNORECASE,
)


def _parse_rogers_date(date_str: str, statement_year: int, statement_end_month: int) -> date:
    """Parse 'Feb18' or 'Feb 18' to a full date using statement context for year."""
    # Normalize: ensure space between month and day
    cleaned = date_str.strip()
    month_name = cleaned[:3].lower()
    day_str = cleaned[3:].strip()
    month = _MONTH_MAP[month_name]
    day = int(day_str)

    # Year boundary: if month > end_month, it's the previous year
    if month > statement_end_month:
        return date(statement_year - 1, month, day)
    return date(statement_year, month, day)


def _parse_amount(s: str) -> Decimal:
    """Parse an amount like '20.00', '-400.00', '1,127.95'."""
    return Decimal(s.replace(",", ""))


def _extract_period(text: str) -> tuple[int, int]:
    """Extract statement year and end month from page text.

    Returns (year_of_end_date, end_month).
    """
    m = _PERIOD_RE.search(text)
    if m:
        end_year = int(m.group(6))
        end_month = _MONTH_MAP.get(m.group(4).lower(), 12)
        return end_year, end_month
    # Fallback
    year_match = re.search(r"\b(20\d{2})\b", text)
    return int(year_match.group(1)) if year_match else 2026, 12


class RogersParser(IssuerParser):
    """Parser for Rogers Bank Mastercard PDF statements."""

    issuer_name: str = "ROGERS"

    def detect(self, filename: str, first_page_text: str) -> bool:
        """Detect Rogers Bank statements by filename or first-page text."""
        if "rogers" in filename.lower():
            return True
        text_lower = first_page_text.lower()
        if "rogers bank" in text_lower or "rogersbank.com" in text_lower:
            return True
        return False

    def parse(self, pdf_bytes: bytes) -> list[ParsedTransaction]:
        """Extract transactions from a Rogers Bank PDF statement."""
        try:
            pdf = pdfplumber.open(BytesIO(pdf_bytes))
        except Exception:
            logger.exception("Failed to open Rogers PDF")
            return []

        transactions: list[ParsedTransaction] = []

        try:
            if not pdf.pages:
                return []

            first_page_text = pdf.pages[0].extract_text() or ""
            statement_year, statement_end_month = _extract_period(first_page_text)

            for page_idx, page in enumerate(pdf.pages):
                page_num = page_idx + 1

                # Page 2 is always legal boilerplate
                if page_num == 2:
                    continue

                try:
                    text = page.extract_text()
                    if not text:
                        continue

                    page_txns = self._parse_page(
                        text, page_num, statement_year, statement_end_month,
                    )
                    transactions.extend(page_txns)

                except Exception:
                    logger.exception(
                        "Page %d: parse error — continuing", page_num,
                    )

        finally:
            pdf.close()

        logger.info(
            "Rogers parser: extracted %d transactions from %d pages",
            len(transactions), len(pdf.pages),
        )
        return transactions

    def _parse_page(
        self,
        text: str,
        page_num: int,
        statement_year: int,
        statement_end_month: int,
    ) -> list[ParsedTransaction]:
        """Parse transaction lines from a single page."""
        results: list[ParsedTransaction] = []
        lines = text.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i].strip()
            i += 1

            if not line:
                continue

            # Stop parsing if we hit non-transaction sections
            if self._is_stop_section(line):
                break

            # Skip known non-transaction lines
            if self._should_skip(line):
                continue

            # Try to match a transaction line
            match = _TXN_LINE_RE.match(line)
            if not match:
                continue

            trans_date_str = match.group(1)
            _post_date_str = match.group(2)
            description = match.group(3).strip()
            amount_str = match.group(4)

            # Skip payment lines
            if "PAYMENT" in description.upper() and "THANKYOU" in description.upper().replace(" ", ""):
                continue

            try:
                cash_date = _parse_rogers_date(
                    trans_date_str, statement_year, statement_end_month,
                )
                cad_amount = _parse_amount(amount_str)
            except (ValueError, InvalidOperation, KeyError) as e:
                logger.warning("Page %d: failed to parse: %s (%s)", page_num, line, e)
                continue

            # Check next line for foreign currency info
            fx_currency: Optional[str] = None
            fx_amount_foreign: Optional[Decimal] = None
            fx_rate: Optional[Decimal] = None

            if i < len(lines):
                fx_match = _FX_LINE_RE.match(lines[i].strip())
                if fx_match:
                    try:
                        fx_currency = fx_match.group(1)
                        fx_amount_foreign = Decimal(fx_match.group(2).replace(",", ""))
                        fx_rate = Decimal(fx_match.group(3))
                    except (InvalidOperation, ValueError):
                        pass
                    i += 1

            original_currency = fx_currency or "CAD"
            original_amount = fx_amount_foreign if fx_amount_foreign is not None else cad_amount

            txn = ParsedTransaction(
                issuer="ROGERS",
                cash_date=cash_date,
                description=description,
                original_amount=original_amount,
                original_currency=original_currency,
                fx_rate=fx_rate,
                fx_rate_source="statement" if fx_rate is not None else None,
                cad_amount=cad_amount,
                statement_page=page_num,
            )
            results.append(txn)

        return results

    def _is_stop_section(self, line: str) -> bool:
        """Detect when we've left the transaction area."""
        stop_markers = [
            "InterestRateChart",
            "Interest Rate Chart",
            "AMOUNT NEW MINIMUM",
            "AMOUNT DUE",
            "POBOX",
            "PO BOX",
        ]
        for marker in stop_markers:
            if marker in line.replace(" ", ""):
                return True
        return False

    def _should_skip(self, line: str) -> bool:
        """Skip non-transaction lines."""
        skip_patterns = [
            "Website", "AccountNumber", "Statement Period",
            "AccountDetails", "Minimum payment", "Paymentduedate",
            "Credit limit", "Available credit", "Amount Due",
            "MinimumPaymentNotice", "If youmakeonlythe",
            "basedontheNew", "Messages", "Pleaseallow",
            "TransactionDetails", "Trans Post", "Date Date Description",
            "CardNumber", "Account Number", "Page",
            "Your rewards", "Inthenew", "redeem cashback",
            "your Rogers", "WiththeRogersBankapp",
            "youraccount", "yourcurrentbalance",
            "cashbackrewards", "app.",
            "Onceyou", "towardsany",
            "andwherever", "Toredeem",
            "GooglePlay", "ForcompletedetailsontheRogers",
            "MastercardRewards", "Conditions",
            "Previous balance", "Payments &credits",
            "Newpurchases", "Cash advances",
            "Promotional balances", "Fees", "Interest",
            "New Balance", "year(s)and",
            "cardaccountterms",
        ]
        for pat in skip_patterns:
            if line.startswith(pat):
                return True
        # Numeric-only lines (card numbers, barcodes)
        if re.match(r"^[\d\s]+$", line):
            return True
        # Address lines
        if line.startswith("ANDREWYU") or line.startswith("ANDREW YU"):
            return True
        if "HAZELBRIDGEWAY" in line or "HAZELBRIDGE" in line:
            return True
        if line.startswith("RICHMOND") or line.startswith("CANADA"):
            return True
        if line.startswith("UNIT"):
            return True
        # Encoded strings
        if re.match(r"^[a-z0-9]{10,}", line):
            return True
        return False
