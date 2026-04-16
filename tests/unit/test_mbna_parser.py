"""Tests for parsers/mbna.py — MBNA Canada statement parser.

Unit tests use synthetic text; integration tests use real statement fixtures.
"""

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from parsers.mbna import MBNAParser, _parse_amount, _parse_date


# ---------------------------------------------------------------------------
# Helper parsing functions
# ---------------------------------------------------------------------------

class TestParseDate:
    def test_standard_date(self) -> None:
        assert _parse_date("02/21/26") == date(2026, 2, 21)

    def test_december_2025(self) -> None:
        assert _parse_date("12/24/25") == date(2025, 12, 24)

    def test_january_first(self) -> None:
        assert _parse_date("01/01/26") == date(2026, 1, 1)


class TestParseAmount:
    def test_positive_amount(self) -> None:
        assert _parse_amount("$27.46") == Decimal("27.46")

    def test_negative_amount(self) -> None:
        assert _parse_amount("-$4.31") == Decimal("-4.31")

    def test_comma_separated(self) -> None:
        assert _parse_amount("$1,127.95") == Decimal("1127.95")

    def test_large_amount(self) -> None:
        assert _parse_amount("$10,234.56") == Decimal("10234.56")


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class TestMBNADetection:
    @pytest.fixture
    def parser(self) -> MBNAParser:
        return MBNAParser()

    def test_detect_by_filename(self, parser: MBNAParser) -> None:
        assert parser.detect("MBNA_2026-03.pdf", "") is True

    def test_detect_by_filename_case_insensitive(self, parser: MBNAParser) -> None:
        assert parser.detect("mbna canada.pdf", "") is True

    def test_detect_by_first_page_text(self, parser: MBNAParser) -> None:
        text = "Your Credit Card Account Statement\nMBNA\nSome other text"
        assert parser.detect("statement.pdf", text) is True

    def test_no_detect_unrelated(self, parser: MBNAParser) -> None:
        assert parser.detect("rogers_statement.pdf", "Rogers Bank") is False

    def test_issuer_name(self, parser: MBNAParser) -> None:
        assert parser.issuer_name == "MBNA"


# ---------------------------------------------------------------------------
# Integration tests with real statements
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _skip_if_no_fixtures():
    """Skip integration tests if real statement fixtures are not available."""
    if not FIXTURES_DIR.exists():
        pytest.skip("No fixtures directory")
    pdfs = list(FIXTURES_DIR.glob("MBNA*.pdf"))
    if not pdfs:
        pytest.skip("No MBNA fixtures available")


class TestMBNAIntegration:
    """Integration tests using real MBNA statement PDFs."""

    @pytest.fixture(autouse=True)
    def _check_fixtures(self) -> None:
        _skip_if_no_fixtures()

    @pytest.fixture
    def parser(self) -> MBNAParser:
        return MBNAParser()

    def test_parse_statement_feb_mar(self, parser: MBNAParser) -> None:
        """MBNA Canada.pdf: Feb 21 - Mar 20, 2026."""
        pdf_path = FIXTURES_DIR / "MBNA Canada.pdf"
        if not pdf_path.exists():
            pytest.skip("MBNA Canada.pdf not found")

        txns = parser.parse(pdf_path.read_bytes())

        assert len(txns) >= 1
        total = sum(t.original_amount for t in txns)
        assert total == Decimal("1127.95")

        # All transactions are MBNA issuer
        assert all(t.issuer == "MBNA" for t in txns)
        # All are CAD
        assert all(t.original_currency == "CAD" for t in txns)
        # Dates are within statement period
        assert all(date(2025, 12, 1) <= t.cash_date <= date(2026, 4, 1) for t in txns)

    def test_parse_statement_jan_feb(self, parser: MBNAParser) -> None:
        """MBNA Canada (1).pdf: Jan 21 - Feb 20, 2026."""
        pdf_path = FIXTURES_DIR / "MBNA Canada (1).pdf"
        if not pdf_path.exists():
            pytest.skip("MBNA Canada (1).pdf not found")

        txns = parser.parse(pdf_path.read_bytes())

        assert len(txns) >= 1
        total = sum(t.original_amount for t in txns)
        assert total == Decimal("758.02")

        # Check refund is preserved as negative
        credits = [t for t in txns if t.original_amount < 0]
        assert len(credits) >= 1
        assert any(t.original_amount == Decimal("-4.31") for t in credits)

    def test_parse_statement_dec_jan(self, parser: MBNAParser) -> None:
        """MBNA Canada (2).pdf: Dec 23, 2025 - Jan 20, 2026."""
        pdf_path = FIXTURES_DIR / "MBNA Canada (2).pdf"
        if not pdf_path.exists():
            pytest.skip("MBNA Canada (2).pdf not found")

        txns = parser.parse(pdf_path.read_bytes())

        assert len(txns) >= 1
        total = sum(t.original_amount for t in txns)
        assert total == Decimal("1601.20")

    def test_no_payment_lines_included(self, parser: MBNAParser) -> None:
        """Payment lines should be excluded from parsed transactions."""
        pdf_path = FIXTURES_DIR / "MBNA Canada.pdf"
        if not pdf_path.exists():
            pytest.skip("MBNA Canada.pdf not found")

        txns = parser.parse(pdf_path.read_bytes())

        # No transaction should have "PAYMENT" as description
        assert not any("PAYMENT" == t.description for t in txns)

    def test_page_numbers_populated(self, parser: MBNAParser) -> None:
        """Each transaction should have a valid page number."""
        pdf_path = FIXTURES_DIR / "MBNA Canada (2).pdf"
        if not pdf_path.exists():
            pytest.skip("MBNA Canada (2).pdf not found")

        txns = parser.parse(pdf_path.read_bytes())

        assert all(t.statement_page >= 3 for t in txns)

    def test_all_three_statements_combined(self, parser: MBNAParser) -> None:
        """Verify total count across all three statements."""
        total_txns = 0
        for pdf_path in sorted(FIXTURES_DIR.glob("MBNA*.pdf")):
            txns = parser.parse(pdf_path.read_bytes())
            total_txns += len(txns)

        # Should have a substantial number of transactions
        assert total_txns >= 100
