"""Tests for parsers/rogers.py — Rogers Bank Mastercard parser."""

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from parsers.rogers import RogersParser, _parse_rogers_date, _parse_amount, _extract_period


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestParseRogersDate:
    def test_no_space(self) -> None:
        assert _parse_rogers_date("Feb18", 2026, 3) == date(2026, 2, 18)

    def test_with_space(self) -> None:
        assert _parse_rogers_date("Feb 18", 2026, 3) == date(2026, 2, 18)

    def test_year_boundary(self) -> None:
        """Dec dates in a Dec-Jan statement should be previous year."""
        assert _parse_rogers_date("Dec17", 2026, 1) == date(2025, 12, 17)

    def test_single_digit_day(self) -> None:
        assert _parse_rogers_date("Mar1", 2026, 3) == date(2026, 3, 1)


class TestParseAmount:
    def test_positive(self) -> None:
        assert _parse_amount("20.00") == Decimal("20.00")

    def test_negative(self) -> None:
        assert _parse_amount("-400.00") == Decimal("-400.00")

    def test_comma(self) -> None:
        assert _parse_amount("1,127.95") == Decimal("1127.95")


class TestExtractPeriod:
    def test_feb_mar(self) -> None:
        text = "Statement Period Feb 18,2026-Mar17,2026"
        year, end_month = _extract_period(text)
        assert year == 2026
        assert end_month == 3

    def test_dec_jan(self) -> None:
        text = "Statement Period Dec 18,2025-Jan 17,2026"
        year, end_month = _extract_period(text)
        assert year == 2026
        assert end_month == 1


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class TestRogersDetection:
    @pytest.fixture
    def parser(self) -> RogersParser:
        return RogersParser()

    def test_detect_by_filename(self, parser: RogersParser) -> None:
        assert parser.detect("rogers_statement.pdf", "") is True

    def test_detect_by_first_page_text(self, parser: RogersParser) -> None:
        assert parser.detect("statement.pdf", "www.rogersbank.com\nAccount") is True

    def test_no_detect_unrelated(self, parser: RogersParser) -> None:
        assert parser.detect("mbna.pdf", "MBNA Credit Card") is False

    def test_issuer_name(self, parser: RogersParser) -> None:
        assert parser.issuer_name == "ROGERS"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "Rogers-bank"


def _skip_if_no_fixtures():
    if not FIXTURES_DIR.exists() or not list(FIXTURES_DIR.glob("*.pdf")):
        pytest.skip("No Rogers fixtures")


class TestRogersIntegration:
    @pytest.fixture(autouse=True)
    def _check(self) -> None:
        _skip_if_no_fixtures()

    @pytest.fixture
    def parser(self) -> RogersParser:
        return RogersParser()

    def test_feb_mar_statement(self, parser: RogersParser) -> None:
        """1776313561958.pdf: Feb 18 - Mar 17, 2026."""
        pdf_path = FIXTURES_DIR / "1776313561958.pdf"
        if not pdf_path.exists():
            pytest.skip()
        txns = parser.parse(pdf_path.read_bytes())
        assert len(txns) >= 10
        total = sum(t.cad_amount for t in txns)
        assert total == Decimal("473.31")

    def test_jan_feb_statement(self, parser: RogersParser) -> None:
        """1776313579061.pdf: Jan 18 - Feb 17, 2026."""
        pdf_path = FIXTURES_DIR / "1776313579061.pdf"
        if not pdf_path.exists():
            pytest.skip()
        txns = parser.parse(pdf_path.read_bytes())
        assert len(txns) >= 15
        total = sum(t.cad_amount for t in txns)
        assert total == Decimal("445.29")

    def test_dec_jan_statement(self, parser: RogersParser) -> None:
        """1776313592798.pdf: Dec 18, 2025 - Jan 17, 2026."""
        pdf_path = FIXTURES_DIR / "1776313592798.pdf"
        if not pdf_path.exists():
            pytest.skip()
        txns = parser.parse(pdf_path.read_bytes())
        assert len(txns) >= 15
        total = sum(t.cad_amount for t in txns)
        assert total == Decimal("612.85")

    def test_payments_excluded(self, parser: RogersParser) -> None:
        """'PAYMENT,THANKYOU' lines should not appear as transactions."""
        for pdf_path in FIXTURES_DIR.glob("*.pdf"):
            txns = parser.parse(pdf_path.read_bytes())
            assert not any("THANKYOU" in t.description.upper().replace(" ", "") for t in txns), pdf_path.name

    def test_fx_transactions_have_rate(self, parser: RogersParser) -> None:
        """Foreign currency transactions should have fx_rate populated."""
        pdf_path = FIXTURES_DIR / "1776313561958.pdf"
        if not pdf_path.exists():
            pytest.skip()
        txns = parser.parse(pdf_path.read_bytes())
        fx_txns = [t for t in txns if t.original_currency != "CAD"]
        assert len(fx_txns) >= 2
        assert all(t.fx_rate is not None for t in fx_txns)
        assert all(t.fx_rate_source == "statement" for t in fx_txns)

    def test_dec_dates_correct_year(self, parser: RogersParser) -> None:
        """Dec dates should be 2025 in the Dec-Jan statement."""
        pdf_path = FIXTURES_DIR / "1776313592798.pdf"
        if not pdf_path.exists():
            pytest.skip()
        txns = parser.parse(pdf_path.read_bytes())
        dec_txns = [t for t in txns if t.cash_date.month == 12]
        assert len(dec_txns) > 0
        assert all(t.cash_date.year == 2025 for t in dec_txns)

    def test_all_issuer_rogers(self, parser: RogersParser) -> None:
        for pdf_path in FIXTURES_DIR.glob("*.pdf"):
            txns = parser.parse(pdf_path.read_bytes())
            assert all(t.issuer == "ROGERS" for t in txns), pdf_path.name

    def test_combined_total(self, parser: RogersParser) -> None:
        """All 3 statements combined should have a substantial count."""
        total_txns = 0
        for pdf_path in FIXTURES_DIR.glob("*.pdf"):
            txns = parser.parse(pdf_path.read_bytes())
            total_txns += len(txns)
        assert total_txns >= 50
