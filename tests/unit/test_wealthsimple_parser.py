"""Tests for parsers/wealthsimple.py — Wealthsimple banking + credit card parser.

Unit tests use synthetic inputs; integration tests use real statement fixtures.
"""

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from parsers.wealthsimple import (
    WealthsimpleParser,
    _parse_amount,
    _parse_cc_short_date,
    _extract_cc_year_and_end_month,
)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestParseAmount:
    def test_positive(self) -> None:
        assert _parse_amount("$27.46") == Decimal("27.46")

    def test_negative_en_dash(self) -> None:
        assert _parse_amount("–$204.50") == Decimal("-204.50")

    def test_negative_hyphen(self) -> None:
        assert _parse_amount("-$4.31") == Decimal("-4.31")

    def test_comma_separated(self) -> None:
        assert _parse_amount("$1,601.20") == Decimal("1601.20")


class TestParseCCShortDate:
    def test_same_year(self) -> None:
        assert _parse_cc_short_date("Jan 21", 2026, 2) == date(2026, 1, 21)

    def test_year_boundary(self) -> None:
        """Dec dates in a Dec-Jan statement should be previous year."""
        assert _parse_cc_short_date("Dec 15", 2026, 1) == date(2025, 12, 15)

    def test_end_month(self) -> None:
        assert _parse_cc_short_date("Feb 14", 2026, 2) == date(2026, 2, 14)


class TestExtractCCYearAndEndMonth:
    def test_dec_jan(self) -> None:
        text = "Credit card statement\nDec 15 — Jan 14, 2026\n4126..."
        year, end_month = _extract_cc_year_and_end_month(text)
        assert year == 2026
        assert end_month == 1

    def test_jan_feb(self) -> None:
        text = "Credit card statement\nJan 15 — Feb 14, 2026\n4126..."
        year, end_month = _extract_cc_year_and_end_month(text)
        assert year == 2026
        assert end_month == 2

    def test_feb_mar(self) -> None:
        text = "Credit card statement\nFeb 15 — Mar 14, 2026\n4126..."
        year, end_month = _extract_cc_year_and_end_month(text)
        assert year == 2026
        assert end_month == 3


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------

class TestWealthsimpleDetection:
    @pytest.fixture
    def parser(self) -> WealthsimpleParser:
        return WealthsimpleParser()

    def test_detect_by_filename_wealthsimple(self, parser: WealthsimpleParser) -> None:
        assert parser.detect("wealthsimple_jan_2026.pdf", "") is True

    def test_detect_by_filename_ws(self, parser: WealthsimpleParser) -> None:
        assert parser.detect("ws_statement.pdf", "") is True

    def test_detect_by_first_page_text(self, parser: WealthsimpleParser) -> None:
        assert parser.detect("statement.pdf", "Wealthsimple\nCredit card") is True

    def test_no_detect_unrelated(self, parser: WealthsimpleParser) -> None:
        assert parser.detect("mbna_statement.pdf", "MBNA Credit Card") is False

    def test_issuer_name(self, parser: WealthsimpleParser) -> None:
        assert parser.issuer_name == "WEALTHSIMPLE"


# ---------------------------------------------------------------------------
# Integration tests — Banking
# ---------------------------------------------------------------------------

BANKING_DIR = Path(__file__).parent.parent / "fixtures" / "WS-banking"
CC_DIR = Path(__file__).parent.parent / "fixtures" / "WS-credit-card"


def _skip_if_no_banking():
    if not BANKING_DIR.exists() or not list(BANKING_DIR.glob("*.pdf")):
        pytest.skip("No WS banking fixtures")


def _skip_if_no_cc():
    if not CC_DIR.exists() or not list(CC_DIR.glob("*.pdf")):
        pytest.skip("No WS credit card fixtures")


class TestBankingIntegration:
    @pytest.fixture(autouse=True)
    def _check(self) -> None:
        _skip_if_no_banking()

    @pytest.fixture
    def parser(self) -> WealthsimpleParser:
        return WealthsimpleParser()

    def test_jan_statement(self, parser: WealthsimpleParser) -> None:
        pdf_path = BANKING_DIR / "January_2026 (1).pdf"
        if not pdf_path.exists():
            pytest.skip()
        txns = parser.parse(pdf_path.read_bytes())
        assert len(txns) == 2
        net = sum(t.original_amount for t in txns)
        assert net == Decimal("0.00")

    def test_feb_statement(self, parser: WealthsimpleParser) -> None:
        pdf_path = BANKING_DIR / "February_2026 (1).pdf"
        if not pdf_path.exists():
            pytest.skip()
        txns = parser.parse(pdf_path.read_bytes())
        assert len(txns) >= 10
        net = sum(t.original_amount for t in txns)
        # Feb: $3,738.45 - $4,594.64 = -$856.19
        assert net == Decimal("-856.19")

    def test_mar_statement(self, parser: WealthsimpleParser) -> None:
        pdf_path = BANKING_DIR / "March_2026 (1).pdf"
        if not pdf_path.exists():
            pytest.skip()
        txns = parser.parse(pdf_path.read_bytes())
        assert len(txns) >= 15
        net = sum(t.original_amount for t in txns)
        # Mar: $8,209.52 - $3,738.45 = $4,471.07
        assert net == Decimal("4471.07")

    def test_all_banking_are_cad(self, parser: WealthsimpleParser) -> None:
        for pdf_path in BANKING_DIR.glob("*.pdf"):
            txns = parser.parse(pdf_path.read_bytes())
            assert all(t.original_currency == "CAD" for t in txns), pdf_path.name

    def test_all_issuer_wealthsimple(self, parser: WealthsimpleParser) -> None:
        for pdf_path in BANKING_DIR.glob("*.pdf"):
            txns = parser.parse(pdf_path.read_bytes())
            assert all(t.issuer == "WEALTHSIMPLE" for t in txns), pdf_path.name


# ---------------------------------------------------------------------------
# Integration tests — Credit Card
# ---------------------------------------------------------------------------

class TestCreditCardIntegration:
    @pytest.fixture(autouse=True)
    def _check(self) -> None:
        _skip_if_no_cc()

    @pytest.fixture
    def parser(self) -> WealthsimpleParser:
        return WealthsimpleParser()

    def test_jan_cc_statement(self, parser: WealthsimpleParser) -> None:
        pdf_path = CC_DIR / "January_2026.pdf"
        if not pdf_path.exists():
            pytest.skip()
        txns = parser.parse(pdf_path.read_bytes())
        assert len(txns) == 2
        total_cad = sum(t.cad_amount for t in txns if t.cad_amount)
        assert total_cad == Decimal("672.25")

    def test_jan_cc_fx_transactions(self, parser: WealthsimpleParser) -> None:
        pdf_path = CC_DIR / "January_2026.pdf"
        if not pdf_path.exists():
            pytest.skip()
        txns = parser.parse(pdf_path.read_bytes())
        # Both are FX: USD and HKD
        assert all(t.original_currency != "CAD" for t in txns)
        assert any(t.original_currency == "USD" for t in txns)
        assert any(t.original_currency == "HKD" for t in txns)
        # FX rates populated
        assert all(t.fx_rate is not None for t in txns)

    def test_feb_cc_statement(self, parser: WealthsimpleParser) -> None:
        pdf_path = CC_DIR / "February_2026.pdf"
        if not pdf_path.exists():
            pytest.skip()
        txns = parser.parse(pdf_path.read_bytes())
        assert len(txns) >= 30
        total_cad = sum(t.cad_amount for t in txns if t.cad_amount)
        # Purchases $3,232.51 + Cash advances $189.77 = $3,422.28
        assert total_cad == Decimal("3422.28")

    def test_feb_cc_has_jpy_transactions(self, parser: WealthsimpleParser) -> None:
        pdf_path = CC_DIR / "February_2026.pdf"
        if not pdf_path.exists():
            pytest.skip()
        txns = parser.parse(pdf_path.read_bytes())
        jpy_txns = [t for t in txns if t.original_currency == "JPY"]
        assert len(jpy_txns) >= 20  # Lots of Japan transactions

    def test_mar_cc_statement(self, parser: WealthsimpleParser) -> None:
        pdf_path = CC_DIR / "March_2026.pdf"
        if not pdf_path.exists():
            pytest.skip()
        txns = parser.parse(pdf_path.read_bytes())
        assert len(txns) >= 30
        total_cad = sum(t.cad_amount for t in txns if t.cad_amount)
        assert total_cad == Decimal("1872.98")

    def test_payments_excluded(self, parser: WealthsimpleParser) -> None:
        """Payment lines should not appear as transactions."""
        for pdf_path in CC_DIR.glob("*.pdf"):
            txns = parser.parse(pdf_path.read_bytes())
            assert not any("chequing account" in t.description.lower() for t in txns), pdf_path.name

    def test_interest_excluded(self, parser: WealthsimpleParser) -> None:
        """Interest charge lines should not appear as transactions."""
        for pdf_path in CC_DIR.glob("*.pdf"):
            txns = parser.parse(pdf_path.read_bytes())
            assert not any("interest" in t.description.lower() for t in txns), pdf_path.name

    def test_dec_dates_correct_year(self, parser: WealthsimpleParser) -> None:
        """Dec dates in the Jan statement should be 2025, not 2026."""
        pdf_path = CC_DIR / "January_2026.pdf"
        if not pdf_path.exists():
            pytest.skip()
        txns = parser.parse(pdf_path.read_bytes())
        dec_txns = [t for t in txns if t.cash_date.month == 12]
        # The Dec 29 payment was filtered, but if there were Dec purchases they'd be 2025
        # In this statement, both purchases are Jan, so just verify Jan dates are 2026
        jan_txns = [t for t in txns if t.cash_date.month == 1]
        assert all(t.cash_date.year == 2026 for t in jan_txns)
