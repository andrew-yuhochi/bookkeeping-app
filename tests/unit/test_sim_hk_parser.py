"""Unit tests for parsers/sim_hk.py — no real PDF required.

Tests cover: _parse_amount, _resolve_year, detect(), _extract_statement_date(),
_parse_page(), and parse() via mocked pdfplumber.
"""

from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from parsers.sim_hk import (
    SIMHKParser,
    _MONTHS,
    _parse_amount,
    _resolve_year,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class _MockPage:
    """Minimal stand-in for a pdfplumber page."""

    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


def _mock_pdf(page_texts: list[str]) -> MagicMock:
    """Return a mock pdfplumber PDF context manager with given page texts."""
    pages = [_MockPage(t) for t in page_texts]
    m = MagicMock()
    m.pages = pages
    m.__enter__ = lambda self: self
    m.__exit__ = MagicMock(return_value=False)
    return m


# Representative page text matching the documented SIM HK format
_STMT_PAGE_HEADER = "Statement Date 22 Jan 2026\n"

_PURCHASE_BLOCK = (
    "Trans Date  Posting Date  Description\n"
    "APPLE STORE\n"
    "05 Jan CANADA ONLINE CAN CAD 29.99 238.12\n"
    "*EXCHANGE RATE:7.94013\n"
)

_CREDIT_BLOCK = (
    "Trans Date  Posting Date  Description\n"
    "10 Jan AUTO PAYMENT - THANK YOU 2,000.00 CR\n"
)

_PURCHASE_NO_FX_BLOCK = (
    "Trans Date  Posting Date  Description\n"
    "15 Jan HONG KONG HKG HKD 500.00 500.00\n"
)

_CLOSING = "GRAND TOTAL 5,000.00\n"


# ---------------------------------------------------------------------------
# _parse_amount
# ---------------------------------------------------------------------------


class TestParseAmount:
    def test_simple_decimal(self) -> None:
        assert _parse_amount("29.99") == Decimal("29.99")

    def test_comma_thousands(self) -> None:
        assert _parse_amount("1,308.98") == Decimal("1308.98")

    def test_large_amount(self) -> None:
        assert _parse_amount("12,345,678.00") == Decimal("12345678.00")

    def test_zero(self) -> None:
        assert _parse_amount("0.00") == Decimal("0.00")


# ---------------------------------------------------------------------------
# _resolve_year
# ---------------------------------------------------------------------------


class TestResolveYear:
    def test_same_month_as_statement(self) -> None:
        assert _resolve_year(5, "Jan", 2026, 1) == 2026

    def test_one_month_before_statement(self) -> None:
        assert _resolve_year(28, "Dec", 2026, 1) == 2025

    def test_normal_mid_year(self) -> None:
        assert _resolve_year(15, "Jun", 2026, 6) == 2026

    def test_december_transaction_on_jan_statement(self) -> None:
        # Dec month (12) > Jan statement_month (1) + 2 → previous year
        assert _resolve_year(31, "Dec", 2026, 1) == 2025

    def test_november_on_february_statement(self) -> None:
        # Nov (11) > Feb (2) + 2 = 4 → previous year
        assert _resolve_year(30, "Nov", 2026, 2) == 2025

    def test_transaction_in_same_month_as_statement(self) -> None:
        assert _resolve_year(1, "Mar", 2026, 3) == 2026


# ---------------------------------------------------------------------------
# SIMHKParser.detect()
# ---------------------------------------------------------------------------


class TestDetect:
    def setup_method(self) -> None:
        self.parser = SIMHKParser()

    def test_detect_sim_in_filename(self) -> None:
        assert self.parser.detect("sim_2026-01.pdf", "") is True

    def test_detect_sim_mixed_case(self) -> None:
        assert self.parser.detect("SIM_statement.pdf", "") is True

    def test_detect_chinese_monthly_marker(self) -> None:
        assert self.parser.detect("信用卡月結單.pdf", "") is True

    def test_detect_sim_world_mastercard_in_text(self) -> None:
        assert self.parser.detect("statement.pdf", "sim World MasterCard® Card No.") is True

    def test_detect_sim_credit_card_in_text(self) -> None:
        assert self.parser.detect("statement.pdf", "sim Credit Card Account Summary") is True

    def test_no_match_unrelated_filename_and_text(self) -> None:
        assert self.parser.detect("mbna_2026.pdf", "TD Bank Statement") is False

    def test_no_match_empty_inputs(self) -> None:
        assert self.parser.detect("statement.pdf", "") is False


# ---------------------------------------------------------------------------
# SIMHKParser._extract_statement_date()
# ---------------------------------------------------------------------------


class TestExtractStatementDate:
    def setup_method(self) -> None:
        self.parser = SIMHKParser()

    def test_statement_date_on_same_line(self) -> None:
        page = _MockPage("Statement Date 22 Jan 2026\n")
        year, month = self.parser._extract_statement_date(page)
        assert year == 2026
        assert month == 1

    def test_chinese_label(self) -> None:
        page = _MockPage("月結單截數日 15 Mar 2026\n")
        year, month = self.parser._extract_statement_date(page)
        assert year == 2026
        assert month == 3

    def test_fallback_date_anywhere_on_page(self) -> None:
        # No "Statement Date" label, but a date exists on the page
        page = _MockPage("Account Summary\n05 Feb 2026 Some text\n")
        year, month = self.parser._extract_statement_date(page)
        assert year == 2026
        assert month == 2

    def test_no_date_falls_back_to_today(self) -> None:
        page = _MockPage("No date information here at all.\n")
        year, month = self.parser._extract_statement_date(page)
        today = date.today()
        assert year == today.year
        assert month == today.month

    def test_empty_page_falls_back_to_today(self) -> None:
        page = _MockPage("")
        year, month = self.parser._extract_statement_date(page)
        today = date.today()
        assert year == today.year


# ---------------------------------------------------------------------------
# SIMHKParser._parse_page()
# ---------------------------------------------------------------------------


class TestParsePage:
    def setup_method(self) -> None:
        self.parser = SIMHKParser()

    def test_empty_text_returns_empty(self) -> None:
        page = _MockPage("")
        result = self.parser._parse_page(page, 1, 2026, 1)
        assert result == []

    def test_no_transaction_section_returns_empty(self) -> None:
        # No "Trans Date" marker
        page = _MockPage("Account summary page only.\nsome text\n")
        result = self.parser._parse_page(page, 1, 2026, 1)
        assert result == []

    def test_purchase_with_fx_rate(self) -> None:
        text = (
            "Trans Date  Posting Date  Description\n"
            "05 Jan CANADA ONLINE CAN CAD 29.99 238.12\n"
            "*EXCHANGE RATE:7.94013\n"
            "GRAND TOTAL 238.12\n"
        )
        page = _MockPage(text)
        result = self.parser._parse_page(page, 1, 2026, 1)
        assert len(result) == 1
        txn = result[0]
        assert txn.issuer == "SIM_HK"
        assert txn.cash_date == date(2026, 1, 5)
        assert txn.original_currency == "CAD"
        assert txn.original_amount == Decimal("29.99")
        assert txn.fx_rate == Decimal("7.94013")
        assert txn.fx_rate_source == "statement"
        assert txn.statement_page == 1

    def test_purchase_without_fx_rate(self) -> None:
        text = (
            "Trans Date  Posting Date  Description\n"
            "15 Jan HONG KONG HKG HKD 500.00 500.00\n"
            "GRAND TOTAL 500.00\n"
        )
        page = _MockPage(text)
        result = self.parser._parse_page(page, 1, 2026, 1)
        assert len(result) == 1
        txn = result[0]
        assert txn.fx_rate is None
        assert txn.fx_rate_source is None
        assert txn.original_currency == "HKD"

    def test_merchant_name_on_preceding_line(self) -> None:
        text = (
            "Trans Date\n"
            "APPLE STORE\n"
            "05 Jan CANADA ONLINE CAN CAD 29.99 238.12\n"
            "GRAND TOTAL 238.12\n"
        )
        page = _MockPage(text)
        result = self.parser._parse_page(page, 1, 2026, 1)
        assert len(result) == 1
        # Merchant from preceding line is used as description
        assert result[0].description == "APPLE STORE"

    def test_credit_line(self) -> None:
        text = (
            "Trans Date  Posting Date\n"
            "10 Jan AUTO PAYMENT - THANK YOU 2,000.00 CR\n"
            "GRAND TOTAL 0.00\n"
        )
        page = _MockPage(text)
        result = self.parser._parse_page(page, 1, 2026, 1)
        assert len(result) == 1
        txn = result[0]
        assert txn.cash_date == date(2026, 1, 10)
        assert txn.original_amount < 0  # credits are negative
        assert txn.original_currency == "HKD"
        assert txn.fx_rate == Decimal("1")

    def test_skip_pattern_previous_balance(self) -> None:
        text = (
            "Trans Date\n"
            "Previous Balance 1,500.00\n"
            "GRAND TOTAL 1,500.00\n"
        )
        page = _MockPage(text)
        result = self.parser._parse_page(page, 1, 2026, 1)
        assert result == []

    def test_grand_total_ends_section(self) -> None:
        text = (
            "Trans Date\n"
            "05 Jan SOMEWHERE CAD CAD 10.00 10.00\n"
            "GRAND TOTAL 10.00\n"
            "12 Jan ANOTHER CAD CAD 20.00 20.00\n"  # after GRAND TOTAL — ignored
        )
        page = _MockPage(text)
        result = self.parser._parse_page(page, 1, 2026, 1)
        # Only the first transaction should be parsed (GRAND TOTAL stops parsing)
        assert len(result) == 1

    def test_end_of_statement_ends_section(self) -> None:
        text = (
            "Trans Date\n"
            "05 Jan SOMEWHERE CAD CAD 10.00 10.00\n"
            "End of Statement\n"
            "12 Jan ANOTHER CAD CAD 20.00 20.00\n"
        )
        page = _MockPage(text)
        result = self.parser._parse_page(page, 1, 2026, 1)
        assert len(result) == 1

    def test_empty_lines_skipped(self) -> None:
        text = (
            "Trans Date\n"
            "\n"
            "\n"
            "05 Jan SOMEWHERE CAD CAD 10.00 10.00\n"
            "GRAND TOTAL 10.00\n"
        )
        page = _MockPage(text)
        result = self.parser._parse_page(page, 1, 2026, 1)
        assert len(result) == 1

    def test_multiple_transactions(self) -> None:
        text = (
            "Trans Date\n"
            "05 Jan CANADA CAD CAD 10.00 10.00\n"
            "12 Jan JAPAN TOKYO JPN JPY 1,000.00 55.00\n"
            "*EXCHANGE RATE:0.05500\n"
            "20 Jan PAYMENT 500.00 CR\n"
            "GRAND TOTAL 9,000.00\n"
        )
        page = _MockPage(text)
        result = self.parser._parse_page(page, 2, 2026, 1)
        assert len(result) == 3
        assert all(t.statement_page == 2 for t in result)

    def test_december_transaction_on_january_statement(self) -> None:
        text = (
            "Trans Date\n"
            "31 Dec CANADA CAD CAD 50.00 50.00\n"
            "GRAND TOTAL 50.00\n"
        )
        page = _MockPage(text)
        result = self.parser._parse_page(page, 1, 2026, 1)
        assert len(result) == 1
        assert result[0].cash_date.year == 2025  # Dec 2025 on Jan 2026 statement

    def test_chinese_label_in_transaction_header(self) -> None:
        text = (
            "交易日期  Posting Date\n"
            "05 Jan CANADA CAD CAD 10.00 10.00\n"
            "GRAND TOTAL 10.00\n"
        )
        page = _MockPage(text)
        result = self.parser._parse_page(page, 1, 2026, 1)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# SIMHKParser.parse() — mocked pdfplumber
# ---------------------------------------------------------------------------


class TestParse:
    def setup_method(self) -> None:
        self.parser = SIMHKParser()

    @patch("parsers.sim_hk.pdfplumber.open")
    def test_invalid_bytes_returns_empty(self, mock_open: MagicMock) -> None:
        mock_open.side_effect = Exception("not a PDF")
        result = self.parser.parse(b"garbage bytes")
        assert result == []

    @patch("parsers.sim_hk.pdfplumber.open")
    def test_full_parse_via_mock(self, mock_open: MagicMock) -> None:
        page1_text = (
            "Statement Date 22 Jan 2026\n"
            "Trans Date  Posting Date\n"
            "APPLE STORE\n"
            "05 Jan CANADA ONLINE CAN CAD 29.99 238.12\n"
            "*EXCHANGE RATE:7.94013\n"
            "GRAND TOTAL 238.12\n"
        )
        mock_open.return_value = _mock_pdf([page1_text])
        result = self.parser.parse(b"fake pdf bytes")
        assert len(result) == 1
        assert result[0].issuer == "SIM_HK"
        assert result[0].original_currency == "CAD"

    @patch("parsers.sim_hk.pdfplumber.open")
    def test_page_exception_is_isolated(self, mock_open: MagicMock) -> None:
        """A parsing failure on page 2 should not prevent page 1 results."""
        page1_text = (
            "Statement Date 22 Jan 2026\n"
            "Trans Date\n"
            "05 Jan SOMEWHERE CAD CAD 50.00 50.00\n"
            "GRAND TOTAL 50.00\n"
        )

        good_page = _MockPage(page1_text)
        bad_page = MagicMock()
        bad_page.extract_text.side_effect = RuntimeError("pdfplumber error")

        mock_pdf = MagicMock()
        mock_pdf.pages = [good_page, bad_page]
        mock_pdf.__enter__ = lambda self: self
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_open.return_value = mock_pdf

        result = self.parser.parse(b"fake bytes")
        # Page 1 results should still be returned
        assert len(result) == 1

    @patch("parsers.sim_hk.pdfplumber.open")
    def test_empty_pdf_returns_empty(self, mock_open: MagicMock) -> None:
        page_text = "Statement Date 22 Jan 2026\nNothing else.\n"
        mock_open.return_value = _mock_pdf([page_text])
        result = self.parser.parse(b"fake bytes")
        assert result == []
