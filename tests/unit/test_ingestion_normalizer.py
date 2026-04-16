"""Tests for ingestion/normalizer.py — TransactionNormalizer and FX resolution.

FXClient is mocked in all tests — no real HTTP calls.
"""

from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from classifier.base import Transaction
from fx.boc_client import FXClient, FXRateNotAvailableError
from ingestion.normalizer import NormalizedTransaction, TransactionNormalizer
from parsers.models import ParsedTransaction


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_fx_client() -> MagicMock:
    """FXClient mock that returns a known HKD rate."""
    client = MagicMock(spec=FXClient)
    client.get_daily_average.return_value = Decimal("0.1755")
    return client


@pytest.fixture
def normalizer(mock_fx_client: MagicMock) -> TransactionNormalizer:
    return TransactionNormalizer(fx_client=mock_fx_client)


def _make_parsed(
    *,
    currency: str = "CAD",
    amount: Decimal = Decimal("50.00"),
    fx_rate: Decimal | None = None,
    description: str = "TEST MERCHANT",
    cash_date: date = date(2026, 3, 15),
    issuer: str = "MBNA",
) -> ParsedTransaction:
    return ParsedTransaction(
        issuer=issuer,
        cash_date=cash_date,
        description=description,
        original_amount=amount,
        original_currency=currency,
        fx_rate=fx_rate,
        fx_rate_source="statement" if fx_rate else None,
        cad_amount=None,
        statement_page=1,
    )


# ---------------------------------------------------------------------------
# CAD path
# ---------------------------------------------------------------------------

class TestCADTransactions:
    def test_cad_amount_equals_original(self, normalizer: TransactionNormalizer) -> None:
        parsed = _make_parsed(currency="CAD", amount=Decimal("123.45"))
        result = normalizer.normalize(parsed)

        assert result.cad_amount == Decimal("123.45")

    def test_cad_fx_rate_is_one(self, normalizer: TransactionNormalizer) -> None:
        parsed = _make_parsed(currency="CAD")
        result = normalizer.normalize(parsed)

        assert result.fx_rate == Decimal("1.0")

    def test_cad_fx_rate_source_is_statement(self, normalizer: TransactionNormalizer) -> None:
        parsed = _make_parsed(currency="CAD")
        result = normalizer.normalize(parsed)

        assert result.fx_rate_source == "statement"

    def test_cad_no_fx_review(self, normalizer: TransactionNormalizer) -> None:
        parsed = _make_parsed(currency="CAD")
        result = normalizer.normalize(parsed)

        assert result.needs_fx_review is False

    def test_cad_fx_client_not_called(
        self, normalizer: TransactionNormalizer, mock_fx_client: MagicMock
    ) -> None:
        parsed = _make_parsed(currency="CAD")
        normalizer.normalize(parsed)

        mock_fx_client.get_daily_average.assert_not_called()


# ---------------------------------------------------------------------------
# HKD with parsed rate (from statement)
# ---------------------------------------------------------------------------

class TestHKDWithParsedRate:
    def test_cad_amount_uses_statement_rate(self, normalizer: TransactionNormalizer) -> None:
        parsed = _make_parsed(
            currency="HKD",
            amount=Decimal("100.00"),
            fx_rate=Decimal("0.1750"),
        )
        result = normalizer.normalize(parsed)

        assert result.cad_amount == Decimal("17.5000")

    def test_fx_rate_source_is_statement(self, normalizer: TransactionNormalizer) -> None:
        parsed = _make_parsed(currency="HKD", fx_rate=Decimal("0.18"))
        result = normalizer.normalize(parsed)

        assert result.fx_rate_source == "statement"
        assert result.fx_rate == Decimal("0.18")

    def test_fx_client_not_called(
        self, normalizer: TransactionNormalizer, mock_fx_client: MagicMock
    ) -> None:
        parsed = _make_parsed(currency="HKD", fx_rate=Decimal("0.18"))
        normalizer.normalize(parsed)

        mock_fx_client.get_daily_average.assert_not_called()

    def test_no_fx_review(self, normalizer: TransactionNormalizer) -> None:
        parsed = _make_parsed(currency="HKD", fx_rate=Decimal("0.18"))
        result = normalizer.normalize(parsed)

        assert result.needs_fx_review is False


# ---------------------------------------------------------------------------
# HKD without parsed rate (BoC fallback)
# ---------------------------------------------------------------------------

class TestHKDWithoutParsedRate:
    def test_calls_fx_client(
        self, normalizer: TransactionNormalizer, mock_fx_client: MagicMock
    ) -> None:
        parsed = _make_parsed(
            currency="HKD",
            amount=Decimal("200.00"),
            cash_date=date(2026, 3, 15),
        )
        normalizer.normalize(parsed)

        mock_fx_client.get_daily_average.assert_called_once_with(
            currency="HKD", period_year=2026, period_month=3
        )

    def test_cad_amount_uses_boc_rate(
        self, normalizer: TransactionNormalizer
    ) -> None:
        parsed = _make_parsed(currency="HKD", amount=Decimal("200.00"))
        result = normalizer.normalize(parsed)

        # Mock returns 0.1755
        expected = Decimal("200.00") * Decimal("0.1755")
        assert result.cad_amount == expected

    def test_fx_rate_source_is_boc_average(self, normalizer: TransactionNormalizer) -> None:
        parsed = _make_parsed(currency="HKD")
        result = normalizer.normalize(parsed)

        assert result.fx_rate_source == "boc_average"
        assert result.fx_rate == Decimal("0.1755")

    def test_no_fx_review(self, normalizer: TransactionNormalizer) -> None:
        parsed = _make_parsed(currency="HKD")
        result = normalizer.normalize(parsed)

        assert result.needs_fx_review is False

    def test_boc_unavailable_falls_to_manual(self, mock_fx_client: MagicMock) -> None:
        mock_fx_client.get_daily_average.side_effect = FXRateNotAvailableError("API down")
        normalizer = TransactionNormalizer(fx_client=mock_fx_client)

        parsed = _make_parsed(currency="HKD")
        result = normalizer.normalize(parsed)

        assert result.cad_amount is None
        assert result.fx_rate is None
        assert result.fx_rate_source == "manual"
        assert result.needs_fx_review is True


# ---------------------------------------------------------------------------
# USD transactions (always manual)
# ---------------------------------------------------------------------------

class TestUSDTransactions:
    def test_cad_amount_is_none(self, normalizer: TransactionNormalizer) -> None:
        parsed = _make_parsed(currency="USD", amount=Decimal("75.00"))
        result = normalizer.normalize(parsed)

        assert result.cad_amount is None

    def test_fx_rate_source_is_manual(self, normalizer: TransactionNormalizer) -> None:
        parsed = _make_parsed(currency="USD")
        result = normalizer.normalize(parsed)

        assert result.fx_rate_source == "manual"

    def test_needs_fx_review(self, normalizer: TransactionNormalizer) -> None:
        parsed = _make_parsed(currency="USD")
        result = normalizer.normalize(parsed)

        assert result.needs_fx_review is True

    def test_fx_client_not_called(
        self, normalizer: TransactionNormalizer, mock_fx_client: MagicMock
    ) -> None:
        parsed = _make_parsed(currency="USD")
        normalizer.normalize(parsed)

        mock_fx_client.get_daily_average.assert_not_called()

    def test_usd_with_statement_rate_uses_it(self, normalizer: TransactionNormalizer) -> None:
        """If the statement already provides a USD→CAD rate, use it."""
        parsed = _make_parsed(
            currency="USD",
            amount=Decimal("50.00"),
            fx_rate=Decimal("1.3500"),
        )
        result = normalizer.normalize(parsed)

        assert result.cad_amount == Decimal("67.5000")
        assert result.fx_rate_source == "statement"
        assert result.needs_fx_review is False


# ---------------------------------------------------------------------------
# Other currencies (manual)
# ---------------------------------------------------------------------------

class TestOtherCurrencies:
    def test_eur_is_manual(self, normalizer: TransactionNormalizer) -> None:
        parsed = _make_parsed(currency="EUR")
        result = normalizer.normalize(parsed)

        assert result.cad_amount is None
        assert result.fx_rate_source == "manual"
        assert result.needs_fx_review is True


# ---------------------------------------------------------------------------
# Accounting period
# ---------------------------------------------------------------------------

class TestAccountingPeriod:
    def test_defaults_to_cash_date(self, normalizer: TransactionNormalizer) -> None:
        parsed = _make_parsed(cash_date=date(2026, 3, 15))
        result = normalizer.normalize(parsed)

        assert result.accounting_period_year == 2026
        assert result.accounting_period_month == 3

    def test_december_date(self, normalizer: TransactionNormalizer) -> None:
        parsed = _make_parsed(cash_date=date(2025, 12, 31))
        result = normalizer.normalize(parsed)

        assert result.accounting_period_year == 2025
        assert result.accounting_period_month == 12

    def test_january_date(self, normalizer: TransactionNormalizer) -> None:
        parsed = _make_parsed(cash_date=date(2026, 1, 1))
        result = normalizer.normalize(parsed)

        assert result.accounting_period_year == 2026
        assert result.accounting_period_month == 1


# ---------------------------------------------------------------------------
# Merchant normalization integration
# ---------------------------------------------------------------------------

class TestMerchantNormalization:
    def test_description_normalized(self, normalizer: TransactionNormalizer) -> None:
        parsed = _make_parsed(description="STARBUCKS #1234 TORONTO ON")
        result = normalizer.normalize(parsed)

        assert result.transaction.normalized_description == "starbucks"
        assert result.transaction.description == "STARBUCKS #1234 TORONTO ON"

    def test_classifier_transaction_has_correct_fields(
        self, normalizer: TransactionNormalizer
    ) -> None:
        parsed = _make_parsed(
            description="WALMART",
            currency="CAD",
            amount=Decimal("99.99"),
            issuer="MBNA",
        )
        result = normalizer.normalize(parsed)

        txn = result.transaction
        assert isinstance(txn, Transaction)
        assert txn.description == "WALMART"
        assert txn.normalized_description == "walmart"
        assert txn.amount_cad == 99.99
        assert txn.original_currency == "CAD"
        assert txn.issuer == "MBNA"


# ---------------------------------------------------------------------------
# Batch normalization
# ---------------------------------------------------------------------------

class TestBatchNormalization:
    def test_batch_preserves_order(self, normalizer: TransactionNormalizer) -> None:
        items = [
            _make_parsed(description=f"MERCHANT_{i}", amount=Decimal(str(i * 10)))
            for i in range(5)
        ]
        results = normalizer.normalize_batch(items)

        assert len(results) == 5
        for i, r in enumerate(results):
            assert r.transaction.description == f"MERCHANT_{i}"

    def test_batch_empty(self, normalizer: TransactionNormalizer) -> None:
        assert normalizer.normalize_batch([]) == []
