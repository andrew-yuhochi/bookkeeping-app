"""Transaction normalizer — bridge between parsing and classification.

Converts ParsedTransaction objects (from issuer parsers) into classifier
Transaction objects, resolving FX rates along the way.

TDD Section 2.3 (currency handling) and Section 2.4 (accounting period).
"""

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from classifier.base import Transaction
from classifier.normalizer import normalize_merchant
from fx.boc_client import FXClient, FXRateNotAvailableError
from parsers.models import ParsedTransaction

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NormalizedTransaction:
    """A fully resolved transaction ready for DB insertion.

    Contains both the classifier-ready Transaction and the resolved
    FX fields needed for the DB row.
    """

    transaction: Transaction
    cad_amount: Optional[Decimal]
    fx_rate: Optional[Decimal]
    fx_rate_source: str
    accounting_period_year: int
    accounting_period_month: int
    needs_fx_review: bool


class TransactionNormalizer:
    """Converts ParsedTransaction → NormalizedTransaction with FX resolution.

    FX resolution order (TDD Section 2.3):
      1. CAD → passthrough (rate=1.0, source="statement")
      2. Foreign currency with parsed rate → use statement rate
      3. HKD without rate → BoC Valet daily average
      4. USD / other without rate → manual (needs_review=True)

    Args:
        fx_client: FXClient instance for BoC Valet lookups.
    """

    def __init__(self, fx_client: Optional[FXClient] = None) -> None:
        self._fx_client = fx_client or FXClient()

    def normalize(self, parsed: ParsedTransaction) -> NormalizedTransaction:
        """Normalize a single parsed transaction.

        Resolves FX rates and produces a classifier-ready Transaction.
        """
        normalized_desc = normalize_merchant(parsed.description)

        # Accounting period defaults to cash_date's calendar month
        acct_year = parsed.cash_date.year
        acct_month = parsed.cash_date.month

        # FX resolution
        cad_amount, fx_rate, fx_rate_source, needs_fx_review = self._resolve_fx(parsed)

        # Build classifier Transaction
        txn = Transaction(
            description=parsed.description,
            normalized_description=normalized_desc,
            amount_cad=float(cad_amount) if cad_amount is not None else 0.0,
            original_currency=parsed.original_currency,
            issuer=parsed.issuer,
        )

        return NormalizedTransaction(
            transaction=txn,
            cad_amount=cad_amount,
            fx_rate=fx_rate,
            fx_rate_source=fx_rate_source,
            accounting_period_year=acct_year,
            accounting_period_month=acct_month,
            needs_fx_review=needs_fx_review,
        )

    def normalize_batch(
        self, parsed_list: list[ParsedTransaction]
    ) -> list[NormalizedTransaction]:
        """Normalize a list of parsed transactions."""
        return [self.normalize(p) for p in parsed_list]

    def _resolve_fx(
        self, parsed: ParsedTransaction
    ) -> tuple[Optional[Decimal], Optional[Decimal], str, bool]:
        """Resolve FX rate for a parsed transaction.

        Returns:
            (cad_amount, fx_rate, fx_rate_source, needs_fx_review)
        """
        currency = parsed.original_currency.upper()

        # Path 1: CAD — passthrough
        if currency == "CAD":
            return (
                parsed.original_amount,
                Decimal("1.0"),
                "statement",
                False,
            )

        # Path 2: Foreign currency with statement-parsed rate
        if parsed.fx_rate is not None:
            cad_amount = parsed.original_amount * parsed.fx_rate
            return (
                cad_amount,
                parsed.fx_rate,
                "statement",
                False,
            )

        # Path 3: HKD without rate — BoC Valet fallback
        if currency == "HKD":
            try:
                boc_rate = self._fx_client.get_daily_average(
                    currency="HKD",
                    period_year=parsed.cash_date.year,
                    period_month=parsed.cash_date.month,
                )
                cad_amount = parsed.original_amount * boc_rate
                return (
                    cad_amount,
                    boc_rate,
                    "boc_average",
                    False,
                )
            except FXRateNotAvailableError:
                logger.warning(
                    "BoC rate unavailable for HKD %s-%02d, flagging for manual entry",
                    parsed.cash_date.year,
                    parsed.cash_date.month,
                )
                return (None, None, "manual", True)

        # Path 4: USD and all other currencies — manual
        logger.info(
            "Currency %s requires manual FX rate entry: %s",
            currency, parsed.description,
        )
        return (None, None, "manual", True)
