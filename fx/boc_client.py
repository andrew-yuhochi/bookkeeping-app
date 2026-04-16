"""Bank of Canada Valet API client for FX rate fallback.

Provides daily-average FX rates for HKD → CAD (and other currency pairs).
Uses the public BoC Valet API — no authentication required.

TDD Section 2.3, DATA-SOURCES.md Source 6.
"""

import logging
import time
from calendar import monthrange
from decimal import Decimal, InvalidOperation
from typing import Optional

import httpx

from src.config import settings

logger = logging.getLogger(__name__)


class FXRateNotAvailableError(Exception):
    """Raised when the BoC API returns no data for the requested period."""


class FXClient:
    """Bank of Canada Valet API wrapper for FX rate lookups.

    Fetches daily observations for a given currency pair and month,
    computes the arithmetic average (excluding null/weekend values),
    and caches the result in-process.

    Args:
        base_url: BoC Valet API base URL. Defaults to settings.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 10.0,
    ) -> None:
        self._base_url = (base_url or settings.boc_valet_base_url).rstrip("/")
        self._timeout = timeout
        self._cache: dict[tuple[str, int, int], Decimal] = {}
        self._last_request_time: float = 0.0

    def get_daily_average(
        self,
        currency: str,
        period_year: int,
        period_month: int,
    ) -> Decimal:
        """Compute the average daily FX rate for a currency pair and month.

        Args:
            currency: Source currency code (e.g., "HKD", "USD").
            period_year: Year of the statement period.
            period_month: Month of the statement period (1-12).

        Returns:
            Arithmetic average of daily observations as a Decimal.

        Raises:
            FXRateNotAvailableError: If the API returns no usable data.
        """
        cache_key = (currency.upper(), period_year, period_month)

        if cache_key in self._cache:
            logger.debug("FX cache hit: %s", cache_key)
            return self._cache[cache_key]

        # Courtesy sleep between distinct HTTP calls
        self._courtesy_sleep()

        series_name = f"FX{currency.upper()}CAD"
        _, last_day = monthrange(period_year, period_month)
        start_date = f"{period_year}-{period_month:02d}-01"
        end_date = f"{period_year}-{period_month:02d}-{last_day:02d}"

        url = (
            f"{self._base_url}/observations/{series_name}/json"
            f"?start_date={start_date}&end_date={end_date}"
        )

        logger.info("Fetching FX rates: %s", url)

        try:
            response = httpx.get(url, timeout=self._timeout)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise FXRateNotAvailableError(
                f"BoC API HTTP {e.response.status_code} for {series_name} "
                f"({start_date} to {end_date}): {e.response.text[:200]}"
            ) from e
        except httpx.RequestError as e:
            raise FXRateNotAvailableError(
                f"BoC API request failed for {series_name} "
                f"({start_date} to {end_date}): {e}"
            ) from e

        data = response.json()
        observations = data.get("observations", [])

        # Extract non-null rate values
        rates: list[Decimal] = []
        for obs in observations:
            rate_obj = obs.get(series_name, {})
            rate_str = rate_obj.get("v") if isinstance(rate_obj, dict) else None
            if rate_str is None:
                continue
            try:
                rates.append(Decimal(rate_str))
            except (InvalidOperation, ValueError):
                logger.warning(
                    "Skipping unparseable rate value %r on %s",
                    rate_str, obs.get("d"),
                )

        if not rates:
            raise FXRateNotAvailableError(
                f"No valid observations for {series_name} "
                f"({start_date} to {end_date}): "
                f"{len(observations)} observations returned, all null or invalid"
            )

        avg_rate = sum(rates) / len(rates)

        logger.info(
            "FX %s/%s-%02d: %d observations, average rate = %s",
            currency.upper(), period_year, period_month, len(rates), avg_rate,
        )

        self._cache[cache_key] = avg_rate
        return avg_rate

    def _courtesy_sleep(self) -> None:
        """Sleep 1 second between distinct HTTP requests (courtesy rate limit)."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if self._last_request_time > 0 and elapsed < 1.0:
            sleep_time = 1.0 - elapsed
            logger.debug("Courtesy sleep: %.2fs", sleep_time)
            time.sleep(sleep_time)
        self._last_request_time = time.monotonic()
