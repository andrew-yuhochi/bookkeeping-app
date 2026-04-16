"""Tests for fx/boc_client.py — Bank of Canada Valet API FX client.

All tests mock httpx — no real HTTP calls in the test suite.
"""

import json
from decimal import Decimal
from unittest.mock import MagicMock, patch

import httpx
import pytest

from fx.boc_client import FXClient, FXRateNotAvailableError


# ---------------------------------------------------------------------------
# Sample API responses
# ---------------------------------------------------------------------------

def _make_response(observations: list[dict], series: str = "FXHKDCAD") -> dict:
    """Build a minimal BoC Valet response dict."""
    return {
        "terms": {"url": "https://www.bankofcanada.ca/terms/"},
        "seriesDetail": {series: {"label": "HKD/CAD"}},
        "observations": observations,
    }


SAMPLE_HKD_OBSERVATIONS = [
    {"d": "2026-03-02", "FXHKDCAD": {"v": "0.1755"}},
    {"d": "2026-03-03", "FXHKDCAD": {"v": "0.1758"}},
    {"d": "2026-03-04", "FXHKDCAD": {"v": "0.1752"}},
    {"d": "2026-03-05", "FXHKDCAD": {"v": "0.1749"}},
    {"d": "2026-03-06", "FXHKDCAD": {"v": "0.1761"}},
    # Weekend gaps (null) — these should be excluded
    {"d": "2026-03-07", "FXHKDCAD": {"v": None}},
    {"d": "2026-03-08", "FXHKDCAD": {"v": None}},
    {"d": "2026-03-09", "FXHKDCAD": {"v": "0.1750"}},
    {"d": "2026-03-10", "FXHKDCAD": {"v": "0.1753"}},
]


def _mock_get_success(url: str, **kwargs) -> httpx.Response:
    """Mock httpx.get returning a successful response."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.json.return_value = _make_response(SAMPLE_HKD_OBSERVATIONS)
    response.raise_for_status = MagicMock()
    return response


def _mock_get_http_error(url: str, **kwargs) -> httpx.Response:
    """Mock httpx.get returning an HTTP 500 error."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = 500
    response.text = "Internal Server Error"
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server Error",
        request=MagicMock(),
        response=response,
    )
    return response


def _mock_get_empty(url: str, **kwargs) -> httpx.Response:
    """Mock httpx.get returning empty observations."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.json.return_value = _make_response([])
    response.raise_for_status = MagicMock()
    return response


def _mock_get_all_null(url: str, **kwargs) -> httpx.Response:
    """Mock httpx.get returning observations where all values are null."""
    observations = [
        {"d": "2026-03-07", "FXHKDCAD": {"v": None}},
        {"d": "2026-03-08", "FXHKDCAD": {"v": None}},
    ]
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.json.return_value = _make_response(observations)
    response.raise_for_status = MagicMock()
    return response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFXClientAveraging:
    """Test the averaging logic with mocked HTTP responses."""

    @patch("fx.boc_client.httpx.get", side_effect=_mock_get_success)
    @patch("fx.boc_client.time.sleep")  # Skip courtesy sleep in tests
    def test_average_excludes_nulls(self, mock_sleep, mock_get) -> None:
        client = FXClient()
        result = client.get_daily_average("HKD", 2026, 3)

        # 7 non-null values: 0.1755 + 0.1758 + 0.1752 + 0.1749 + 0.1761 + 0.1750 + 0.1753
        expected_sum = Decimal("0.1755") + Decimal("0.1758") + Decimal("0.1752") + \
                       Decimal("0.1749") + Decimal("0.1761") + Decimal("0.1750") + Decimal("0.1753")
        expected_avg = expected_sum / 7

        assert isinstance(result, Decimal)
        assert abs(result - expected_avg) < Decimal("0.0001")

    @patch("fx.boc_client.httpx.get", side_effect=_mock_get_success)
    @patch("fx.boc_client.time.sleep")
    def test_result_close_to_known_rate(self, mock_sleep, mock_get) -> None:
        """Average should be close to 0.1755 (within ±5%)."""
        client = FXClient()
        result = client.get_daily_average("HKD", 2026, 3)

        known_rate = Decimal("0.1755")
        tolerance = known_rate * Decimal("0.05")
        assert abs(result - known_rate) < tolerance


class TestFXClientCaching:
    """Test in-process caching behavior."""

    @patch("fx.boc_client.httpx.get", side_effect=_mock_get_success)
    @patch("fx.boc_client.time.sleep")
    def test_cached_response_no_second_request(self, mock_sleep, mock_get) -> None:
        client = FXClient()

        result1 = client.get_daily_average("HKD", 2026, 3)
        result2 = client.get_daily_average("HKD", 2026, 3)

        assert result1 == result2
        # Only one HTTP call made
        assert mock_get.call_count == 1

    @patch("fx.boc_client.httpx.get", side_effect=_mock_get_success)
    @patch("fx.boc_client.time.sleep")
    def test_different_periods_make_separate_requests(self, mock_sleep, mock_get) -> None:
        client = FXClient()

        client.get_daily_average("HKD", 2026, 3)
        client.get_daily_average("HKD", 2026, 4)

        assert mock_get.call_count == 2


class TestFXClientErrors:
    """Test error handling."""

    @patch("fx.boc_client.httpx.get", side_effect=_mock_get_http_error)
    @patch("fx.boc_client.time.sleep")
    def test_http_error_raises(self, mock_sleep, mock_get) -> None:
        client = FXClient()

        with pytest.raises(FXRateNotAvailableError, match="HTTP 500"):
            client.get_daily_average("HKD", 2026, 3)

    @patch("fx.boc_client.httpx.get", side_effect=_mock_get_empty)
    @patch("fx.boc_client.time.sleep")
    def test_empty_observations_raises(self, mock_sleep, mock_get) -> None:
        client = FXClient()

        with pytest.raises(FXRateNotAvailableError, match="No valid observations"):
            client.get_daily_average("HKD", 2026, 3)

    @patch("fx.boc_client.httpx.get", side_effect=_mock_get_all_null)
    @patch("fx.boc_client.time.sleep")
    def test_all_null_observations_raises(self, mock_sleep, mock_get) -> None:
        client = FXClient()

        with pytest.raises(FXRateNotAvailableError, match="all null"):
            client.get_daily_average("HKD", 2026, 3)

    @patch("fx.boc_client.httpx.get", side_effect=httpx.ConnectError("Connection refused"))
    @patch("fx.boc_client.time.sleep")
    def test_network_error_raises(self, mock_sleep, mock_get) -> None:
        client = FXClient()

        with pytest.raises(FXRateNotAvailableError, match="request failed"):
            client.get_daily_average("HKD", 2026, 3)


class TestFXClientCourtesySleep:
    """Test the courtesy sleep between requests."""

    @patch("fx.boc_client.httpx.get", side_effect=_mock_get_success)
    @patch("fx.boc_client.time.sleep")
    def test_sleep_called_between_requests(self, mock_sleep, mock_get) -> None:
        client = FXClient()

        client.get_daily_average("HKD", 2026, 3)
        client.get_daily_average("HKD", 2026, 4)  # Different period → new request

        # Sleep should have been called at least once (between the two requests)
        # The first call may or may not trigger sleep depending on _last_request_time
        # but the second call should trigger it
        assert mock_sleep.call_count >= 1


class TestFXClientURLConstruction:
    """Test that the correct URL is constructed."""

    @patch("fx.boc_client.httpx.get", side_effect=_mock_get_success)
    @patch("fx.boc_client.time.sleep")
    def test_url_format(self, mock_sleep, mock_get) -> None:
        client = FXClient(base_url="https://example.com/valet")
        client.get_daily_average("HKD", 2026, 3)

        called_url = mock_get.call_args[0][0]
        assert "observations/FXHKDCAD/json" in called_url
        assert "start_date=2026-03-01" in called_url
        assert "end_date=2026-03-31" in called_url

    @patch("fx.boc_client.httpx.get", side_effect=_mock_get_success)
    @patch("fx.boc_client.time.sleep")
    def test_currency_uppercased(self, mock_sleep, mock_get) -> None:
        client = FXClient(base_url="https://example.com/valet")
        client.get_daily_average("hkd", 2026, 3)

        called_url = mock_get.call_args[0][0]
        assert "FXHKDCAD" in called_url

    @patch("fx.boc_client.httpx.get", side_effect=_mock_get_success)
    @patch("fx.boc_client.time.sleep")
    def test_february_end_date(self, mock_sleep, mock_get) -> None:
        """February 2026 has 28 days."""
        client = FXClient(base_url="https://example.com/valet")
        client.get_daily_average("HKD", 2026, 2)

        called_url = mock_get.call_args[0][0]
        assert "end_date=2026-02-28" in called_url
