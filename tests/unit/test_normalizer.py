"""Tests for classifier/normalizer.py — merchant string normalization."""

import pytest

from classifier.normalizer import normalize_merchant


class TestNormalizeMerchant:
    """Test deterministic normalization of raw merchant descriptions."""

    def test_empty_string(self) -> None:
        assert normalize_merchant("") == ""

    def test_whitespace_only(self) -> None:
        assert normalize_merchant("   ") == ""

    def test_none_like(self) -> None:
        # Empty after strip
        assert normalize_merchant("  \t\n  ") == ""

    def test_simple_lowercase(self) -> None:
        assert normalize_merchant("Walmart") == "walmart"

    def test_deterministic(self) -> None:
        """Same input always produces same output."""
        raw = "STARBUCKS #1234 TORONTO ON"
        result1 = normalize_merchant(raw)
        result2 = normalize_merchant(raw)
        assert result1 == result2

    # --- Branch/store ID stripping ---

    def test_strip_hash_store_id(self) -> None:
        assert normalize_merchant("STARBUCKS #1234") == "starbucks"

    def test_strip_numeric_store_id(self) -> None:
        """Standalone 3-6 digit numbers are stripped."""
        assert normalize_merchant("LOBLAWS 0042") == "loblaws"

    def test_preserve_short_numbers(self) -> None:
        """1-2 digit numbers should not be stripped (e.g., 'Pho 37')."""
        assert normalize_merchant("Pho 37") == "pho 37"

    # --- Same-merchant normalization (acceptance criteria) ---

    def test_starbucks_same_key(self) -> None:
        """'STARBUCKS #1234 TORONTO ON' and 'STARBUCKS 5678 OTTAWA' normalize to same key."""
        key1 = normalize_merchant("STARBUCKS #1234 TORONTO ON")
        key2 = normalize_merchant("STARBUCKS 5678 OTTAWA ON")
        assert key1 == key2
        assert key1 == "starbucks"

    def test_loblaws_purchase_authorization(self) -> None:
        """'PURCHASE AUTHORIZATION LOBLAWS 0042 TORONTO ON' normalizes to 'loblaws'."""
        result = normalize_merchant("PURCHASE AUTHORIZATION LOBLAWS 0042 TORONTO ON")
        assert result == "loblaws"

    # --- Card-network noise stripping ---

    def test_strip_purchase(self) -> None:
        assert normalize_merchant("PURCHASE COSTCO") == "costco"

    def test_strip_pos(self) -> None:
        assert normalize_merchant("POS TIM HORTONS") == "tim hortons"

    def test_strip_payment(self) -> None:
        assert normalize_merchant("PAYMENT H-MART") == "h-mart"

    def test_strip_authorization(self) -> None:
        assert normalize_merchant("AUTHORIZATION SHOPPERS") == "shoppers"

    def test_strip_multiple_noise_words(self) -> None:
        result = normalize_merchant("PURCHASE AUTHORIZATION VISA WALMART")
        assert result == "walmart"

    # --- Province code and city stripping ---

    def test_strip_trailing_province(self) -> None:
        assert normalize_merchant("REAL CDN SUPERSTORE BC") == "real cdn superstore"

    def test_strip_city_and_province(self) -> None:
        assert normalize_merchant("SHOPPERS DRUG MART VANCOUVER BC") == "shoppers drug mart"

    def test_strip_toronto_on(self) -> None:
        assert normalize_merchant("H-MART TORONTO ON") == "h-mart"

    # --- Chinese characters pass-through ---

    def test_chinese_characters_passthrough(self) -> None:
        """Chinese merchant names should pass through normalization unchanged (lowered)."""
        assert normalize_merchant("大統華") == "大統華"

    def test_mixed_chinese_english(self) -> None:
        result = normalize_merchant("T&T 大統華 #123")
        assert "大統華" in result
        assert "#123" not in result

    # --- Amount in description ---

    def test_amount_in_description(self) -> None:
        """Amounts like $12.34 should be stripped as noise or kept as-is."""
        # A price-like pattern in the description shouldn't break normalization
        result = normalize_merchant("UBER EATS 15.99")
        assert "uber eats" in result

    # --- Edge cases ---

    def test_already_clean(self) -> None:
        """Already-clean historical data should normalize trivially."""
        assert normalize_merchant("Costco Food") == "costco food"
        assert normalize_merchant("TnT") == "tnt"
        assert normalize_merchant("Uber Eat") == "uber eat"

    def test_multiple_spaces_collapsed(self) -> None:
        assert normalize_merchant("TIM   HORTONS") == "tim hortons"

    def test_leading_trailing_whitespace(self) -> None:
        assert normalize_merchant("  WALMART  ") == "walmart"
