"""Merchant string normalization — Layer 0 preprocessing.

Produces a deterministic normalized key from raw merchant descriptions,
stripping card-network noise, branch/store IDs, province codes, and
city name trailing patterns.

TDD Section 2.2 Layer 0.
"""

import re

# Card-network noise words to strip (order matters — longer phrases first)
_NOISE_PHRASES: list[str] = [
    "PURCHASE AUTHORIZATION",
    "PURCHASE",
    "AUTHORIZATION",
    "PAYMENT",
    "POS",
    "DEBIT",
    "CREDIT",
    "INTERAC",
    "VISA",
    "MASTERCARD",
    "MC",
]

# Canadian province codes (2-letter)
_PROVINCE_CODES: set[str] = {
    "AB", "BC", "MB", "NB", "NL", "NS", "NT", "NU", "ON", "PE", "QC", "SK", "YT",
}

_PROVINCE_PATTERN = "|".join(_PROVINCE_CODES)

# Regex: branch/store IDs — #1234 or standalone 3-6 digit numbers
_BRANCH_RE = re.compile(r"\s*#\d+|\s+\d{3,6}\b")

# Known Canadian cities that commonly appear on credit card statements.
# Used for targeted stripping — avoids false positives on merchant words.
_KNOWN_CITIES: set[str] = {
    "TORONTO", "VANCOUVER", "MONTREAL", "CALGARY", "EDMONTON", "OTTAWA",
    "WINNIPEG", "QUEBEC", "HAMILTON", "KITCHENER", "LONDON", "VICTORIA",
    "HALIFAX", "OSHAWA", "WINDSOR", "SASKATOON", "REGINA", "BARRIE",
    "SHERBROOKE", "KELOWNA", "ABBOTSFORD", "KINGSTON", "SUDBURY",
    "BURNABY", "RICHMOND", "SURREY", "COQUITLAM", "MISSISSAUGA",
    "BRAMPTON", "MARKHAM", "SCARBOROUGH", "ETOBICOKE", "NORTH YORK",
    "NORTH VANCOUVER", "WEST VANCOUVER", "NEW WESTMINSTER", "LANGLEY",
    "DELTA", "MAPLE RIDGE", "PORT MOODY", "PORT COQUITLAM",
}

# Build regex: trailing "<KNOWN_CITY> <PROVINCE>" or trailing "<PROVINCE>" alone
_KNOWN_CITIES_PATTERN = "|".join(
    re.escape(c) for c in sorted(_KNOWN_CITIES, key=len, reverse=True)
)

# Regex: trailing known city + province
_CITY_PROVINCE_RE = re.compile(
    r"\s+(?:" + _KNOWN_CITIES_PATTERN + r")\s+(?:" + _PROVINCE_PATTERN + r")\s*$",
    re.IGNORECASE,
)

# Regex: trailing province code alone at end of string (no city)
_TRAILING_PROVINCE_RE = re.compile(
    r"\s+(?:" + _PROVINCE_PATTERN + r")\s*$",
    re.IGNORECASE,
)

# Regex: collapse multiple spaces
_MULTI_SPACE_RE = re.compile(r"\s{2,}")


def normalize_merchant(raw: str) -> str:
    """Normalize a raw merchant description to a canonical cache key.

    Deterministic: same input always produces the same output.
    Handles:
      - Lowercasing
      - Stripping card-network noise words
      - Stripping branch/store IDs (#1234, standalone 3-6 digit numbers)
      - Stripping province codes and city name trailing patterns
      - Collapsing whitespace

    Non-ASCII characters (e.g., Chinese) are passed through unchanged.

    Args:
        raw: Raw merchant description string from a PDF statement or manual entry.

    Returns:
        Normalized merchant key string. Empty string if input is empty/whitespace.
    """
    if not raw or not raw.strip():
        return ""

    text = raw.strip()

    # Strip card-network noise phrases (case-insensitive)
    for phrase in _NOISE_PHRASES:
        text = re.sub(
            r"\b" + re.escape(phrase) + r"\b",
            "",
            text,
            flags=re.IGNORECASE,
        )

    # Strip branch/store IDs
    text = _BRANCH_RE.sub("", text)

    # Strip trailing city + province (e.g., "TORONTO ON")
    text = _CITY_PROVINCE_RE.sub("", text)

    # Strip trailing province code alone
    text = _TRAILING_PROVINCE_RE.sub("", text)

    # Collapse whitespace and lowercase
    text = _MULTI_SPACE_RE.sub(" ", text).strip().lower()

    return text
