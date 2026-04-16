from abc import ABC, abstractmethod

from parsers.models import ParsedTransaction


class UnknownIssuerError(Exception):
    """Raised when no registered parser matches the given file."""


class IssuerParser(ABC):
    """Abstract base class for per-issuer PDF statement parsers.

    Adding a new issuer requires:
    1. Create parsers/<issuer>.py implementing this ABC.
    2. Register it in parsers/registry.py (append to REGISTERED_PARSERS).
    3. No other changes.

    This is also the future open-banking migration path (TDD 2.1):
    an API-based connector overrides detect() and parse() internally.
    """

    issuer_name: str

    @abstractmethod
    def detect(self, filename: str, first_page_text: str) -> bool:
        """Return True if this parser handles the given file.

        Detection uses filename heuristics and/or first-page text signatures.
        """

    @abstractmethod
    def parse(self, pdf_bytes: bytes) -> list[ParsedTransaction]:
        """Extract transactions from PDF bytes.

        Must handle partial failures gracefully: if one page fails,
        return successfully parsed pages and record errors.
        """
