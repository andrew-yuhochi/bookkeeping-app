import logging

from parsers.base import IssuerParser, UnknownIssuerError

logger = logging.getLogger(__name__)

# Append new parser instances here. Order matters for detection priority.
REGISTERED_PARSERS: list[IssuerParser] = []


class ParserRegistry:
    """Dispatches PDF files to the correct per-issuer parser.

    Detection tries each registered parser in order and returns the
    first match. If no parser matches, raises UnknownIssuerError.
    """

    def __init__(self, parsers: list[IssuerParser] | None = None) -> None:
        self._parsers = parsers if parsers is not None else REGISTERED_PARSERS

    def detect_issuer(self, filename: str, first_page_text: str) -> IssuerParser:
        """Find the parser that handles this file.

        Args:
            filename: Original filename of the uploaded PDF.
            first_page_text: Text extracted from the first page of the PDF.

        Returns:
            The matching IssuerParser instance.

        Raises:
            UnknownIssuerError: If no registered parser matches.
        """
        for parser in self._parsers:
            if parser.detect(filename, first_page_text):
                logger.info(
                    "Detected issuer '%s' for file '%s'",
                    parser.issuer_name,
                    filename,
                )
                return parser

        raise UnknownIssuerError(
            f"No registered parser matches file '{filename}'. "
            f"Registered issuers: {[p.issuer_name for p in self._parsers]}"
        )
