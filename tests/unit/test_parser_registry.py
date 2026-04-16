from datetime import date
from decimal import Decimal

import pytest

from parsers.base import IssuerParser, UnknownIssuerError
from parsers.models import ParsedTransaction
from parsers.registry import ParserRegistry


class FakeParserA(IssuerParser):
    issuer_name = "FAKE_A"

    def detect(self, filename: str, first_page_text: str) -> bool:
        return "fake_a" in filename.lower()

    def parse(self, pdf_bytes: bytes) -> list[ParsedTransaction]:
        return [
            ParsedTransaction(
                issuer=self.issuer_name,
                cash_date=date(2026, 1, 15),
                description="Test merchant A",
                original_amount=Decimal("100.00"),
                original_currency="CAD",
                statement_page=1,
            )
        ]


class FakeParserB(IssuerParser):
    issuer_name = "FAKE_B"

    def detect(self, filename: str, first_page_text: str) -> bool:
        return "fake_b" in filename.lower()

    def parse(self, pdf_bytes: bytes) -> list[ParsedTransaction]:
        return [
            ParsedTransaction(
                issuer=self.issuer_name,
                cash_date=date(2026, 2, 20),
                description="Test merchant B",
                original_amount=Decimal("50.00"),
                original_currency="HKD",
                fx_rate=Decimal("0.1755"),
                fx_rate_source="statement",
                cad_amount=Decimal("8.78"),
                statement_page=2,
            )
        ]


@pytest.fixture
def registry() -> ParserRegistry:
    return ParserRegistry(parsers=[FakeParserA(), FakeParserB()])


class TestParserRegistry:
    def test_detect_first_parser(self, registry: ParserRegistry) -> None:
        parser = registry.detect_issuer("FAKE_A_2026-01.pdf", "")
        assert parser.issuer_name == "FAKE_A"

    def test_detect_second_parser(self, registry: ParserRegistry) -> None:
        parser = registry.detect_issuer("statement_fake_b.pdf", "some page text")
        assert parser.issuer_name == "FAKE_B"

    def test_detect_case_insensitive(self, registry: ParserRegistry) -> None:
        parser = registry.detect_issuer("FAKE_A_STATEMENT.PDF", "")
        assert parser.issuer_name == "FAKE_A"

    def test_detect_unknown_raises(self, registry: ParserRegistry) -> None:
        with pytest.raises(UnknownIssuerError, match="No registered parser"):
            registry.detect_issuer("unknown_bank.pdf", "")

    def test_detect_unknown_lists_registered(self, registry: ParserRegistry) -> None:
        with pytest.raises(UnknownIssuerError, match="FAKE_A.*FAKE_B"):
            registry.detect_issuer("unknown.pdf", "")

    def test_empty_registry_raises(self) -> None:
        empty = ParserRegistry(parsers=[])
        with pytest.raises(UnknownIssuerError):
            empty.detect_issuer("anything.pdf", "")

    def test_first_match_wins(self) -> None:
        """If both parsers match, the first registered one wins."""

        class GreedyParser(IssuerParser):
            issuer_name = "GREEDY"

            def detect(self, filename: str, first_page_text: str) -> bool:
                return True  # matches everything

            def parse(self, pdf_bytes: bytes) -> list[ParsedTransaction]:
                return []

        registry = ParserRegistry(parsers=[GreedyParser(), FakeParserA()])
        parser = registry.detect_issuer("fake_a.pdf", "")
        assert parser.issuer_name == "GREEDY"


class TestParsedTransaction:
    def test_cad_transaction(self) -> None:
        txn = ParsedTransaction(
            issuer="MBNA",
            cash_date=date(2026, 3, 15),
            description="STARBUCKS #1234",
            original_amount=Decimal("5.50"),
            original_currency="CAD",
            statement_page=1,
        )
        assert txn.fx_rate is None
        assert txn.fx_rate_source is None
        assert txn.cad_amount is None

    def test_fx_transaction(self) -> None:
        txn = ParsedTransaction(
            issuer="SIM_HK",
            cash_date=date(2026, 3, 20),
            description="Ocean Park",
            original_amount=Decimal("480.00"),
            original_currency="HKD",
            fx_rate=Decimal("0.1755"),
            fx_rate_source="statement",
            cad_amount=Decimal("84.24"),
            statement_page=3,
        )
        assert txn.original_currency == "HKD"
        assert txn.fx_rate == Decimal("0.1755")


class TestIssuerParserABC:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            IssuerParser()  # type: ignore[abstract]

    def test_subclass_must_implement_detect(self) -> None:
        class Incomplete(IssuerParser):
            issuer_name = "INCOMPLETE"

            def parse(self, pdf_bytes: bytes) -> list[ParsedTransaction]:
                return []

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_subclass_must_implement_parse(self) -> None:
        class Incomplete(IssuerParser):
            issuer_name = "INCOMPLETE"

            def detect(self, filename: str, first_page_text: str) -> bool:
                return False

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_complete_subclass_works(self) -> None:
        parser = FakeParserA()
        assert parser.issuer_name == "FAKE_A"
        result = parser.parse(b"fake pdf")
        assert len(result) == 1
        assert result[0].issuer == "FAKE_A"
