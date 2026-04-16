"""Integration tests for ingestion/pipeline.py — Full Ingestion Pipeline.

Tests use a real MBNA PDF fixture against an in-memory SQLite DB
with seeded categories and a trained classifier.
"""

from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from classifier.base import ClassificationResult, ClassifierClient, Transaction
from classifier.offline import OfflineClassifierClient
from db.models import (
    Base,
    Category,
    Household,
    Statement,
    Transaction as TransactionModel,
    User,
)
from ingestion.pipeline import IngestionPipeline, IngestionResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MBNA_FIXTURES = Path(__file__).parent.parent / "fixtures" / "MBNA"
WS_CC_FIXTURES = Path(__file__).parent.parent / "fixtures" / "WS-credit-card"
ROGERS_FIXTURES = Path(__file__).parent.parent / "fixtures" / "Rogers-bank"

HOUSEHOLD_ID = "test-household"

# Minimal set of categories matching what the classifier might return
_CATEGORIES = [
    ("cat-food", "Food & Beverage", "expense", "household", "A/K"),
    ("cat-living", "Living Expense", "expense", "household", "A/K"),
    ("cat-utility", "Utility", "expense", "household", "A/K"),
    ("cat-transport", "Transportation", "expense", "household", "A"),
    ("cat-dressing", "Dressing & Beauty", "expense", "household", "A/K"),
    ("cat-sport", "Sport", "expense", "household", "A/K"),
    ("cat-education", "Education", "expense", "household", "A/K"),
    ("cat-insurance", "Insurance", "expense", "household", "A/K"),
    ("cat-shared", "Shared Enjoyment", "expense", "shared_discretionary", "A/K"),
    ("cat-personal", "Personal Expense", "expense", "personal_pocket", "A"),
    ("cat-other", "Other", "expense", "household", "A/K"),
    ("cat-salary", "Salary", "income", "household", "A/K"),
    ("cat-interest", "Interest", "income", "household", "A/K"),
    ("cat-cashback", "Cashback", "income", "household", "A/K"),
    ("cat-personal-income", "Personal Income", "income", "personal_pocket", "A"),
]


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def seeded_session(engine):
    """Session with household, users, and all categories."""
    with Session(engine) as s:
        s.add(Household(id=HOUSEHOLD_ID, name="Test Household"))
        s.add(User(id="user-a", household_id=HOUSEHOLD_ID, display_name="Andrew", person_code="A"))
        s.add(User(id="user-k", household_id=HOUSEHOLD_ID, display_name="Kristy", person_code="K"))

        for cat_id, name, cat_type, tier, default_split in _CATEGORIES:
            s.add(Category(
                id=cat_id,
                household_id=HOUSEHOLD_ID,
                name=name,
                slug=name.lower().replace(" & ", "-").replace(" ", "-"),
                category_type=cat_type,
                household_tier=tier,
                tax_context="taxable",
                default_split=default_split,
                sort_order=0,
            ))

        s.commit()
        yield s


class _StubClassifier(ClassifierClient):
    """Minimal classifier that always returns 'Other' with low confidence."""

    @property
    def is_online(self) -> bool:
        return False

    def classify(self, transaction: Transaction) -> ClassificationResult:
        return ClassificationResult(
            category="cat-other",
            responsibility="A/K",
            confidence=0.5,
            source="stub",
            needs_review=True,
        )

    def classify_batch(self, transactions: list[Transaction]) -> list[ClassificationResult]:
        return [self.classify(t) for t in transactions]

    def update_from_correction(self, transaction, correct_category, correct_responsibility):
        pass

    def retrain(self) -> dict[str, object]:
        return {}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _skip_if_no_mbna():
    if not MBNA_FIXTURES.exists() or not list(MBNA_FIXTURES.glob("*.pdf")):
        pytest.skip("No MBNA fixtures")


class TestIngestionPipelineMBNA:
    @pytest.fixture(autouse=True)
    def _check(self) -> None:
        _skip_if_no_mbna()

    def test_ingest_produces_correct_db_rows(self, seeded_session: Session) -> None:
        pipeline = IngestionPipeline(
            session=seeded_session,
            household_id=HOUSEHOLD_ID,
            classifier=_StubClassifier(),
        )
        pdf_bytes = (MBNA_FIXTURES / "MBNA Canada.pdf").read_bytes()
        result = pipeline.ingest(pdf_bytes, "MBNA Canada.pdf")

        assert result.parsed >= 1
        assert result.classified == result.parsed
        assert result.errors == 0

        # Verify DB rows
        seeded_session.flush()
        db_count = seeded_session.execute(
            select(func.count(TransactionModel.id)).where(
                TransactionModel.source == "pdf_upload"
            )
        ).scalar()
        assert db_count == result.classified

    def test_ingest_creates_statement(self, seeded_session: Session) -> None:
        pipeline = IngestionPipeline(
            session=seeded_session,
            household_id=HOUSEHOLD_ID,
            classifier=_StubClassifier(),
        )
        pdf_bytes = (MBNA_FIXTURES / "MBNA Canada.pdf").read_bytes()
        result = pipeline.ingest(pdf_bytes, "MBNA Canada.pdf")

        stmt = seeded_session.execute(
            select(Statement).where(Statement.id == result.statement_id)
        ).scalar_one()
        assert stmt.issuer == "MBNA"
        assert stmt.filename == "MBNA Canada.pdf"
        assert stmt.parse_status == "ok"

    def test_duplicate_detection(self, seeded_session: Session) -> None:
        pipeline = IngestionPipeline(
            session=seeded_session,
            household_id=HOUSEHOLD_ID,
            classifier=_StubClassifier(),
        )
        pdf_bytes = (MBNA_FIXTURES / "MBNA Canada.pdf").read_bytes()

        result1 = pipeline.ingest(pdf_bytes, "MBNA Canada.pdf")
        seeded_session.flush()

        result2 = pipeline.ingest(pdf_bytes, "MBNA Canada.pdf")

        assert result1.classified > 0
        assert result2.duplicates == result1.parsed
        assert result2.classified == 0

    def test_needs_review_flag(self, seeded_session: Session) -> None:
        pipeline = IngestionPipeline(
            session=seeded_session,
            household_id=HOUSEHOLD_ID,
            classifier=_StubClassifier(),  # Always returns needs_review=True
        )
        pdf_bytes = (MBNA_FIXTURES / "MBNA Canada.pdf").read_bytes()
        result = pipeline.ingest(pdf_bytes, "MBNA Canada.pdf")

        assert result.flagged == result.classified  # All flagged with stub classifier

        seeded_session.flush()
        flagged_count = seeded_session.execute(
            select(func.count(TransactionModel.id)).where(
                TransactionModel.source == "pdf_upload",
                TransactionModel.needs_review == True,
            )
        ).scalar()
        assert flagged_count == result.flagged

    def test_transactions_have_correct_fields(self, seeded_session: Session) -> None:
        pipeline = IngestionPipeline(
            session=seeded_session,
            household_id=HOUSEHOLD_ID,
            classifier=_StubClassifier(),
        )
        pdf_bytes = (MBNA_FIXTURES / "MBNA Canada.pdf").read_bytes()
        pipeline.ingest(pdf_bytes, "MBNA Canada.pdf")
        seeded_session.flush()

        txn = seeded_session.execute(
            select(TransactionModel).where(
                TransactionModel.source == "pdf_upload"
            ).limit(1)
        ).scalar_one()

        assert txn.household_id == HOUSEHOLD_ID
        assert txn.statement_id is not None
        assert txn.cash_date is not None
        assert txn.description != ""
        assert txn.normalized_description != ""
        assert txn.original_currency == "CAD"
        assert txn.source == "pdf_upload"
        assert txn.classifier_source == "stub"


class TestIngestionPipelineErrors:
    def test_unknown_issuer(self, seeded_session: Session) -> None:
        pipeline = IngestionPipeline(
            session=seeded_session,
            household_id=HOUSEHOLD_ID,
            classifier=_StubClassifier(),
        )
        # Not a valid PDF — will fail at issuer detection
        result = pipeline.ingest(b"%PDF-1.4 fake content", "unknown_bank.pdf")

        assert result.errors >= 1
        assert result.classified == 0

    def test_classifier_only_abc_imported(self) -> None:
        """Constraint 2A-4: only ClassifierClient imported in this module."""
        import ast
        source = (Path(__file__).parent.parent.parent / "ingestion" / "pipeline.py").read_text()
        tree = ast.parse(source)

        banned = {"TfidfVectorizer", "LogisticRegression", "pickle", "SentenceTransformer"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    assert alias.name not in banned, f"Banned import: {alias.name}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in banned, f"Banned import: {alias.name}"


class TestIngestionPipelineMultiIssuer:
    """Test that the pipeline handles different issuers correctly."""

    def _skip_if_no(self, path: Path):
        if not path.exists() or not list(path.glob("*.pdf")):
            pytest.skip(f"No fixtures in {path}")

    def test_rogers_ingestion(self, seeded_session: Session) -> None:
        self._skip_if_no(ROGERS_FIXTURES)
        pipeline = IngestionPipeline(
            session=seeded_session,
            household_id=HOUSEHOLD_ID,
            classifier=_StubClassifier(),
        )
        pdf_bytes = next(ROGERS_FIXTURES.glob("*.pdf")).read_bytes()
        result = pipeline.ingest(pdf_bytes, "rogers_statement.pdf")

        assert result.issuer == "ROGERS"
        assert result.parsed >= 1
        assert result.errors == 0

    def test_wealthsimple_cc_ingestion(self, seeded_session: Session) -> None:
        self._skip_if_no(WS_CC_FIXTURES)
        pipeline = IngestionPipeline(
            session=seeded_session,
            household_id=HOUSEHOLD_ID,
            classifier=_StubClassifier(),
        )
        pdf_bytes = next(WS_CC_FIXTURES.glob("*.pdf")).read_bytes()
        result = pipeline.ingest(pdf_bytes, "ws_credit_card.pdf")

        assert result.issuer == "WEALTHSIMPLE"
        assert result.parsed >= 1
        assert result.errors == 0
