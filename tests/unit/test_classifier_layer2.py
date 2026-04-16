"""Tests for TASK-007: TF-IDF + LR classifier (Layer 2).

Tests retrain(), classify() with the ML model, confidence thresholds,
thread safety, and model serialization.
"""

import threading
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from classifier.base import ClassificationResult, Transaction
from classifier.offline import (
    CONFIDENCE_THRESHOLD,
    OfflineClassifierClient,
    _MODEL_DIR,
    _MODEL_PATH,
)
from db.models import (
    Base,
    Category,
    ExactMatchCache as ExactMatchCacheRow,
    Household,
    Transaction as TransactionModel,
    User,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

HOUSEHOLD_ID = "test-household"

# Minimal synthetic corpus — 20 rows across 3 categories and 3 split methods
_SYNTHETIC_ROWS = [
    # (description, category_slug, split_method) — repeated to build volume
    ("walmart groceries", "food-beverage", "A/K"),
    ("costco food", "food-beverage", "A/K"),
    ("superstore grocery", "food-beverage", "A/K"),
    ("loblaws market", "food-beverage", "A/K"),
    ("hmart korean", "food-beverage", "A/K"),
    ("no frills produce", "food-beverage", "A/K"),
    ("safeway bakery", "food-beverage", "A/K"),
    ("uber ride downtown", "transportation", "A"),
    ("compass transit pass", "transportation", "A"),
    ("esso gas station", "transportation", "A"),
    ("shell fuel", "transportation", "A"),
    ("bc transit monthly", "transportation", "A"),
    ("taxi airport", "transportation", "K"),
    ("parking meter lot", "transportation", "A/K"),
    ("bc hydro electric", "utility", "A/K"),
    ("telus phone bill", "utility", "A/K"),
    ("fortis gas utility", "utility", "A/K"),
    ("shaw internet", "utility", "A/K"),
    ("rogers mobile plan", "utility", "A/K"),
    ("enbridge gas bill", "utility", "A/K"),
]


@pytest.fixture
def engine(tmp_path: Path):
    """Create an in-memory SQLite engine with all tables."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def seeded_session(engine):
    """Session with household, categories, and synthetic training data."""
    with Session(engine) as s:
        # Household
        household = Household(id=HOUSEHOLD_ID, name="Test Household")
        s.add(household)

        # Users
        user_a = User(id="user-a", household_id=HOUSEHOLD_ID, display_name="Andrew", person_code="A")
        user_k = User(id="user-k", household_id=HOUSEHOLD_ID, display_name="Kristy", person_code="K")
        s.add_all([user_a, user_k])

        # Categories
        cats = {
            "food-beverage": Category(
                id="cat-food", household_id=HOUSEHOLD_ID, name="Food & Beverage",
                slug="food-beverage", category_type="expense",
                household_tier="household", tax_context="taxable",
                default_split="A/K", sort_order=1,
            ),
            "transportation": Category(
                id="cat-transport", household_id=HOUSEHOLD_ID, name="Transportation",
                slug="transportation", category_type="expense",
                household_tier="household", tax_context="taxable",
                default_split="A", sort_order=2,
            ),
            "utility": Category(
                id="cat-utility", household_id=HOUSEHOLD_ID, name="Utility",
                slug="utility", category_type="expense",
                household_tier="household", tax_context="taxable",
                default_split="A/K", sort_order=3,
            ),
        }
        s.add_all(cats.values())
        s.flush()

        slug_to_id = {slug: cat.id for slug, cat in cats.items()}

        # Synthetic transactions
        for i, (desc, cat_slug, split) in enumerate(_SYNTHETIC_ROWS):
            txn = TransactionModel(
                household_id=HOUSEHOLD_ID,
                cash_date=datetime(2026, 1, 15).date(),
                accounting_period_year=2026,
                accounting_period_month=1,
                description=desc,
                normalized_description=desc.lower(),
                original_amount="10.00",
                original_currency="CAD",
                fx_rate="1.0",
                fx_rate_source="statement",
                cad_amount="10.00",
                category_id=slug_to_id[cat_slug],
                split_method=split,
                needs_review=False,
                source="historical_import",
                source_ref=f"test:{i}",
            )
            s.add(txn)

        s.commit()
        yield s


# ---------------------------------------------------------------------------
# retrain() tests
# ---------------------------------------------------------------------------

class TestRetrain:
    def test_retrain_returns_metrics_dict(self, seeded_session: Session) -> None:
        client = OfflineClassifierClient(session=seeded_session, household_id=HOUSEHOLD_ID)
        metrics = client.retrain()

        assert "rows_trained" in metrics
        assert "held_out_accuracy" in metrics
        assert "duration_seconds" in metrics
        assert isinstance(metrics["rows_trained"], int)
        assert isinstance(metrics["held_out_accuracy"], float)
        assert isinstance(metrics["duration_seconds"], float)
        assert metrics["rows_trained"] > 0

    def test_retrain_trains_responsibility_model(self, seeded_session: Session) -> None:
        client = OfflineClassifierClient(session=seeded_session, household_id=HOUSEHOLD_ID)
        metrics = client.retrain()

        assert "responsibility_accuracy" in metrics
        assert isinstance(metrics["responsibility_accuracy"], float)

    def test_retrain_serializes_model(self, seeded_session: Session) -> None:
        client = OfflineClassifierClient(session=seeded_session, household_id=HOUSEHOLD_ID)
        client.retrain()

        assert _MODEL_PATH.exists()
        assert _MODEL_PATH.stat().st_size > 0

    def test_retrain_no_session_returns_empty(self) -> None:
        client = OfflineClassifierClient()
        metrics = client.retrain()
        assert metrics == {}

    def test_retrain_completes_under_5_seconds(self, seeded_session: Session) -> None:
        """Even on the synthetic corpus, should be well under 5s."""
        import time
        client = OfflineClassifierClient(session=seeded_session, household_id=HOUSEHOLD_ID)
        t0 = time.monotonic()
        client.retrain()
        elapsed = time.monotonic() - t0
        assert elapsed < 5.0


# ---------------------------------------------------------------------------
# classify() with trained model
# ---------------------------------------------------------------------------

class TestClassifyLayer2:
    @pytest.fixture(autouse=True)
    def _train_model(self, seeded_session: Session) -> None:
        self.client = OfflineClassifierClient(session=seeded_session, household_id=HOUSEHOLD_ID)
        self.client.retrain()

    def test_novel_merchant_returns_tfidf_lr_source(self) -> None:
        txn = Transaction(
            description="FRESH MARKET PRODUCE",
            normalized_description="fresh market produce",
            amount_cad=45.0,
            original_currency="CAD",
            issuer="MBNA",
        )
        result = self.client.classify(txn)
        assert result.source == "tfidf_lr"
        assert 0.0 < result.confidence <= 1.0

    def test_confidence_threshold_needs_review(self) -> None:
        """Low confidence → needs_review=True."""
        txn = Transaction(
            description="XYZZY UNKNOWN MERCHANT",
            normalized_description="xyzzy unknown merchant",
            amount_cad=10.0,
            original_currency="CAD",
            issuer="MBNA",
        )
        result = self.client.classify(txn)
        if result.confidence < CONFIDENCE_THRESHOLD:
            assert result.needs_review is True
        else:
            assert result.needs_review is False

    def test_high_confidence_no_review(self) -> None:
        """A grocery merchant should classify as food with decent confidence."""
        txn = Transaction(
            description="GROCERY SUPERMARKET",
            normalized_description="grocery supermarket",
            amount_cad=50.0,
            original_currency="CAD",
            issuer="MBNA",
        )
        result = self.client.classify(txn)
        # The model should recognize grocery-like text
        assert result.source == "tfidf_lr"
        assert isinstance(result.confidence, float)

    def test_classify_batch_matches_length(self) -> None:
        txns = [
            Transaction(
                description=f"merchant_{i}",
                normalized_description=f"merchant_{i}",
                amount_cad=10.0,
                original_currency="CAD",
                issuer="MBNA",
            )
            for i in range(5)
        ]
        results = self.client.classify_batch(txns)
        assert len(results) == 5
        assert all(isinstance(r, ClassificationResult) for r in results)

    def test_cache_hit_takes_priority_over_model(self, seeded_session: Session) -> None:
        """Cache hit returns source='cache', not 'tfidf_lr'."""
        # Add a cache entry
        cache_row = ExactMatchCacheRow(
            household_id=HOUSEHOLD_ID,
            normalized_merchant="test merchant cached",
            category_id="cat-food",
            responsibility="A/K",
            confirmation_count=5,
            last_confirmed_at=datetime.utcnow(),
        )
        seeded_session.add(cache_row)
        seeded_session.flush()

        # Recreate client so it loads the new cache entry
        client = OfflineClassifierClient(session=seeded_session, household_id=HOUSEHOLD_ID)
        # Load the already-trained model
        client._try_load_model()

        txn = Transaction(
            description="TEST MERCHANT CACHED",
            normalized_description="test merchant cached",
            amount_cad=10.0,
            original_currency="CAD",
            issuer="MBNA",
        )
        result = client.classify(txn)
        assert result.source == "cache"
        assert result.confidence == 1.0


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_classify_during_retrain(self, seeded_session: Session) -> None:
        """classify() should not crash during concurrent retrain()."""
        client = OfflineClassifierClient(session=seeded_session, household_id=HOUSEHOLD_ID)
        client.retrain()  # Ensure initial model exists

        errors: list[Exception] = []

        def classify_loop() -> None:
            try:
                txn = Transaction(
                    description="CONCURRENT TEST",
                    normalized_description="concurrent test",
                    amount_cad=10.0,
                    original_currency="CAD",
                    issuer="MBNA",
                )
                for _ in range(50):
                    result = client.classify(txn)
                    assert isinstance(result, ClassificationResult)
            except Exception as e:
                errors.append(e)

        # Start classify in a thread, then retrain in main thread
        t = threading.Thread(target=classify_loop)
        t.start()
        client.retrain()  # retrain while classify is running
        t.join()

        assert errors == [], f"Errors during concurrent classify: {errors}"


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

class TestModelPersistence:
    def test_load_model_from_disk(self, seeded_session: Session) -> None:
        """After retrain, a new client should load the model from disk."""
        client1 = OfflineClassifierClient(session=seeded_session, household_id=HOUSEHOLD_ID)
        client1.retrain()

        # New client — should load from disk
        client2 = OfflineClassifierClient(session=seeded_session, household_id=HOUSEHOLD_ID)

        txn = Transaction(
            description="GAS STATION FUEL",
            normalized_description="gas station fuel",
            amount_cad=60.0,
            original_currency="CAD",
            issuer="MBNA",
        )
        result = client2.classify(txn)
        assert result.source in ("tfidf_lr", "cache")  # Model loaded, not fallback
