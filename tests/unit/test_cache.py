"""Tests for classifier/cache.py — ExactMatchCache."""

import pytest
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from classifier.cache import CacheHit, ExactMatchCache
from db.models import Base, Category, ExactMatchCache as ExactMatchCacheRow, Household


@pytest.fixture
def engine():
    """Create an in-memory SQLite engine with all tables."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def session(engine):
    """Create a session with a seeded household and categories."""
    with Session(engine) as s:
        household = Household(id="test-household", name="Test Household")
        s.add(household)

        cat_food = Category(
            id="cat-food",
            household_id="test-household",
            name="Food & Beverage",
            slug="food-beverage",
            category_type="expense",
            household_tier="household",
            tax_context="taxable",
            default_split="A/K",
            sort_order=1,
        )
        cat_transport = Category(
            id="cat-transport",
            household_id="test-household",
            name="Transportation",
            slug="transportation",
            category_type="expense",
            household_tier="household",
            tax_context="taxable",
            default_split="A/K",
            sort_order=2,
        )
        s.add_all([cat_food, cat_transport])
        s.commit()

        yield s


class TestExactMatchCache:
    def test_empty_cache(self, session: Session) -> None:
        cache = ExactMatchCache(session, "test-household")
        assert cache.size == 0
        assert cache.lookup("starbucks") is None

    def test_put_and_lookup(self, session: Session) -> None:
        cache = ExactMatchCache(session, "test-household")
        cache.put("starbucks", "cat-food", "A/K")
        session.flush()

        hit = cache.lookup("starbucks")
        assert hit is not None
        assert hit.category_id == "cat-food"
        assert hit.responsibility == "A/K"

    def test_lookup_empty_key_returns_none(self, session: Session) -> None:
        cache = ExactMatchCache(session, "test-household")
        cache.put("starbucks", "cat-food", "A/K")
        assert cache.lookup("") is None

    def test_put_updates_existing(self, session: Session) -> None:
        cache = ExactMatchCache(session, "test-household")
        cache.put("starbucks", "cat-food", "A/K", confirmation_count=1)
        session.flush()

        # Update to different category
        cache.put("starbucks", "cat-transport", "A", confirmation_count=3)
        session.flush()

        hit = cache.lookup("starbucks")
        assert hit is not None
        assert hit.category_id == "cat-transport"
        assert hit.responsibility == "A"

        # Only one row in DB
        rows = session.query(ExactMatchCacheRow).filter_by(
            normalized_merchant="starbucks"
        ).all()
        assert len(rows) == 1
        assert rows[0].confirmation_count == 3

    def test_size_property(self, session: Session) -> None:
        cache = ExactMatchCache(session, "test-household")
        assert cache.size == 0

        cache.put("starbucks", "cat-food", "A/K")
        assert cache.size == 1

        cache.put("uber", "cat-transport", "A")
        assert cache.size == 2

    def test_loads_from_db_on_init(self, session: Session) -> None:
        """Pre-populate DB, then create a new cache instance — it should load."""
        row = ExactMatchCacheRow(
            household_id="test-household",
            normalized_merchant="costco",
            category_id="cat-food",
            responsibility="A/K",
            confirmation_count=5,
            last_confirmed_at=datetime.utcnow(),
        )
        session.add(row)
        session.flush()

        # New cache instance should load from DB
        cache = ExactMatchCache(session, "test-household")
        assert cache.size == 1
        hit = cache.lookup("costco")
        assert hit is not None
        assert hit.category_id == "cat-food"

    def test_household_isolation(self, session: Session) -> None:
        """Cache entries from one household should not leak to another."""
        # Add entry for test-household
        row = ExactMatchCacheRow(
            household_id="test-household",
            normalized_merchant="walmart",
            category_id="cat-food",
            responsibility="A/K",
            confirmation_count=1,
            last_confirmed_at=datetime.utcnow(),
        )
        session.add(row)
        session.flush()

        # Create cache for a different household
        cache = ExactMatchCache(session, "other-household")
        assert cache.size == 0
        assert cache.lookup("walmart") is None

    def test_put_empty_merchant_is_noop(self, session: Session) -> None:
        cache = ExactMatchCache(session, "test-household")
        cache.put("", "cat-food", "A/K")
        assert cache.size == 0


class TestCacheWithClassifier:
    """Test cache integration with OfflineClassifierClient."""

    def test_cache_hit_returns_confidence_1(self, session: Session) -> None:
        from classifier.base import Transaction
        from classifier.offline import OfflineClassifierClient

        # Pre-populate cache
        cache_row = ExactMatchCacheRow(
            household_id="test-household",
            normalized_merchant="starbucks",
            category_id="cat-food",
            responsibility="A/K",
            confirmation_count=5,
            last_confirmed_at=datetime.utcnow(),
        )
        session.add(cache_row)
        session.flush()

        client = OfflineClassifierClient(session=session, household_id="test-household")

        txn = Transaction(
            description="STARBUCKS #1234",
            normalized_description="starbucks",
            amount_cad=5.50,
            original_currency="CAD",
            issuer="MBNA",
        )
        result = client.classify(txn)
        assert result.confidence == 1.0
        assert result.source == "cache"
        assert result.needs_review is False
        assert result.category == "cat-food"
        assert result.responsibility == "A/K"

    def test_cache_miss_falls_through(self, session: Session) -> None:
        from classifier.base import Transaction
        from classifier.offline import OfflineClassifierClient

        client = OfflineClassifierClient(session=session, household_id="test-household")

        txn = Transaction(
            description="UNKNOWN MERCHANT",
            normalized_description="unknown merchant",
            amount_cad=99.00,
            original_currency="CAD",
            issuer="MBNA",
        )
        result = client.classify(txn)
        assert result.confidence == 0.0
        assert result.source == "stub"
        assert result.needs_review is True

    def test_no_session_disables_cache(self) -> None:
        from classifier.base import Transaction
        from classifier.offline import OfflineClassifierClient

        client = OfflineClassifierClient()  # No session

        txn = Transaction(
            description="STARBUCKS",
            normalized_description="starbucks",
            amount_cad=5.50,
            original_currency="CAD",
            issuer="MBNA",
        )
        result = client.classify(txn)
        assert result.source == "stub"
        assert result.needs_review is True
