"""Exact-match cache — Layer 1 of the classification pipeline.

Backed by the `exact_match_cache` DB table. Provides O(1) lookup for
previously-seen normalized merchant strings.

TDD Section 2.2 Layer 1.
"""

import logging
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import ExactMatchCache as ExactMatchCacheRow

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CacheHit:
    """Result of a successful cache lookup."""

    category_id: str
    responsibility: str


class ExactMatchCache:
    """In-memory + DB-backed exact-match cache for merchant classification.

    On initialization, loads the full cache from DB into memory for fast
    lookups. The in-memory dict is the primary lookup; DB is the durable store.
    """

    def __init__(self, session: Session, household_id: str) -> None:
        self._session = session
        self._household_id = household_id
        self._cache: dict[str, CacheHit] = {}
        self._load_from_db()

    def _load_from_db(self) -> None:
        """Load all cache entries for this household from the DB."""
        rows = self._session.execute(
            select(ExactMatchCacheRow).where(
                ExactMatchCacheRow.household_id == self._household_id
            )
        ).scalars().all()

        for row in rows:
            self._cache[row.normalized_merchant] = CacheHit(
                category_id=row.category_id,
                responsibility=row.responsibility,
            )

        logger.info(
            "Loaded %d exact-match cache entries for household %s",
            len(self._cache),
            self._household_id,
        )

    def lookup(self, normalized_merchant: str) -> CacheHit | None:
        """Look up a normalized merchant string in the cache.

        Args:
            normalized_merchant: Output of normalize_merchant().

        Returns:
            CacheHit with category_id and responsibility on hit, None on miss.
        """
        if not normalized_merchant:
            return None
        return self._cache.get(normalized_merchant)

    def put(
        self,
        normalized_merchant: str,
        category_id: str,
        responsibility: str,
        confirmation_count: int = 1,
    ) -> None:
        """Write an entry to the cache (both in-memory and DB).

        If the merchant already exists in the DB, updates the existing row.
        Does NOT flush/commit — caller is responsible for session management.

        Args:
            normalized_merchant: Normalized merchant key.
            category_id: Category UUID.
            responsibility: Split method ("A", "K", or "A/K").
            confirmation_count: Number of times this mapping has been confirmed.
        """
        if not normalized_merchant:
            return

        # Update in-memory
        self._cache[normalized_merchant] = CacheHit(
            category_id=category_id,
            responsibility=responsibility,
        )

        # Upsert in DB
        existing = self._session.execute(
            select(ExactMatchCacheRow).where(
                ExactMatchCacheRow.household_id == self._household_id,
                ExactMatchCacheRow.normalized_merchant == normalized_merchant,
            )
        ).scalar_one_or_none()

        now = datetime.utcnow()

        if existing:
            existing.category_id = category_id
            existing.responsibility = responsibility
            existing.confirmation_count = confirmation_count
            existing.last_confirmed_at = now
        else:
            row = ExactMatchCacheRow(
                household_id=self._household_id,
                normalized_merchant=normalized_merchant,
                category_id=category_id,
                responsibility=responsibility,
                confirmation_count=confirmation_count,
                last_confirmed_at=now,
            )
            self._session.add(row)

    @property
    def size(self) -> int:
        """Number of entries in the in-memory cache."""
        return len(self._cache)
