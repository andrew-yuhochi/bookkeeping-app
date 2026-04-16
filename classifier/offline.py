"""OfflineClassifierClient — local ML-based classifier.

Layered classification pipeline:
- Layer 0: Merchant normalization (classifier/normalizer.py)
- Layer 1: Exact-match cache (classifier/cache.py) — TASK-006
- Layer 2: TF-IDF + Logistic Regression — TASK-007 (stub for now)
"""

import logging
from typing import Optional

from sqlalchemy.orm import Session

from classifier.base import (
    ClassificationResult,
    ClassifierClient,
    SplitMethod,
    Transaction,
)
from classifier.cache import CacheHit, ExactMatchCache
from classifier.normalizer import normalize_merchant

logger = logging.getLogger(__name__)


class OfflineClassifierClient(ClassifierClient):
    """Offline classifier using exact-match cache + TF-IDF + LR.

    This implementation never makes network calls (is_online = False).

    Args:
        session: SQLAlchemy session for cache DB access. If None, cache is disabled.
        household_id: Household UUID for scoped cache lookups. Required if session is provided.
    """

    def __init__(
        self,
        session: Optional[Session] = None,
        household_id: Optional[str] = None,
    ) -> None:
        self._session = session
        self._household_id = household_id
        self._cache: Optional[ExactMatchCache] = None

        if session is not None and household_id is not None:
            self._cache = ExactMatchCache(session, household_id)

    @property
    def is_online(self) -> bool:
        return False

    def classify(self, transaction: Transaction) -> ClassificationResult:
        """Classify a single transaction through the layered pipeline.

        Layer 1 (exact-match cache): if the normalized description matches
        a cached entry, return confidence=1.0, source="cache".

        Layer 2 (TF-IDF + LR): stub — falls through to needs_review=True.
        """
        # Layer 1: exact-match cache lookup
        if self._cache is not None:
            hit = self._cache.lookup(transaction.normalized_description)
            if hit is not None:
                return ClassificationResult(
                    category=hit.category_id,
                    responsibility=hit.responsibility,
                    confidence=1.0,
                    source="cache",
                    needs_review=False,
                )

        # Layer 2: TF-IDF + LR (stub until TASK-007)
        return ClassificationResult(
            category="Other",
            responsibility="A/K",
            confidence=0.0,
            source="stub",
            needs_review=True,
        )

    def classify_batch(
        self, transactions: list[Transaction]
    ) -> list[ClassificationResult]:
        """Classify each transaction individually."""
        return [self.classify(txn) for txn in transactions]

    def update_from_correction(
        self,
        transaction: Transaction,
        correct_category: str,
        correct_responsibility: SplitMethod,
    ) -> None:
        """Record a user correction. Real cache-write gating in TASK-010."""
        logger.debug(
            "Correction recorded: %s → category=%s, responsibility=%s",
            transaction.normalized_description,
            correct_category,
            correct_responsibility,
        )

    def retrain(self) -> dict[str, object]:
        """Stub: no-op. Real implementation in TASK-007."""
        logger.info("retrain() called (stub) — no model to train yet")
        return {}
