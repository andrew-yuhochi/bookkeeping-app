"""OfflineClassifierClient — local ML-based classifier.

Skeleton implementation for TASK-005. All methods return stub values:
- classify() returns confidence=0.0, needs_review=True, source="stub"
- retrain() is a no-op returning an empty dict

Real classification layers are wired in subsequent tasks:
- TASK-006: Layer 1 (exact-match cache)
- TASK-007: Layer 2 (TF-IDF + Logistic Regression)
"""

import logging

from classifier.base import (
    ClassificationResult,
    ClassifierClient,
    SplitMethod,
    Transaction,
)

logger = logging.getLogger(__name__)


class OfflineClassifierClient(ClassifierClient):
    """Offline classifier using exact-match cache + TF-IDF + LR.

    This implementation never makes network calls (is_online = False).
    """

    @property
    def is_online(self) -> bool:
        return False

    def classify(self, transaction: Transaction) -> ClassificationResult:
        """Stub: returns needs_review=True for all transactions."""
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
        """Stub: no-op. Real implementation in TASK-006+."""
        logger.debug(
            "Correction recorded (stub): %s → category=%s, responsibility=%s",
            transaction.normalized_description,
            correct_category,
            correct_responsibility,
        )

    def retrain(self) -> dict[str, object]:
        """Stub: no-op. Real implementation in TASK-007."""
        logger.info("retrain() called (stub) — no model to train yet")
        return {}
