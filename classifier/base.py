"""ClassifierClient abstraction — the ONLY symbol external modules may import.

Constraint 2A-4: no module outside classifier/ may import TfidfVectorizer,
LogisticRegression, pickle, or SentenceTransformer. External callers import
only from this module:

    from classifier.base import ClassifierClient, Transaction, ClassificationResult

Contract from RESEARCH-REPORT-v2 Section 1.3.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

SplitMethod = Literal["A", "K", "A/K"]


@dataclass(frozen=True)
class Transaction:
    """Normalized transaction as produced by the PDF parsing pipeline."""

    description: str
    normalized_description: str
    amount_cad: float
    original_currency: str
    issuer: str


@dataclass(frozen=True)
class ClassificationResult:
    """Result of classifying a single transaction."""

    category: str
    responsibility: SplitMethod
    confidence: float
    source: str
    needs_review: bool


class ClassifierClient(ABC):
    """Abstract interface for transaction classification.

    All components that classify transactions must call ONLY this interface.
    No classifier internals may be imported outside the classifier package.

    Implementations:
      - OfflineClassifierClient  (PoC: exact-match cache + TF-IDF + LR)
      - SentenceTransformerClient (optional Layer 3 upgrade)
      - ClaudeAPIClient           (future: LLM-based classification)
    """

    @abstractmethod
    def classify(self, transaction: Transaction) -> ClassificationResult:
        """Classify a single transaction.

        Must be synchronous and side-effect-free. Safe to call from
        multiple threads concurrently.
        """

    @abstractmethod
    def classify_batch(
        self, transactions: list[Transaction]
    ) -> list[ClassificationResult]:
        """Classify a list of transactions.

        Implementations may vectorize for efficiency. Order of results
        matches order of input.
        """

    @abstractmethod
    def update_from_correction(
        self,
        transaction: Transaction,
        correct_category: str,
        correct_responsibility: SplitMethod,
    ) -> None:
        """Record a user correction.

        The client decides internally how to store the correction. The
        caller also writes to the DB corrections table separately.
        Does NOT trigger synchronous retraining.
        """

    @abstractmethod
    def retrain(self) -> dict[str, object]:
        """Trigger a full model retrain from all available labeled data.

        Blocks until complete. Returns a metrics dict:
          {"rows_trained": int, "held_out_accuracy": float, "duration_seconds": float}

        For implementations that do not support retraining, returns an empty dict.
        """

    @property
    @abstractmethod
    def is_online(self) -> bool:
        """True if this implementation requires network access.

        OfflineClassifierClient must return False.
        """
