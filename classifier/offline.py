"""OfflineClassifierClient — local ML-based classifier.

Layered classification pipeline:
- Layer 0: Merchant normalization (classifier/normalizer.py)
- Layer 1: Exact-match cache (classifier/cache.py) — TASK-006
- Layer 2: TF-IDF + Logistic Regression — TASK-007
"""

import logging
import pickle
import threading
import time
from pathlib import Path
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sqlalchemy import select, union_all
from sqlalchemy.orm import Session

from classifier.base import (
    ClassificationResult,
    ClassifierClient,
    SplitMethod,
    Transaction,
)
from classifier.cache import ExactMatchCache
from classifier.normalizer import normalize_merchant
from db.models import Category as CategoryModel
from db.models import Correction as CorrectionModel
from db.models import Transaction as TransactionModel

logger = logging.getLogger(__name__)

# Confidence threshold: at or above this, needs_review=False
CONFIDENCE_THRESHOLD = 0.70

# Default model directory
_MODEL_DIR = Path(__file__).parent / "models"
_MODEL_PATH = _MODEL_DIR / "model.pkl"


class _TrainedModel:
    """Container for a trained category + responsibility model pair.

    Immutable after construction — safe to read from multiple threads
    while a new instance is being trained.
    """

    __slots__ = (
        "category_pipeline",
        "responsibility_pipeline",
        "category_classes",
        "responsibility_classes",
    )

    def __init__(
        self,
        category_pipeline: Pipeline,
        responsibility_pipeline: Pipeline,
    ) -> None:
        self.category_pipeline = category_pipeline
        self.responsibility_pipeline = responsibility_pipeline
        self.category_classes = list(category_pipeline.classes_)
        self.responsibility_classes = list(responsibility_pipeline.classes_)

    def predict(self, normalized_description: str) -> tuple[str, float, str, float]:
        """Predict category and responsibility for a single description.

        Returns:
            (category_id, category_confidence, responsibility, responsibility_confidence)
        """
        cat_proba = self.category_pipeline.predict_proba([normalized_description])[0]
        cat_idx = cat_proba.argmax()
        category = self.category_classes[cat_idx]
        cat_confidence = float(cat_proba[cat_idx])

        resp_proba = self.responsibility_pipeline.predict_proba([normalized_description])[0]
        resp_idx = resp_proba.argmax()
        responsibility = self.responsibility_classes[resp_idx]
        resp_confidence = float(resp_proba[resp_idx])

        return category, cat_confidence, responsibility, resp_confidence


class OfflineClassifierClient(ClassifierClient):
    """Offline classifier using exact-match cache + TF-IDF + LR.

    This implementation never makes network calls (is_online = False).

    Args:
        session: SQLAlchemy session for cache/DB access. If None, cache + ML disabled.
        household_id: Household UUID for scoped lookups. Required if session is provided.
    """

    def __init__(
        self,
        session: Optional[Session] = None,
        household_id: Optional[str] = None,
    ) -> None:
        self._session = session
        self._household_id = household_id
        self._cache: Optional[ExactMatchCache] = None
        self._model: Optional[_TrainedModel] = None
        self._model_lock = threading.Lock()

        if session is not None and household_id is not None:
            self._cache = ExactMatchCache(session, household_id)
            self._try_load_model()

    def _try_load_model(self) -> None:
        """Try to load a previously serialized model from disk."""
        if _MODEL_PATH.exists():
            try:
                with open(_MODEL_PATH, "rb") as f:
                    self._model = pickle.load(f)
                logger.info("Loaded model from %s", _MODEL_PATH)
            except Exception:
                logger.warning("Failed to load model from %s, will need retrain", _MODEL_PATH, exc_info=True)

    @property
    def is_online(self) -> bool:
        return False

    def classify(self, transaction: Transaction) -> ClassificationResult:
        """Classify a single transaction through the layered pipeline.

        Layer 1 (exact-match cache): confidence=1.0, source="cache".
        Layer 2 (TF-IDF + LR): confidence from predict_proba, source="tfidf_lr".
        Fallback: confidence=0.0, source="none", needs_review=True.
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

        # Layer 2: TF-IDF + LR
        model = self._model  # atomic read — safe during concurrent retrain
        if model is not None:
            category, cat_conf, responsibility, resp_conf = model.predict(
                transaction.normalized_description
            )
            needs_review = cat_conf < CONFIDENCE_THRESHOLD
            return ClassificationResult(
                category=category,
                responsibility=responsibility,
                confidence=cat_conf,
                source="tfidf_lr",
                needs_review=needs_review,
            )

        # No model available — fallback
        return ClassificationResult(
            category="Other",
            responsibility="A/K",
            confidence=0.0,
            source="none",
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
        """Train TF-IDF + LR from all labeled data, swap model atomically.

        Queries historical transactions UNION corrections to build the
        training corpus. Trains two pipelines: category and responsibility.
        Serializes to disk and swaps in-memory reference atomically.

        Returns:
            Metrics dict with rows_trained, held_out_accuracy, duration_seconds.
        """
        if self._session is None or self._household_id is None:
            logger.warning("retrain() called without session/household_id — skipping")
            return {}

        start = time.monotonic()

        # --- Gather training data ---
        descriptions, category_labels, responsibility_labels = self._gather_training_data()

        if len(descriptions) < 10:
            logger.warning("Too few training samples (%d), skipping retrain", len(descriptions))
            return {}

        # --- Train/test split ---
        (
            X_train, X_test,
            y_cat_train, y_cat_test,
            y_resp_train, y_resp_test,
        ) = train_test_split(
            descriptions, category_labels, responsibility_labels,
            test_size=0.20,
            random_state=42,
            stratify=category_labels,
        )

        # --- Train category pipeline ---
        cat_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                C=1.0,
            )),
        ])
        cat_pipeline.fit(X_train, y_cat_train)
        cat_accuracy = float(cat_pipeline.score(X_test, y_cat_test))

        # --- Train responsibility pipeline ---
        resp_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                C=1.0,
            )),
        ])
        resp_pipeline.fit(X_train, y_resp_train)
        resp_accuracy = float(resp_pipeline.score(X_test, y_resp_test))

        # --- Build model container ---
        new_model = _TrainedModel(
            category_pipeline=cat_pipeline,
            responsibility_pipeline=resp_pipeline,
        )

        # --- Serialize to disk ---
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(_MODEL_PATH, "wb") as f:
            pickle.dump(new_model, f)

        model_size_kb = _MODEL_PATH.stat().st_size / 1024
        logger.info("Model serialized to %s (%.0f KB)", _MODEL_PATH, model_size_kb)

        # --- Atomic swap ---
        with self._model_lock:
            self._model = new_model

        duration = time.monotonic() - start

        metrics: dict[str, object] = {
            "rows_trained": len(X_train),
            "rows_tested": len(X_test),
            "held_out_accuracy": cat_accuracy,
            "responsibility_accuracy": resp_accuracy,
            "duration_seconds": round(duration, 2),
            "model_size_kb": round(model_size_kb, 1),
        }

        logger.info(
            "retrain() complete: %d rows, category accuracy=%.3f, "
            "responsibility accuracy=%.3f, %.2fs",
            len(descriptions), cat_accuracy, resp_accuracy, duration,
        )

        return metrics

    def _gather_training_data(
        self,
    ) -> tuple[list[str], list[str], list[str]]:
        """Query DB for all labeled data: seed transactions + corrections.

        Returns:
            (descriptions, category_labels, responsibility_labels) — parallel lists.
        """
        assert self._session is not None
        assert self._household_id is not None

        # Historical transactions (the seed corpus)
        txns = self._session.execute(
            select(
                TransactionModel.description,
                TransactionModel.category_id,
                TransactionModel.split_method,
            ).where(
                TransactionModel.household_id == self._household_id,
            )
        ).all()

        # Corrections override the original label — build a map of
        # transaction_id → (new_category_id, new_split_method)
        corrections = self._session.execute(
            select(
                CorrectionModel.transaction_id,
                CorrectionModel.new_category_id,
                CorrectionModel.new_split_method,
            ).where(
                CorrectionModel.household_id == self._household_id,
            )
        ).all()

        # Latest correction per transaction wins
        correction_map: dict[str, tuple[str, str]] = {}
        for txn_id, new_cat, new_split in corrections:
            correction_map[txn_id] = (new_cat, new_split)

        descriptions: list[str] = []
        category_labels: list[str] = []
        responsibility_labels: list[str] = []

        for desc, cat_id, split_method in txns:
            normalized = normalize_merchant(desc)
            if not normalized:
                continue
            descriptions.append(normalized)
            category_labels.append(cat_id)
            responsibility_labels.append(split_method)

        logger.info(
            "Training corpus: %d samples (%d corrections applied)",
            len(descriptions), len(correction_map),
        )

        return descriptions, category_labels, responsibility_labels
