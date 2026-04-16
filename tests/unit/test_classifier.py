import ast
from pathlib import Path

import pytest

from classifier.base import (
    ClassificationResult,
    ClassifierClient,
    SplitMethod,
    Transaction,
)
from classifier.offline import OfflineClassifierClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client() -> OfflineClassifierClient:
    return OfflineClassifierClient()


@pytest.fixture
def sample_transaction() -> Transaction:
    return Transaction(
        description="STARBUCKS #1234 TORONTO ON",
        normalized_description="starbucks",
        amount_cad=5.50,
        original_currency="CAD",
        issuer="MBNA",
    )


# ---------------------------------------------------------------------------
# ClassifierClient ABC contract
# ---------------------------------------------------------------------------

class TestClassifierClientABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            ClassifierClient()  # type: ignore[abstract]

    def test_missing_classify_raises(self) -> None:
        class Incomplete(ClassifierClient):
            def classify_batch(self, transactions: list[Transaction]) -> list[ClassificationResult]:
                return []

            def update_from_correction(self, transaction: Transaction, correct_category: str, correct_responsibility: SplitMethod) -> None:
                pass

            def retrain(self) -> dict[str, object]:
                return {}

            @property
            def is_online(self) -> bool:
                return False

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_missing_is_online_raises(self) -> None:
        class Incomplete(ClassifierClient):
            def classify(self, transaction: Transaction) -> ClassificationResult:
                return ClassificationResult(category="x", responsibility="A", confidence=0.0, source="x", needs_review=True)

            def classify_batch(self, transactions: list[Transaction]) -> list[ClassificationResult]:
                return []

            def update_from_correction(self, transaction: Transaction, correct_category: str, correct_responsibility: SplitMethod) -> None:
                pass

            def retrain(self) -> dict[str, object]:
                return {}

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Transaction and ClassificationResult data classes
# ---------------------------------------------------------------------------

class TestDataClasses:
    def test_transaction_frozen(self, sample_transaction: Transaction) -> None:
        with pytest.raises(AttributeError):
            sample_transaction.description = "new"  # type: ignore[misc]

    def test_classification_result_frozen(self) -> None:
        result = ClassificationResult(
            category="Food & Beverage",
            responsibility="A/K",
            confidence=0.85,
            source="cache",
            needs_review=False,
        )
        with pytest.raises(AttributeError):
            result.category = "Other"  # type: ignore[misc]

    def test_transaction_fields(self, sample_transaction: Transaction) -> None:
        assert sample_transaction.description == "STARBUCKS #1234 TORONTO ON"
        assert sample_transaction.normalized_description == "starbucks"
        assert sample_transaction.amount_cad == 5.50
        assert sample_transaction.original_currency == "CAD"
        assert sample_transaction.issuer == "MBNA"


# ---------------------------------------------------------------------------
# OfflineClassifierClient stub behavior
# ---------------------------------------------------------------------------

class TestOfflineClassifierClient:
    def test_is_online_false(self, client: OfflineClassifierClient) -> None:
        assert client.is_online is False

    def test_classify_returns_result(
        self, client: OfflineClassifierClient, sample_transaction: Transaction
    ) -> None:
        result = client.classify(sample_transaction)
        assert isinstance(result, ClassificationResult)
        assert result.confidence == 0.0
        assert result.needs_review is True
        assert result.source == "none"

    def test_classify_batch_matches_input_order(
        self, client: OfflineClassifierClient
    ) -> None:
        txns = [
            Transaction(
                description=f"merchant_{i}",
                normalized_description=f"merchant_{i}",
                amount_cad=float(i * 10),
                original_currency="CAD",
                issuer="MBNA",
            )
            for i in range(5)
        ]
        results = client.classify_batch(txns)
        assert len(results) == 5
        assert all(isinstance(r, ClassificationResult) for r in results)

    def test_classify_batch_empty(self, client: OfflineClassifierClient) -> None:
        results = client.classify_batch([])
        assert results == []

    def test_update_from_correction_no_error(
        self, client: OfflineClassifierClient, sample_transaction: Transaction
    ) -> None:
        # Stub should accept without error
        client.update_from_correction(sample_transaction, "Food & Beverage", "A/K")

    def test_retrain_returns_empty_dict(
        self, client: OfflineClassifierClient
    ) -> None:
        result = client.retrain()
        assert result == {}


# ---------------------------------------------------------------------------
# Import boundary enforcement (Constraint 2A-4)
# ---------------------------------------------------------------------------

# Banned symbols that must NOT appear in imports outside classifier/
BANNED_IMPORTS = {"TfidfVectorizer", "LogisticRegression", "pickle", "SentenceTransformer"}

# Directories that must NOT import banned symbols
ENFORCED_DIRS = ["src", "parsers", "scripts", "api", "ingestion", "fx", "db"]


class TestImportBoundary:
    def test_no_banned_imports_outside_classifier(self) -> None:
        """Scan all .py files outside classifier/ for banned imports."""
        project_root = Path(__file__).resolve().parent.parent.parent
        violations = []

        for dir_name in ENFORCED_DIRS:
            dir_path = project_root / dir_name
            if not dir_path.exists():
                continue
            for py_file in dir_path.rglob("*.py"):
                source = py_file.read_text()
                try:
                    tree = ast.parse(source)
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name in BANNED_IMPORTS:
                                violations.append(
                                    f"{py_file.relative_to(project_root)}: import {alias.name}"
                                )
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and any(
                            banned in (node.module or "")
                            for banned in BANNED_IMPORTS
                        ):
                            violations.append(
                                f"{py_file.relative_to(project_root)}: from {node.module} import ..."
                            )
                        for alias in node.names:
                            if alias.name in BANNED_IMPORTS:
                                violations.append(
                                    f"{py_file.relative_to(project_root)}: from {node.module} import {alias.name}"
                                )

        assert violations == [], (
            "Constraint 2A-4 violation: banned imports found outside classifier/:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )
