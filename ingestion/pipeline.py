"""Full ingestion pipeline — wires parser, normalizer, classifier, and DB.

The callable surface for the FastAPI upload route:
  IngestionPipeline.ingest(pdf_bytes, filename) → IngestionResult

TDD Section 2.1 (parsing), 2.2 (classification), 2.3 (FX resolution).
"""

import hashlib
import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from classifier.base import ClassificationResult, ClassifierClient, Transaction
from db.models import Category, Statement, Transaction as TransactionModel
from fx.boc_client import FXClient
from ingestion.normalizer import NormalizedTransaction, TransactionNormalizer
from parsers.base import IssuerParser, UnknownIssuerError
from parsers.models import ParsedTransaction
from parsers.registry import ParserRegistry

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of ingesting a single PDF statement."""

    filename: str
    issuer: str = ""
    statement_id: str = ""
    parsed: int = 0
    classified: int = 0
    flagged: int = 0
    duplicates: int = 0
    errors: int = 0
    error_messages: list[str] = field(default_factory=list)


class IngestionPipeline:
    """End-to-end PDF statement ingestion pipeline.

    Orchestrates: issuer detection → parsing → FX normalization →
    classification → duplicate detection → DB write.

    Args:
        session: SQLAlchemy session for DB operations.
        household_id: Household UUID for scoped operations.
        classifier: ClassifierClient instance (only ABC imported — 2A-4).
        fx_client: FXClient for BoC Valet lookups (optional, creates default).
        registry: ParserRegistry (optional, uses default REGISTERED_PARSERS).
    """

    def __init__(
        self,
        session: Session,
        household_id: str,
        classifier: ClassifierClient,
        fx_client: Optional[FXClient] = None,
        registry: Optional[ParserRegistry] = None,
    ) -> None:
        self._session = session
        self._household_id = household_id
        self._classifier = classifier
        self._normalizer = TransactionNormalizer(fx_client=fx_client)
        self._registry = registry or ParserRegistry()
        self._category_lookup: Optional[dict[str, str]] = None

    def ingest(self, pdf_bytes: bytes, filename: str) -> IngestionResult:
        """Ingest a single PDF statement end-to-end.

        Returns an IngestionResult with counts of parsed, classified,
        flagged, duplicate, and errored transactions.
        """
        result = IngestionResult(filename=filename)

        # --- Step 1: Detect issuer ---
        try:
            import pdfplumber
            from io import BytesIO

            pdf = pdfplumber.open(BytesIO(pdf_bytes))
            first_page_text = pdf.pages[0].extract_text() or "" if pdf.pages else ""
            pdf.close()
        except Exception as e:
            result.errors = 1
            result.error_messages.append(f"Failed to read PDF: {e}")
            return result

        try:
            parser = self._registry.detect_issuer(filename, first_page_text)
            result.issuer = parser.issuer_name
        except UnknownIssuerError as e:
            result.errors = 1
            result.error_messages.append(str(e))
            return result

        # --- Step 2: Parse ---
        try:
            parsed_txns = parser.parse(pdf_bytes)
            result.parsed = len(parsed_txns)
        except Exception as e:
            result.errors = 1
            result.error_messages.append(f"Parse error: {e}")
            return result

        if not parsed_txns:
            return result

        # --- Step 3: Create statement record ---
        statement = Statement(
            household_id=self._household_id,
            issuer=parser.issuer_name,
            filename=filename,
            parse_status="ok" if result.errors == 0 else "partial",
        )
        self._session.add(statement)
        self._session.flush()  # Get the statement ID
        result.statement_id = statement.id

        # --- Step 4: Normalize (FX resolution) ---
        normalized = self._normalizer.normalize_batch(parsed_txns)

        # --- Step 5: Classify ---
        classifier_txns = [n.transaction for n in normalized]
        classifications = self._classifier.classify_batch(classifier_txns)

        # --- Step 6: Deduplicate and write to DB ---
        category_lookup = self._get_category_lookup()

        for parsed, norm, classification in zip(parsed_txns, normalized, classifications):
            # Duplicate detection
            if self._is_duplicate(parsed):
                result.duplicates += 1
                continue

            # Resolve category_id from classification
            category_id = classification.category
            if category_id not in category_lookup:
                # Classification returned a category name or unknown ID;
                # fall back to "Other"
                category_id = self._get_other_category_id()

            needs_review = classification.needs_review or norm.needs_fx_review

            db_txn = TransactionModel(
                household_id=self._household_id,
                statement_id=statement.id,
                cash_date=parsed.cash_date,
                accounting_period_year=norm.accounting_period_year,
                accounting_period_month=norm.accounting_period_month,
                accounting_period_is_override=False,
                description=parsed.description,
                normalized_description=norm.transaction.normalized_description,
                original_amount=parsed.original_amount,
                original_currency=parsed.original_currency,
                fx_rate=norm.fx_rate,
                fx_rate_source=norm.fx_rate_source,
                cad_amount=norm.cad_amount,
                category_id=category_id,
                split_method=classification.responsibility,
                classifier_confidence=classification.confidence,
                classifier_source=classification.source,
                needs_review=needs_review,
                source="pdf_upload",
                source_ref=f"pdf:{filename}:p{parsed.statement_page}",
            )
            self._session.add(db_txn)

            result.classified += 1
            if needs_review:
                result.flagged += 1

        self._session.flush()

        logger.info(
            "Ingestion complete: %s — %d parsed, %d classified, %d flagged, "
            "%d duplicates, %d errors",
            filename, result.parsed, result.classified, result.flagged,
            result.duplicates, result.errors,
        )

        return result

    def _is_duplicate(self, parsed: ParsedTransaction) -> bool:
        """Check if a transaction already exists in the DB.

        Duplicate key: (issuer, cash_date, original_amount, description_hash).
        """
        desc_hash = hashlib.md5(parsed.description.encode()).hexdigest()[:16]

        existing = self._session.execute(
            select(TransactionModel.id).where(
                TransactionModel.household_id == self._household_id,
                TransactionModel.cash_date == parsed.cash_date,
                TransactionModel.original_amount == parsed.original_amount,
                TransactionModel.source_ref.like(f"pdf:%"),
                TransactionModel.description == parsed.description,
            ).limit(1)
        ).scalar_one_or_none()

        return existing is not None

    def _get_category_lookup(self) -> dict[str, str]:
        """Build a category_id → name lookup, cached per pipeline instance."""
        if self._category_lookup is None:
            categories = self._session.execute(
                select(Category).where(
                    Category.household_id == self._household_id
                )
            ).scalars().all()
            self._category_lookup = {cat.id: cat.name for cat in categories}
        return self._category_lookup

    def _get_other_category_id(self) -> str:
        """Get the category_id for 'Other' (fallback)."""
        category_lookup = self._get_category_lookup()
        for cat_id, name in category_lookup.items():
            if name == "Other":
                return cat_id
        # Should never happen if seed data is correct
        raise ValueError("No 'Other' category found in the database")
