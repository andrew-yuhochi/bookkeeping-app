"""FastAPI dependency injection — DB session, classifier, FX client.

All request-scoped dependencies are defined here and injected via
FastAPI's Depends() mechanism.
"""

import logging
from collections.abc import Generator
from datetime import date
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from classifier.base import ClassifierClient
from classifier.offline import OfflineClassifierClient
from db.models import Category
from db.session import SessionLocal
from fx.boc_client import FXClient
from src.config import settings

logger = logging.getLogger(__name__)

# Module-level singletons (created once at startup)
_fx_client: Optional[FXClient] = None
_classifier: Optional[OfflineClassifierClient] = None


def get_db() -> Generator[Session, None, None]:
    """Yield a request-scoped DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_fx_client() -> FXClient:
    """Return the singleton FXClient."""
    global _fx_client
    if _fx_client is None:
        _fx_client = FXClient()
    return _fx_client


def get_classifier(db: Session) -> ClassifierClient:
    """Return the classifier, initializing on first call."""
    global _classifier
    if _classifier is None:
        _classifier = OfflineClassifierClient(
            session=db,
            household_id=settings.household_id,
        )
    return _classifier


def get_household_id() -> str:
    """Return the configured household ID."""
    return settings.household_id


def get_categories(db: Session) -> list[Category]:
    """Return all categories for the current household, ordered by sort_order."""
    return list(
        db.execute(
            select(Category)
            .where(Category.household_id == settings.household_id)
            .order_by(Category.sort_order)
        ).scalars().all()
    )


def get_current_period() -> tuple[int, int]:
    """Return the current (year, month) for default period selection."""
    today = date.today()
    return today.year, today.month
