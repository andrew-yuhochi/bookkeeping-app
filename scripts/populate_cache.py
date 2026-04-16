"""Populate the exact-match cache from historical seed data.

For each normalized merchant in the transactions table, determines the
most-frequent (category_id, split_method) pair and writes it to the
exact_match_cache table.

Idempotent: re-running overwrites existing cache entries with fresh counts.

TDD Section 2.2 Layer 1.
"""

import logging
import sys
from collections import Counter
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classifier.cache import ExactMatchCache  # noqa: E402
from classifier.normalizer import normalize_merchant  # noqa: E402
from db.models import Base, Transaction  # noqa: E402
from src.config import settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    engine = create_engine(settings.database_url, echo=False)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        # Load all historical transactions
        txns = session.execute(
            select(
                Transaction.description,
                Transaction.normalized_description,
                Transaction.category_id,
                Transaction.split_method,
            ).where(Transaction.source == "historical_import")
        ).all()

        logger.info("Loaded %d historical transactions", len(txns))

        # Group by normalized merchant → count (category_id, split_method) pairs
        merchant_counts: dict[str, Counter[tuple[str, str]]] = {}

        for description, normalized_desc, category_id, split_method in txns:
            # Re-normalize using the canonical normalizer (seed data used
            # simple .lower().strip(), but we want consistent keys)
            key = normalize_merchant(description)
            if not key:
                continue

            if key not in merchant_counts:
                merchant_counts[key] = Counter()
            merchant_counts[key][(category_id, split_method)] += 1

        logger.info("Found %d distinct normalized merchants", len(merchant_counts))

        # Build cache: for each merchant, pick the most-frequent pair
        cache = ExactMatchCache(session, settings.household_id)

        populated = 0
        for merchant_key, counts in merchant_counts.items():
            (best_category_id, best_split_method), best_count = counts.most_common(1)[0]

            cache.put(
                normalized_merchant=merchant_key,
                category_id=best_category_id,
                responsibility=best_split_method,
                confirmation_count=best_count,
            )
            populated += 1

        session.commit()

        logger.info("Populated %d cache entries", populated)

        # Measure hit rate on the historical data
        hits = 0
        total = 0
        for description, _norm, _cat, _split in txns:
            key = normalize_merchant(description)
            if not key:
                continue
            total += 1
            if cache.lookup(key) is not None:
                hits += 1

        hit_rate = hits / total if total > 0 else 0.0
        logger.info(
            "Cache hit rate on historical data: %d/%d = %.1f%%",
            hits, total, hit_rate * 100,
        )


if __name__ == "__main__":
    main()
