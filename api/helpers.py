"""Shared helpers for route handlers."""

from decimal import Decimal, ROUND_HALF_UP

from classifier.offline import OfflineClassifierClient
from db.models import Category, Transaction

TWO_PLACES = Decimal("0.01")


def compute_split_amounts(
    cad_amount: Decimal, split_method: str,
) -> tuple[Decimal, Decimal]:
    """Return (andrew_amount, kristy_amount) for a given split method."""
    if split_method == "A":
        return cad_amount, Decimal("0")
    elif split_method == "K":
        return Decimal("0"), cad_amount
    else:  # A/K — 50/50
        half = (cad_amount / 2).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
        return half, cad_amount - half


def get_top_guesses(
    classifier: OfflineClassifierClient,
    txn: Transaction,
    category_map: dict[str, Category],
    slug_map: dict[str, Category],
) -> list[dict]:
    """Get top-3 category guesses for a transaction from the classifier.

    Falls back to just the stored classification if the model can't produce
    top-N predictions (e.g. no model loaded, cache-classified).

    The model may store category labels as UUIDs or slugs depending on how
    training data was prepared — slug_map handles the latter case.
    """
    model = classifier._model
    if model is not None:
        from classifier.normalizer import normalize_merchant
        normalized = normalize_merchant(txn.description)
        if normalized:
            top_n = model.predict_top_n(normalized, n=3)
            guesses = []
            for cat_id, conf in top_n:
                cat = category_map.get(cat_id) or slug_map.get(cat_id)
                if cat:
                    guesses.append({
                        "category_id": cat.id,
                        "category_name": cat.name,
                        "category_slug": cat.slug,
                        "confidence": conf,
                    })
            if guesses:
                return guesses

    # Fallback: just show the stored classification as the single guess
    cat = category_map.get(txn.category_id)
    return [{
        "category_id": txn.category_id,
        "category_name": cat.name if cat else "Unknown",
        "category_slug": cat.slug if cat else "unknown",
        "confidence": txn.classifier_confidence or 0.0,
    }]
