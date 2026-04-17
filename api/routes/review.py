"""Needs-Review Queue routes — review flagged transactions."""

import logging
from decimal import Decimal, ROUND_HALF_UP

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.dependencies import (
    get_categories,
    get_classifier,
    get_current_period,
    get_db,
    get_household_id,
)
from api.session_store import (
    _review_sessions,
    get_session_token,
    is_reviewed_in_session,
    set_review_correction,
    set_session_cookie,
)
from classifier.offline import OfflineClassifierClient
from db.models import Category, Transaction

logger = logging.getLogger(__name__)

router = APIRouter()

TWO_PLACES = Decimal("0.01")


def _compute_split_amounts(
    cad_amount: Decimal, split_method: str,
) -> tuple[Decimal, Decimal]:
    """Return (andrew_amount, kristy_amount) for a given split method."""
    if split_method == "A":
        return cad_amount, Decimal("0")
    elif split_method == "K":
        return Decimal("0"), cad_amount
    else:  # A/K — 50/50
        half = (cad_amount / 2).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
        return cad_amount - half, half


def _get_top_guesses(
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


@router.get("/review", response_class=HTMLResponse)
async def review_queue(
    request: Request,
    year: int | None = None,
    month: int | None = None,
    db: Session = Depends(get_db),
    household_id: str = Depends(get_household_id),
) -> HTMLResponse:
    """Needs-Review Queue page — transactions flagged for human review."""
    current_year, current_month = get_current_period()
    period_year = year or current_year
    period_month = month or current_month

    # Fetch all needs_review transactions for the period
    transactions = list(
        db.execute(
            select(Transaction).where(
                Transaction.household_id == household_id,
                Transaction.needs_review == True,  # noqa: E712
                Transaction.accounting_period_year == period_year,
                Transaction.accounting_period_month == period_month,
            ).order_by(Transaction.cash_date.asc(), Transaction.created_at.asc())
        ).scalars().all()
    )

    # Filter out transactions already reviewed in this session
    token = get_session_token(request)
    pending_txns = [
        t for t in transactions
        if not is_reviewed_in_session(token, t.id)
    ]

    # Build category lookup (by UUID and by slug — model may use either)
    categories = get_categories(db)
    category_map = {c.id: c for c in categories}
    slug_map = {c.slug: c for c in categories}

    # Get classifier for top-N predictions
    classifier = get_classifier(db)

    # Build review items with top-3 guesses
    review_items = []
    for txn in pending_txns:
        top_guesses = _get_top_guesses(classifier, txn, category_map, slug_map)
        cat = category_map.get(txn.category_id)
        review_items.append({
            "txn": txn,
            "category": cat,
            "top_guesses": top_guesses,
            "source_label": txn.source or "unknown",
        })

    # Total review count (including already-reviewed, for the sidebar badge)
    total_review_count = len(transactions)
    remaining_count = len(pending_txns)

    response = request.app.state.templates.TemplateResponse(
        "review_queue.html",
        {
            "request": request,
            "period_year": period_year,
            "period_month": period_month,
            "categories": categories,
            "active_page": "review",
            "review_items": review_items,
            "review_count": total_review_count,
            "remaining_count": remaining_count,
        },
    )
    set_session_cookie(response, token)
    return response


class CorrectionRequest(BaseModel):
    category_id: str
    split_method: str


@router.post("/review/corrections/{txn_id}", response_class=HTMLResponse)
async def accept_correction(
    request: Request,
    txn_id: str,
    body: CorrectionRequest,
    db: Session = Depends(get_db),
    household_id: str = Depends(get_household_id),
) -> HTMLResponse:
    """Accept or correct a single review-queue item.

    Stores the correction in session state and returns an HTMX partial
    that removes the item from the queue + updates the badge count.
    """
    txn = db.execute(
        select(Transaction).where(
            Transaction.id == txn_id,
            Transaction.household_id == household_id,
        )
    ).scalar_one_or_none()

    if txn is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Transaction not found")

    # Compute split amounts
    cad_amount = (
        Decimal(str(txn.cad_amount))
        if txn.cad_amount is not None
        else Decimal(str(txn.original_amount))
    )
    andrew_amt, kristy_amt = _compute_split_amounts(cad_amount, body.split_method)

    token = get_session_token(request)
    set_review_correction(
        token=token,
        txn_id=txn_id,
        category_id=body.category_id,
        original_category_id=txn.category_id,
        split_method=body.split_method,
        andrew_amount=str(andrew_amt),
        kristy_amount=str(kristy_amt),
    )

    # Count remaining unreviewed items for badge update
    current_year, current_month = get_current_period()
    period_year = int(request.query_params.get("year", current_year))
    period_month = int(request.query_params.get("month", current_month))

    all_review_txns = list(
        db.execute(
            select(Transaction.id).where(
                Transaction.household_id == household_id,
                Transaction.needs_review == True,  # noqa: E712
                Transaction.accounting_period_year == period_year,
                Transaction.accounting_period_month == period_month,
            )
        ).scalars().all()
    )
    remaining = sum(
        1 for tid in all_review_txns
        if not is_reviewed_in_session(token, tid)
    )

    templates = request.app.state.templates
    content = templates.TemplateResponse(
        "partials/review_accepted.html",
        {
            "request": request,
            "txn_id": txn_id,
            "remaining_count": remaining,
        },
    ).body.decode()

    response = HTMLResponse(content=content)
    set_session_cookie(response, token)
    return response


@router.post("/review/accept-all-confident", response_class=HTMLResponse)
async def accept_all_confident(
    request: Request,
    year: int | None = None,
    month: int | None = None,
    db: Session = Depends(get_db),
    household_id: str = Depends(get_household_id),
) -> HTMLResponse:
    """Batch-accept all review items where top guess confidence >= 0.70.

    Redirects back to the review page (which will show remaining items).
    """
    current_year, current_month = get_current_period()
    period_year = year or current_year
    period_month = month or current_month

    transactions = list(
        db.execute(
            select(Transaction).where(
                Transaction.household_id == household_id,
                Transaction.needs_review == True,  # noqa: E712
                Transaction.accounting_period_year == period_year,
                Transaction.accounting_period_month == period_month,
            )
        ).scalars().all()
    )

    token = get_session_token(request)
    accepted_count = 0

    for txn in transactions:
        if is_reviewed_in_session(token, txn.id):
            continue

        conf = txn.classifier_confidence or 0.0
        if conf >= 0.70:
            cad_amount = (
                Decimal(str(txn.cad_amount))
                if txn.cad_amount is not None
                else Decimal(str(txn.original_amount))
            )
            andrew_amt, kristy_amt = _compute_split_amounts(
                cad_amount, txn.split_method,
            )
            set_review_correction(
                token=token,
                txn_id=txn.id,
                category_id=txn.category_id,
                original_category_id=txn.category_id,
                split_method=txn.split_method,
                andrew_amount=str(andrew_amt),
                kristy_amount=str(kristy_amt),
            )
            accepted_count += 1

    logger.info("Batch-accepted %d confident transactions", accepted_count)

    from fastapi.responses import RedirectResponse
    response = RedirectResponse(
        url=f"/review?year={period_year}&month={period_month}",
        status_code=303,
    )
    set_session_cookie(response, token)
    return response
