"""Needs-Review Queue routes — review flagged transactions."""

import logging
import threading
import uuid as _uuid_lib
from datetime import datetime
from decimal import Decimal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
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
from api.helpers import compute_split_amounts, get_top_guesses
from api.session_store import (
    SESSION_COOKIE,
    _review_sessions,
    clear_session,
    get_pending_count,
    get_pending_edits,
    get_session_token,
    is_reviewed_in_session,
    set_review_correction,
    set_session_cookie,
)
from db.models import Correction, ExactMatchCache, Transaction, User

# Retrain status store: {retrain_id: {"status": "running"|"done"|"error", ...}}
_retrain_status: dict[str, dict] = {}

logger = logging.getLogger(__name__)

router = APIRouter()


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
        top_guesses = get_top_guesses(classifier, txn, category_map, slug_map)
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
    andrew_amt, kristy_amt = compute_split_amounts(cad_amount, body.split_method)

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
            andrew_amt, kristy_amt = compute_split_amounts(
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


# ---------------------------------------------------------------------------
# Pending count (for save button badge)
# ---------------------------------------------------------------------------


@router.get("/review/pending-count")
async def pending_count_endpoint(request: Request) -> JSONResponse:
    """Return the number of pending edits for this session."""
    token = request.cookies.get(SESSION_COOKIE, "")
    count = get_pending_count(token)
    return JSONResponse(content={"count": count})


# ---------------------------------------------------------------------------
# Save review session
# ---------------------------------------------------------------------------


@router.post("/review/save-session")
async def save_session(
    request: Request,
    db: Session = Depends(get_db),
    household_id: str = Depends(get_household_id),
) -> JSONResponse:
    """Commit all pending session edits to the DB and trigger classifier retrain.

    All DB writes happen in a single transaction — if any write fails,
    all changes roll back.  Classifier retrain runs in a background thread
    so the endpoint returns immediately after committing.
    """
    token = get_session_token(request)
    pending = get_pending_edits(request)

    if not pending:
        return JSONResponse(
            content={"success": False, "reason": "no_changes"},
            status_code=400,
        )

    # Resolve a user_id for Correction rows (Andrew = person_code "A")
    andrew_user = db.execute(
        select(User).where(
            User.household_id == household_id,
            User.person_code == "A",
        )
    ).scalar_one_or_none()
    user_id = andrew_user.id if andrew_user else household_id

    corrections_count = 0
    try:
        for txn_id, edit in pending.items():
            txn = db.execute(
                select(Transaction).where(Transaction.id == txn_id)
            ).scalar_one_or_none()
            if txn is None:
                logger.warning("save-session: txn %s not found — skipping", txn_id)
                continue

            orig_category_id = txn.category_id
            orig_split = txn.split_method
            new_category_id = edit.get("category_id", txn.category_id)
            new_split = edit.get("split_method", txn.split_method)

            # Apply all field updates
            txn.category_id = new_category_id
            txn.split_method = new_split
            if "andrew_amount" in edit:
                txn.andrew_amount = edit["andrew_amount"]
            if "kristy_amount" in edit:
                txn.kristy_amount = edit["kristy_amount"]
            if "notes" in edit and edit["notes"] is not None:
                txn.notes = edit["notes"]
            if "accounting_period_year" in edit:
                txn.accounting_period_year = edit["accounting_period_year"]
                txn.accounting_period_month = edit["accounting_period_month"]
                txn.accounting_period_is_override = edit.get(
                    "accounting_period_is_override", False
                )
            if edit.get("reviewed"):
                txn.needs_review = False
                txn.is_manually_reviewed = True

            # Insert Correction row when category or split changed
            if new_category_id != orig_category_id or new_split != orig_split:
                db.add(
                    Correction(
                        household_id=household_id,
                        transaction_id=txn_id,
                        user_id=user_id,
                        prev_category_id=orig_category_id,
                        new_category_id=new_category_id,
                        prev_split_method=orig_split,
                        new_split_method=new_split,
                    )
                )
                corrections_count += 1

            # Upsert ExactMatchCache for this merchant
            normalized = txn.normalized_description
            if normalized:
                now = datetime.utcnow()
                cache_row = db.execute(
                    select(ExactMatchCache).where(
                        ExactMatchCache.household_id == household_id,
                        ExactMatchCache.normalized_merchant == normalized,
                    )
                ).scalar_one_or_none()
                if cache_row:
                    if cache_row.category_id == new_category_id:
                        cache_row.confirmation_count += 1
                    else:
                        cache_row.category_id = new_category_id
                        cache_row.responsibility = new_split
                        cache_row.confirmation_count = 1
                    cache_row.last_confirmed_at = now
                else:
                    db.add(
                        ExactMatchCache(
                            household_id=household_id,
                            normalized_merchant=normalized,
                            category_id=new_category_id,
                            responsibility=new_split,
                            confirmation_count=1,
                            last_confirmed_at=now,
                        )
                    )

        db.commit()

    except Exception as exc:
        db.rollback()
        logger.error("save-session DB error: %s", exc, exc_info=True)
        return JSONResponse(
            content={"success": False, "reason": "db_error", "message": str(exc)},
            status_code=500,
        )

    # Clear session state
    clear_session(token)

    # Kick off background retrain with a fresh DB session
    retrain_id = str(_uuid_lib.uuid4())
    _retrain_status[retrain_id] = {"status": "running"}

    classifier = get_classifier(db)

    def _retrain_worker() -> None:
        from db.session import SessionLocal

        fresh_db = SessionLocal()
        try:
            classifier._session = fresh_db
            metrics = classifier.retrain()
            _retrain_status[retrain_id] = {"status": "done", "metrics": metrics}
            logger.info("Background retrain complete: %s", metrics)
        except Exception as exc:  # noqa: BLE001
            _retrain_status[retrain_id] = {"status": "error", "message": str(exc)}
            logger.error("Background retrain failed: %s", exc, exc_info=True)
        finally:
            fresh_db.close()

    threading.Thread(target=_retrain_worker, daemon=True).start()

    response = JSONResponse(
        content={
            "success": True,
            "corrections_count": corrections_count,
            "retrain_id": retrain_id,
        }
    )
    set_session_cookie(response, token)
    return response


# ---------------------------------------------------------------------------
# Retrain status poll
# ---------------------------------------------------------------------------


@router.get("/review/retrain-status/{retrain_id}")
async def get_retrain_status(retrain_id: str) -> JSONResponse:
    """Return the current status of a background retrain job."""
    status = _retrain_status.get(retrain_id, {"status": "unknown"})
    return JSONResponse(content=status)
