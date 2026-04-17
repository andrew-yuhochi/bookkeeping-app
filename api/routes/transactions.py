"""Transaction edit routes — inline responsibility toggle, move, drawer, update."""

import logging
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Request
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
    SPLIT_CYCLE,
    _review_sessions,
    get_pending_edits,
    get_session_token,
    remove_category_move,
    set_pending_edit,
    set_session_cookie,
)
from db.models import Category, Transaction

logger = logging.getLogger(__name__)

router = APIRouter()


def _calc_category_totals(
    db: Session,
    household_id: str,
    category_id: str,
    period_year: int,
    period_month: int,
    all_pending: dict[str, dict],
) -> tuple[Decimal, Decimal, Decimal]:
    """Calculate Andrew/Kristy/combined totals for a category page.

    Accounts for pending split changes and category moves.
    """
    page_txns = list(
        db.execute(
            select(Transaction).where(
                Transaction.household_id == household_id,
                Transaction.category_id == category_id,
                Transaction.accounting_period_year == period_year,
                Transaction.accounting_period_month == period_month,
            )
        ).scalars().all()
    )

    andrew_total = Decimal("0")
    kristy_total = Decimal("0")
    for t in page_txns:
        edits = all_pending.get(t.id)
        # Skip transactions that have been moved OUT of this category
        if edits and edits.get("category_id") and edits["category_id"] != category_id:
            continue
        if edits and "andrew_amount" in edits:
            andrew_total += Decimal(edits["andrew_amount"])
            kristy_total += Decimal(edits["kristy_amount"])
        else:
            if t.andrew_amount is not None:
                andrew_total += Decimal(str(t.andrew_amount))
            if t.kristy_amount is not None:
                kristy_total += Decimal(str(t.kristy_amount))

    # Also include transactions moved INTO this category from other categories
    for txn_id, edits in all_pending.items():
        if edits.get("category_id") == category_id:
            # Check this txn isn't already in page_txns (it was originally elsewhere)
            if not any(t.id == txn_id for t in page_txns):
                if "andrew_amount" in edits:
                    andrew_total += Decimal(edits["andrew_amount"])
                    kristy_total += Decimal(edits["kristy_amount"])
                else:
                    # Moved without a split change — fetch amounts from DB
                    moved = db.execute(
                        select(Transaction).where(Transaction.id == txn_id)
                    ).scalar_one_or_none()
                    if moved:
                        if moved.andrew_amount is not None:
                            andrew_total += Decimal(str(moved.andrew_amount))
                        if moved.kristy_amount is not None:
                            kristy_total += Decimal(str(moved.kristy_amount))

    return andrew_total, kristy_total, andrew_total + kristy_total


# ---------------------------------------------------------------------------
# Responsibility toggle
# ---------------------------------------------------------------------------


@router.post("/transactions/{txn_id}/responsibility", response_class=HTMLResponse)
async def toggle_responsibility(
    request: Request,
    txn_id: str,
    db: Session = Depends(get_db),
    household_id: str = Depends(get_household_id),
) -> HTMLResponse:
    """Cycle the responsibility split on a transaction (A -> K -> A/K -> A).

    Returns an HTMX partial: the updated <tr> plus OOB swaps for the
    Andrew/Kristy totals on the current category page.
    """
    txn = db.execute(
        select(Transaction).where(
            Transaction.id == txn_id,
            Transaction.household_id == household_id,
        )
    ).scalar_one_or_none()

    if txn is None:
        raise HTTPException(status_code=404, detail="Transaction not found")

    token = get_session_token(request)
    pending = _review_sessions.get(token, {})

    # Current effective split
    current_split = pending[txn_id]["split_method"] if txn_id in pending else txn.split_method
    new_split = SPLIT_CYCLE.get(current_split, "A")

    # Compute new amounts
    cad_amount = Decimal(str(txn.cad_amount)) if txn.cad_amount is not None else Decimal(str(txn.original_amount))
    new_andrew, new_kristy = compute_split_amounts(cad_amount, new_split)

    # Store in session (preserve any existing category move)
    edit: dict = {
        "split_method": new_split,
        "andrew_amount": str(new_andrew),
        "kristy_amount": str(new_kristy),
    }
    existing = pending.get(txn_id, {})
    if "category_id" in existing:
        edit["category_id"] = existing["category_id"]
        edit["original_category_id"] = existing["original_category_id"]
    set_pending_edit(token, txn_id, edit)

    # The category the transaction currently belongs to on-screen
    effective_cat_id = existing.get("category_id", txn.category_id)

    all_pending = _review_sessions.get(token, {})
    andrew_total, kristy_total, combined_total = _calc_category_totals(
        db, household_id, effective_cat_id,
        txn.accounting_period_year, txn.accounting_period_month,
        all_pending,
    )

    category = db.execute(
        select(Category).where(Category.id == effective_cat_id)
    ).scalar_one()

    templates = request.app.state.templates
    content = templates.TemplateResponse(
        "partials/responsibility_swap.html",
        {
            "request": request,
            "txn": txn,
            "category": category,
            "eff_split": new_split,
            "andrew_total": andrew_total,
            "kristy_total": kristy_total,
            "combined_total": combined_total,
        },
    ).body.decode()

    response = HTMLResponse(content=content)
    set_session_cookie(response, token)
    return response


# ---------------------------------------------------------------------------
# Move (drag-to-recategorize)
# ---------------------------------------------------------------------------


class MoveRequest(BaseModel):
    target_category_slug: str


@router.post("/transactions/{txn_id}/move")
async def move_transaction(
    request: Request,
    txn_id: str,
    body: MoveRequest,
    db: Session = Depends(get_db),
    household_id: str = Depends(get_household_id),
) -> JSONResponse:
    """Move a transaction to a different category (session-only until Save).

    Called by the drag-to-sidebar JS. Returns JSON with updated source
    page totals and metadata for the toast.
    """
    txn = db.execute(
        select(Transaction).where(
            Transaction.id == txn_id,
            Transaction.household_id == household_id,
        )
    ).scalar_one_or_none()

    if txn is None:
        raise HTTPException(status_code=404, detail="Transaction not found")

    # Resolve target category
    target_cat = db.execute(
        select(Category).where(
            Category.household_id == household_id,
            Category.slug == body.target_category_slug,
        )
    ).scalar_one_or_none()

    if target_cat is None:
        raise HTTPException(status_code=404, detail="Target category not found")

    token = get_session_token(request)
    all_pending = _review_sessions.get(token, {})

    # Current effective category
    current_cat_id = all_pending.get(txn_id, {}).get("category_id", txn.category_id)

    # Same category → no-op
    if current_cat_id == target_cat.id:
        resp = JSONResponse(content={"no_op": True})
        set_session_cookie(resp, token)
        return resp

    # Build the pending edit, preserving any existing split change
    existing = all_pending.get(txn_id, {})
    edit: dict = {
        "category_id": target_cat.id,
        "original_category_id": existing.get("original_category_id", txn.category_id),
    }
    # Preserve split changes
    if "split_method" in existing:
        edit["split_method"] = existing["split_method"]
        edit["andrew_amount"] = existing["andrew_amount"]
        edit["kristy_amount"] = existing["kristy_amount"]
    else:
        # Compute current split amounts for the totals calculation
        cad = Decimal(str(txn.cad_amount)) if txn.cad_amount is not None else Decimal(str(txn.original_amount))
        a_amt, k_amt = compute_split_amounts(cad, txn.split_method)
        edit["andrew_amount"] = str(a_amt)
        edit["kristy_amount"] = str(k_amt)

    set_pending_edit(token, txn_id, edit)

    # Recalculate source page totals (transaction is now excluded)
    all_pending = _review_sessions.get(token, {})
    andrew_total, kristy_total, combined_total = _calc_category_totals(
        db, household_id, current_cat_id,
        txn.accounting_period_year, txn.accounting_period_month,
        all_pending,
    )

    resp = JSONResponse(content={
        "no_op": False,
        "success": True,
        "merchant_name": txn.description,
        "target_category_name": target_cat.name,
        "andrew_total": f"{andrew_total:.2f}",
        "kristy_total": f"{kristy_total:.2f}",
        "combined_total": f"{combined_total:.2f}",
    })
    set_session_cookie(resp, token)
    return resp


@router.post("/transactions/{txn_id}/undo-move")
async def undo_move_transaction(
    request: Request,
    txn_id: str,
    db: Session = Depends(get_db),
    household_id: str = Depends(get_household_id),
) -> JSONResponse:
    """Undo a pending category move (revert to original category)."""
    token = get_session_token(request)
    all_pending = _review_sessions.get(token, {})

    if txn_id not in all_pending or "category_id" not in all_pending[txn_id]:
        resp = JSONResponse(content={"success": False, "reason": "No pending move"})
        set_session_cookie(resp, token)
        return resp

    remove_category_move(token, txn_id)

    resp = JSONResponse(content={"success": True})
    set_session_cookie(resp, token)
    return resp


# ---------------------------------------------------------------------------
# Transaction detail drawer
# ---------------------------------------------------------------------------


@router.get("/transactions/{txn_id}/drawer", response_class=HTMLResponse)
async def transaction_drawer(
    request: Request,
    txn_id: str,
    db: Session = Depends(get_db),
    household_id: str = Depends(get_household_id),
) -> HTMLResponse:
    """Render the detail drawer partial for a transaction."""
    txn = db.execute(
        select(Transaction).where(
            Transaction.id == txn_id,
            Transaction.household_id == household_id,
        )
    ).scalar_one_or_none()

    if txn is None:
        raise HTTPException(status_code=404, detail="Transaction not found")

    categories = get_categories(db)
    category_map = {c.id: c for c in categories}
    slug_map = {c.slug: c for c in categories}

    classifier = get_classifier(db)
    top_guesses = get_top_guesses(classifier, txn, category_map, slug_map)

    # Pending session edits for this transaction
    token = get_session_token(request)
    pending = _review_sessions.get(token, {}).get(txn_id, {})

    # Effective values (session overrides DB)
    effective_category_id = pending.get("category_id", txn.category_id)
    effective_split = pending.get("split_method", txn.split_method)

    current_year, current_month = get_current_period()

    # Review count for sidebar badge
    period_year = txn.accounting_period_year
    period_month = txn.accounting_period_month
    review_count = db.execute(
        select(func.count()).select_from(Transaction).where(
            Transaction.household_id == household_id,
            Transaction.needs_review == True,  # noqa: E712
            Transaction.accounting_period_year == period_year,
            Transaction.accounting_period_month == period_month,
        )
    ).scalar() or 0

    templates = request.app.state.templates
    content = templates.TemplateResponse(
        "partials/drawer.html",
        {
            "request": request,
            "txn": txn,
            "category": category_map.get(txn.category_id),
            "categories": categories,
            "top_guesses": top_guesses,
            "pending": pending,
            "effective_category_id": effective_category_id,
            "effective_split": effective_split,
            "period_year": period_year,
            "period_month": period_month,
            "review_count": review_count,
        },
    ).body.decode()

    response = HTMLResponse(content=content)
    set_session_cookie(response, token)
    return response


# ---------------------------------------------------------------------------
# Transaction update (from drawer)
# ---------------------------------------------------------------------------


class DrawerUpdateRequest(BaseModel):
    category_id: str
    split_method: str
    accounting_period: str  # "YYYY-MM" format
    notes: str


@router.post("/transactions/{txn_id}/update")
async def update_transaction(
    request: Request,
    txn_id: str,
    body: DrawerUpdateRequest,
    db: Session = Depends(get_db),
    household_id: str = Depends(get_household_id),
) -> JSONResponse:
    """Apply drawer changes to session state.

    Stores category, responsibility, accounting period override, and notes
    as pending edits. Changes are committed to DB in TASK-022 (Save Session).
    """
    txn = db.execute(
        select(Transaction).where(
            Transaction.id == txn_id,
            Transaction.household_id == household_id,
        )
    ).scalar_one_or_none()

    if txn is None:
        raise HTTPException(status_code=404, detail="Transaction not found")

    token = get_session_token(request)
    existing = _review_sessions.get(token, {}).get(txn_id, {})

    cad_amount = (
        Decimal(str(txn.cad_amount))
        if txn.cad_amount is not None
        else Decimal(str(txn.original_amount))
    )
    andrew_amt, kristy_amt = compute_split_amounts(cad_amount, body.split_method)

    # Parse accounting period
    period_year = txn.accounting_period_year
    period_month = txn.accounting_period_month
    period_is_override = False
    if body.accounting_period:
        try:
            parts = body.accounting_period.split("-")
            new_year = int(parts[0])
            new_month = int(parts[1])
            if new_year != txn.accounting_period_year or new_month != txn.accounting_period_month:
                period_year = new_year
                period_month = new_month
                period_is_override = True
        except (ValueError, IndexError):
            pass  # Keep original period if parsing fails

    edit: dict = {
        "category_id": body.category_id,
        "original_category_id": existing.get("original_category_id", txn.category_id),
        "split_method": body.split_method,
        "andrew_amount": str(andrew_amt),
        "kristy_amount": str(kristy_amt),
        "notes": body.notes,
        "accounting_period_year": period_year,
        "accounting_period_month": period_month,
        "accounting_period_is_override": period_is_override,
    }
    set_pending_edit(token, txn_id, edit)

    resp = JSONResponse(content={"success": True})
    set_session_cookie(resp, token)
    return resp
