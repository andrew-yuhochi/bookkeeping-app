"""Transaction edit routes — inline responsibility toggle, move, update."""

import logging
from decimal import Decimal, ROUND_HALF_UP

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.dependencies import get_db, get_household_id
from api.session_store import (
    SPLIT_CYCLE,
    get_pending_edits,
    get_session_token,
    set_pending_edit,
    set_session_cookie,
)
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
        return half, cad_amount - half


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
    # Load the transaction
    txn = db.execute(
        select(Transaction).where(
            Transaction.id == txn_id,
            Transaction.household_id == household_id,
        )
    ).scalar_one_or_none()

    if txn is None:
        raise HTTPException(status_code=404, detail="Transaction not found")

    # Get or create session
    token = get_session_token(request)
    pending = get_pending_edits(request)

    # Current effective split: from session if pending, else from DB
    current_split = pending[txn_id]["split_method"] if txn_id in pending else txn.split_method

    # Cycle
    new_split = SPLIT_CYCLE.get(current_split, "A")

    # Compute new amounts
    cad_amount = Decimal(str(txn.cad_amount)) if txn.cad_amount is not None else Decimal(str(txn.original_amount))
    new_andrew, new_kristy = _compute_split_amounts(cad_amount, new_split)

    # Store in session
    set_pending_edit(token, txn_id, {
        "split_method": new_split,
        "andrew_amount": str(new_andrew),
        "kristy_amount": str(new_kristy),
    })

    # Recalculate page totals for this category + period (with all pending edits applied).
    # Read directly by token since the cookie may not exist yet on the first toggle.
    from api.session_store import _review_sessions
    all_pending = _review_sessions.get(token, {})

    page_txns = list(
        db.execute(
            select(Transaction).where(
                Transaction.household_id == household_id,
                Transaction.category_id == txn.category_id,
                Transaction.accounting_period_year == txn.accounting_period_year,
                Transaction.accounting_period_month == txn.accounting_period_month,
            )
        ).scalars().all()
    )

    andrew_total = Decimal("0")
    kristy_total = Decimal("0")
    for t in page_txns:
        if t.id in all_pending:
            andrew_total += Decimal(all_pending[t.id]["andrew_amount"])
            kristy_total += Decimal(all_pending[t.id]["kristy_amount"])
        else:
            if t.andrew_amount is not None:
                andrew_total += Decimal(str(t.andrew_amount))
            if t.kristy_amount is not None:
                kristy_total += Decimal(str(t.kristy_amount))
    combined_total = andrew_total + kristy_total

    # Load category for the row partial
    category = db.execute(
        select(Category).where(Category.id == txn.category_id)
    ).scalar_one()

    # Render HTMX response
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
