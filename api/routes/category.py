"""Category page routes — per-category transaction table with envelope bars."""

import logging
from decimal import Decimal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.dependencies import get_categories, get_current_period, get_db, get_household_id
from api.session_store import get_pending_edits
from db.models import BudgetEnvelope, Category, Transaction, User

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_category_page_data(
    request: Request,
    category_slug: str,
    category_type: str,
    year: int | None,
    month: int | None,
    db: Session,
    household_id: str,
) -> dict:
    """Build template context for a category page (shared by expense and income routes)."""
    current_year, current_month = get_current_period()
    period_year = year or current_year
    period_month = month or current_month

    # Look up the category
    category = db.execute(
        select(Category).where(
            Category.household_id == household_id,
            Category.slug == category_slug,
            Category.category_type == category_type,
        )
    ).scalar_one_or_none()

    if category is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Category '{category_slug}' not found")

    # Fetch transactions for this category + period, sorted by cash_date desc
    transactions = list(
        db.execute(
            select(Transaction).where(
                Transaction.household_id == household_id,
                Transaction.category_id == category.id,
                Transaction.accounting_period_year == period_year,
                Transaction.accounting_period_month == period_month,
            ).order_by(Transaction.cash_date.desc(), Transaction.created_at.desc())
        ).scalars().all()
    )

    # Read pending session edits (responsibility toggles not yet saved to DB)
    pending_edits = get_pending_edits(request)

    # Compute per-person totals, overlaying pending edits on DB values
    andrew_total = Decimal("0")
    kristy_total = Decimal("0")
    for txn in transactions:
        if txn.id in pending_edits:
            andrew_total += Decimal(pending_edits[txn.id]["andrew_amount"])
            kristy_total += Decimal(pending_edits[txn.id]["kristy_amount"])
        else:
            if txn.andrew_amount is not None:
                andrew_total += Decimal(str(txn.andrew_amount))
            if txn.kristy_amount is not None:
                kristy_total += Decimal(str(txn.kristy_amount))
    combined_total = andrew_total + kristy_total

    # Fetch users for this household
    users = list(
        db.execute(
            select(User).where(User.household_id == household_id)
        ).scalars().all()
    )
    andrew_user = next((u for u in users if u.person_code == "A"), None)
    kristy_user = next((u for u in users if u.person_code == "K"), None)

    # Fetch budget envelopes for this category + year
    envelopes = list(
        db.execute(
            select(BudgetEnvelope).where(
                BudgetEnvelope.household_id == household_id,
                BudgetEnvelope.category_id == category.id,
                BudgetEnvelope.period_year == period_year,
            )
        ).scalars().all()
    )

    andrew_envelope = Decimal("0")
    kristy_envelope = Decimal("0")
    for env in envelopes:
        if andrew_user and env.user_id == andrew_user.id:
            andrew_envelope = Decimal(str(env.amount_cad))
        elif kristy_user and env.user_id == kristy_user.id:
            kristy_envelope = Decimal(str(env.amount_cad))

    # Compute envelope percentages (avoid division by zero)
    andrew_pct = int(andrew_total * 100 / andrew_envelope) if andrew_envelope else 0
    kristy_pct = int(kristy_total * 100 / kristy_envelope) if kristy_envelope else 0

    # Review count for sidebar badge (all needs_review in this period)
    review_count = db.execute(
        select(func.count()).select_from(Transaction).where(
            Transaction.household_id == household_id,
            Transaction.needs_review == True,  # noqa: E712
            Transaction.accounting_period_year == period_year,
            Transaction.accounting_period_month == period_month,
        )
    ).scalar() or 0

    categories = get_categories(db)

    active_prefix = "category" if category_type == "expense" else "income"

    return {
        "request": request,
        "period_year": period_year,
        "period_month": period_month,
        "categories": categories,
        "active_page": f"{active_prefix}:{category_slug}",
        "category": category,
        "transactions": transactions,
        "pending_edits": pending_edits,
        "andrew_total": andrew_total,
        "kristy_total": kristy_total,
        "combined_total": combined_total,
        "andrew_envelope": andrew_envelope,
        "kristy_envelope": kristy_envelope,
        "andrew_pct": andrew_pct,
        "kristy_pct": kristy_pct,
        "review_count": review_count,
    }


@router.get("/category/{category_slug}", response_class=HTMLResponse)
async def category_page(
    request: Request,
    category_slug: str,
    year: int | None = None,
    month: int | None = None,
    db: Session = Depends(get_db),
    household_id: str = Depends(get_household_id),
) -> HTMLResponse:
    """Expense category page — envelope bars, per-person totals, transaction table."""
    ctx = _get_category_page_data(
        request, category_slug, "expense", year, month, db, household_id,
    )
    return request.app.state.templates.TemplateResponse("category.html", ctx)


@router.get("/income/{category_slug}", response_class=HTMLResponse)
async def income_page(
    request: Request,
    category_slug: str,
    year: int | None = None,
    month: int | None = None,
    db: Session = Depends(get_db),
    household_id: str = Depends(get_household_id),
) -> HTMLResponse:
    """Income category page — per-person totals and transaction table."""
    ctx = _get_category_page_data(
        request, category_slug, "income", year, month, db, household_id,
    )
    return request.app.state.templates.TemplateResponse("category.html", ctx)
