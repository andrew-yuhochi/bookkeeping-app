"""Settings page routes — budget envelope management."""

import logging
from decimal import Decimal, InvalidOperation

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.dependencies import get_categories, get_current_period, get_db, get_household_id
from db.models import BudgetEnvelope, Transaction, User
from sqlalchemy import func

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_envelopes_for_year(
    db: Session, household_id: str, period_year: int
) -> dict[tuple[str, str], Decimal]:
    """Return {(user_id, category_id): annual_amount} for the given year."""
    rows = db.execute(
        select(BudgetEnvelope).where(
            BudgetEnvelope.household_id == household_id,
            BudgetEnvelope.period_year == period_year,
        )
    ).scalars().all()
    return {(r.user_id, r.category_id): Decimal(str(r.amount_cad)) for r in rows}


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(
    request: Request,
    year: int | None = None,
    db: Session = Depends(get_db),
    household_id: str = Depends(get_household_id),
) -> HTMLResponse:
    """Settings page — view and edit annual budget envelopes."""
    current_year, current_month = get_current_period()
    period_year = year or current_year

    categories = get_categories(db)
    expense_categories = [c for c in categories if c.category_type == "expense"]

    users = db.execute(
        select(User).where(User.household_id == household_id)
    ).scalars().all()
    user_by_code = {u.person_code: u for u in users}

    env_map = _get_envelopes_for_year(db, household_id, period_year)

    # Build rows: {category, andrew_annual, kristy_annual}
    envelope_rows = []
    for cat in expense_categories:
        a_user = user_by_code.get("A")
        k_user = user_by_code.get("K")
        a_annual = env_map.get((a_user.id, cat.id), Decimal("0")) if a_user else Decimal("0")
        k_annual = env_map.get((k_user.id, cat.id), Decimal("0")) if k_user else Decimal("0")
        envelope_rows.append({
            "category": cat,
            "andrew_annual": a_annual,
            "kristy_annual": k_annual,
        })

    review_count = db.execute(
        select(func.count()).select_from(Transaction).where(
            Transaction.household_id == household_id,
            Transaction.needs_review == True,  # noqa: E712
            Transaction.accounting_period_year == current_year,
            Transaction.accounting_period_month == current_month,
        )
    ).scalar() or 0

    error = request.query_params.get("error")

    return request.app.state.templates.TemplateResponse(
        "settings.html",
        {
            "request": request,
            "period_year": period_year,
            "period_month": current_month,
            "categories": categories,
            "active_page": "settings",
            "review_count": review_count,
            "envelope_rows": envelope_rows,
            "user_by_code": user_by_code,
            "error": error,
            "years": list(range(2024, current_year + 3)),
        },
    )


@router.post("/settings/envelopes")
async def save_envelopes(
    request: Request,
    db: Session = Depends(get_db),
    household_id: str = Depends(get_household_id),
) -> RedirectResponse:
    """Save budget envelope amounts for the selected year.

    Expects form fields named `envelope_{category_id}_{person_code}` with
    decimal amount values (annual totals in CAD).  Upserts into budget_envelopes.
    """
    form = await request.form()
    current_year, _ = get_current_period()
    period_year_raw = form.get("period_year", str(current_year))

    try:
        period_year = int(period_year_raw)
    except ValueError:
        return RedirectResponse(url="/settings?error=invalid_year", status_code=303)

    users = db.execute(
        select(User).where(User.household_id == household_id)
    ).scalars().all()
    user_by_code = {u.person_code: u for u in users}

    # Parse and validate all envelope values before touching the DB
    updates: list[tuple[str, str, Decimal]] = []  # (user_id, category_id, amount)
    for key, raw_value in form.items():
        if not key.startswith("envelope_"):
            continue
        parts = key.split("_", 2)  # ["envelope", category_id, person_code]
        if len(parts) != 3:
            continue
        _, category_id, person_code = parts
        user = user_by_code.get(person_code)
        if user is None:
            continue
        try:
            amount = Decimal(str(raw_value).strip())
        except InvalidOperation:
            return RedirectResponse(
                url=f"/settings?year={period_year}&error=invalid_amount",
                status_code=303,
            )
        if amount < 0:
            return RedirectResponse(
                url=f"/settings?year={period_year}&error=negative_amount",
                status_code=303,
            )
        updates.append((user.id, category_id, amount))

    # Upsert all validated values
    for user_id, category_id, amount in updates:
        existing = db.execute(
            select(BudgetEnvelope).where(
                BudgetEnvelope.household_id == household_id,
                BudgetEnvelope.category_id == category_id,
                BudgetEnvelope.user_id == user_id,
                BudgetEnvelope.period_year == period_year,
            )
        ).scalar_one_or_none()

        if existing:
            existing.amount_cad = str(amount)
        else:
            db.add(
                BudgetEnvelope(
                    household_id=household_id,
                    category_id=category_id,
                    user_id=user_id,
                    period_year=period_year,
                    amount_cad=str(amount),
                )
            )

    db.commit()
    logger.info("Saved %d envelope entries for %s/%d", len(updates), household_id, period_year)

    return RedirectResponse(
        url=f"/settings?year={period_year}&saved=1",
        status_code=303,
    )
