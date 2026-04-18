"""Overview page route — monthly snapshot, envelope grid, classifier panel."""

from decimal import Decimal, ROUND_HALF_UP

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.dependencies import get_categories, get_current_period, get_db, get_household_id
from db.models import BudgetEnvelope, Category, Correction, Transaction, User

router = APIRouter()

_ZERO = Decimal("0")
_CENT = Decimal("0.01")


def _pct(actual: Decimal, budget: Decimal) -> float | None:
    if budget <= 0:
        return None
    return float(actual / budget * 100)


def _status(pct: float | None) -> str:
    if pct is None:
        return "none"
    if pct < 90:
        return "green"
    if pct < 100:
        return "amber"
    return "red"


@router.get("/overview", response_class=HTMLResponse)
async def overview(
    request: Request,
    year: int | None = None,
    month: int | None = None,
    db: Session = Depends(get_db),
    household_id: str = Depends(get_household_id),
) -> HTMLResponse:
    """Overview page — monthly snapshot, envelope compliance, MoM, classifier panel."""
    current_year, current_month = get_current_period()
    period_year = year or current_year
    period_month = month or current_month

    categories = get_categories(db)
    expense_categories = [c for c in categories if c.category_type == "expense"]

    # --- All transactions for the period ---
    txns_with_cats = db.execute(
        select(Transaction, Category)
        .join(Category, Transaction.category_id == Category.id)
        .where(
            Transaction.household_id == household_id,
            Transaction.accounting_period_year == period_year,
            Transaction.accounting_period_month == period_month,
        )
    ).all()

    has_transactions = len(txns_with_cats) > 0

    # --- Review count for sidebar badge ---
    review_count = db.execute(
        select(func.count()).select_from(Transaction).where(
            Transaction.household_id == household_id,
            Transaction.needs_review == True,  # noqa: E712
            Transaction.accounting_period_year == period_year,
            Transaction.accounting_period_month == period_month,
        )
    ).scalar() or 0

    # --- Income / expense summary totals ---
    income_andrew = _ZERO
    income_kristy = _ZERO
    expense_andrew = _ZERO
    expense_kristy = _ZERO

    cat_actuals: dict[str, tuple[Decimal, Decimal]] = {}

    for txn, cat in txns_with_cats:
        a = Decimal(str(txn.andrew_amount or 0))
        k = Decimal(str(txn.kristy_amount or 0))
        if cat.category_type == "income":
            income_andrew += a
            income_kristy += k
        else:
            expense_andrew += a
            expense_kristy += k
        prev_a, prev_k = cat_actuals.get(cat.id, (_ZERO, _ZERO))
        cat_actuals[cat.id] = (prev_a + a, prev_k + k)

    net_andrew = income_andrew - expense_andrew
    net_kristy = income_kristy - expense_kristy
    income_combined = income_andrew + income_kristy
    expense_combined = expense_andrew + expense_kristy
    net_combined = income_combined - expense_combined

    # --- Budget envelopes ---
    users = db.execute(
        select(User).where(User.household_id == household_id)
    ).scalars().all()
    user_by_code = {u.person_code: u.id for u in users}

    envelopes = db.execute(
        select(BudgetEnvelope).where(
            BudgetEnvelope.household_id == household_id,
            BudgetEnvelope.period_year == period_year,
        )
    ).scalars().all()
    env_map: dict[tuple[str, str], Decimal] = {
        (e.user_id, e.category_id): Decimal(str(e.amount_cad))
        for e in envelopes
    }

    envelope_rows = []
    for cat in expense_categories:
        a_uid = user_by_code.get("A")
        k_uid = user_by_code.get("K")
        a_annual = env_map.get((a_uid, cat.id), _ZERO) if a_uid else _ZERO
        k_annual = env_map.get((k_uid, cat.id), _ZERO) if k_uid else _ZERO
        a_monthly = (a_annual / 12).quantize(_CENT, rounding=ROUND_HALF_UP)
        k_monthly = (k_annual / 12).quantize(_CENT, rounding=ROUND_HALF_UP)
        a_actual, k_actual = cat_actuals.get(cat.id, (_ZERO, _ZERO))
        a_pct = _pct(a_actual, a_monthly)
        k_pct = _pct(k_actual, k_monthly)
        envelope_rows.append({
            "category": cat,
            "andrew_actual": a_actual,
            "andrew_budget": a_monthly,
            "andrew_pct": a_pct,
            "andrew_status": _status(a_pct),
            "kristy_actual": k_actual,
            "kristy_budget": k_monthly,
            "kristy_pct": k_pct,
            "kristy_status": _status(k_pct),
        })

    # --- Month-over-month (last month's expense totals) ---
    prev_year = period_year if period_month > 1 else period_year - 1
    prev_month = period_month - 1 if period_month > 1 else 12

    prev_txns = db.execute(
        select(Transaction, Category)
        .join(Category, Transaction.category_id == Category.id)
        .where(
            Transaction.household_id == household_id,
            Transaction.accounting_period_year == prev_year,
            Transaction.accounting_period_month == prev_month,
            Category.category_type == "expense",
        )
    ).all()

    prev_cat_totals: dict[str, Decimal] = {}
    for txn, cat in prev_txns:
        a = Decimal(str(txn.andrew_amount or 0))
        k = Decimal(str(txn.kristy_amount or 0))
        prev_cat_totals[cat.id] = prev_cat_totals.get(cat.id, _ZERO) + a + k

    mom_rows = []
    for cat in expense_categories:
        a_actual, k_actual = cat_actuals.get(cat.id, (_ZERO, _ZERO))
        this_combined = a_actual + k_actual
        last_combined = prev_cat_totals.get(cat.id, _ZERO)
        change = this_combined - last_combined
        change_pct = float(change / last_combined * 100) if last_combined > 0 else None
        mom_rows.append({
            "category": cat,
            "this_month": this_combined,
            "last_month": last_combined,
            "change": change,
            "change_pct": change_pct,
        })

    # --- Classifier performance ---
    corrections_count = db.execute(
        select(func.count()).select_from(Correction).where(
            Correction.household_id == household_id,
        )
    ).scalar() or 0

    period_confidences = [
        float(txn.classifier_confidence)
        for txn, _ in txns_with_cats
        if txn.classifier_confidence is not None
    ]

    max_bin = max((h["count"] for h in _build_histogram(period_confidences)), default=1)

    return request.app.state.templates.TemplateResponse(
        "overview.html",
        {
            "request": request,
            "period_year": period_year,
            "period_month": period_month,
            "categories": categories,
            "active_page": "overview",
            "review_count": review_count,
            "has_transactions": has_transactions,
            # Summary
            "income_andrew": income_andrew,
            "income_kristy": income_kristy,
            "expense_andrew": expense_andrew,
            "expense_kristy": expense_kristy,
            "net_andrew": net_andrew,
            "net_kristy": net_kristy,
            "income_combined": income_combined,
            "expense_combined": expense_combined,
            "net_combined": net_combined,
            # Envelope grid
            "envelope_rows": envelope_rows,
            # Month-over-month
            "mom_rows": mom_rows,
            "prev_year": prev_year,
            "prev_month": prev_month,
            # Classifier
            "review_queue_count": review_count,
            "corrections_count": corrections_count,
            "confidence_histogram": _build_histogram(period_confidences),
            "histogram_max": max_bin or 1,
            "txns_count": len(txns_with_cats),
        },
    )


def _build_histogram(confidences: list[float]) -> list[dict]:
    bins = []
    for i in range(10):
        lo = i / 10
        hi = (i + 1) / 10
        if i == 9:
            count = sum(1 for c in confidences if lo <= c <= hi)
        else:
            count = sum(1 for c in confidences if lo <= c < hi)
        bins.append({"label": f"{lo:.1f}–{hi:.1f}", "count": count})
    return bins
