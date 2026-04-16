"""Overview page route."""

from datetime import date

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.dependencies import get_categories, get_current_period, get_db, get_household_id
from db.models import Category, Transaction

router = APIRouter()


@router.get("/overview", response_class=HTMLResponse)
async def overview(
    request: Request,
    year: int | None = None,
    month: int | None = None,
    db: Session = Depends(get_db),
    household_id: str = Depends(get_household_id),
) -> HTMLResponse:
    """Overview page — monthly snapshot."""
    current_year, current_month = get_current_period()
    period_year = year or current_year
    period_month = month or current_month

    categories = get_categories(db)

    return request.app.state.templates.TemplateResponse(
        "overview.html",
        {
            "request": request,
            "period_year": period_year,
            "period_month": period_month,
            "categories": categories,
            "active_page": "overview",
        },
    )
