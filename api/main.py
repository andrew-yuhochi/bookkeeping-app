"""FastAPI application — Bookkeeping App.

Serves Jinja2 templates with HTMX for the interactive web UI.
"""

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from api.routes.category import router as category_router
from api.routes.health import router as health_router
from api.routes.overview import router as overview_router
from api.routes.upload import router as upload_router

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

app = FastAPI(title="Bookkeeping App", version="0.1.0")

# Paths
_BASE_DIR = Path(__file__).resolve().parent.parent
_TEMPLATE_DIR = _BASE_DIR / "templates"
_STATIC_DIR = _BASE_DIR / "static"

# Templates
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
app.state.templates = templates

# Static files
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# Routes
app.include_router(health_router)
app.include_router(overview_router)
app.include_router(upload_router)
app.include_router(category_router)


@app.get("/")
async def root() -> RedirectResponse:
    """Redirect root to overview."""
    return RedirectResponse(url="/overview", status_code=307)
