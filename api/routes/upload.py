"""Upload flow — file upload, ingestion dispatch, and SSE progress.

TDD Section 2.5 (upload route, SSE streaming), UX-SPEC Touchpoint 1.
"""

import json
import logging
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from threading import Thread
from typing import Optional

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlalchemy.orm import Session

from api.dependencies import (
    get_categories,
    get_classifier,
    get_current_period,
    get_db,
    get_fx_client,
    get_household_id,
)
from classifier.base import ClassifierClient
from db.session import SessionLocal
from fx.boc_client import FXClient
from ingestion.pipeline import IngestionPipeline, IngestionResult
from src.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# In-memory upload session store
# ---------------------------------------------------------------------------


@dataclass
class FileStatus:
    """Status of a single file in an upload session."""

    filename: str
    status: str = "queued"  # queued, parsing, classifying, done, error
    issuer: str = ""
    parsed: int = 0
    classified: int = 0
    flagged: int = 0
    duplicates: int = 0
    error_message: str = ""


@dataclass
class UploadSession:
    """Tracks the state of a multi-file upload."""

    session_id: str
    files: dict[str, FileStatus] = field(default_factory=dict)
    complete: bool = False


# Session store — keyed by session_id
_sessions: dict[str, UploadSession] = {}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/upload", response_class=HTMLResponse)
async def upload_page(
    request: Request,
    db: Session = Depends(get_db),
) -> HTMLResponse:
    """Render the upload page."""
    categories = get_categories(db)
    period_year, period_month = get_current_period()

    return request.app.state.templates.TemplateResponse(
        "upload.html",
        {
            "request": request,
            "categories": categories,
            "period_year": period_year,
            "period_month": period_month,
            "active_page": "upload",
        },
    )


@router.post("/upload")
async def upload_files(
    request: Request,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
) -> dict:
    """Accept uploaded PDF files and start ingestion in a background thread.

    Returns a session_id for SSE status polling.
    """
    session_id = str(uuid.uuid4())
    session = UploadSession(session_id=session_id)

    # Read file bytes up front (UploadFile is async/temporary)
    file_data: list[tuple[str, bytes]] = []
    for f in files:
        content = await f.read()
        filename = f.filename or f"upload_{uuid.uuid4().hex[:8]}.pdf"
        session.files[filename] = FileStatus(filename=filename)
        file_data.append((filename, content))

    _sessions[session_id] = session

    # Start background ingestion thread
    household_id = get_household_id()
    thread = Thread(
        target=_run_ingestion,
        args=(session_id, file_data, household_id),
        daemon=True,
    )
    thread.start()

    return {"session_id": session_id, "file_count": len(file_data)}


@router.get("/upload/status/{session_id}")
async def upload_status(session_id: str) -> StreamingResponse:
    """SSE endpoint streaming ingestion progress per file."""
    return StreamingResponse(
        _sse_generator(session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Background ingestion
# ---------------------------------------------------------------------------


def _run_ingestion(
    session_id: str,
    file_data: list[tuple[str, bytes]],
    household_id: str,
) -> None:
    """Run ingestion for all files in a background thread.

    Each file is processed independently — a failure in one does not
    affect others.
    """
    session = _sessions.get(session_id)
    if session is None:
        return

    # Create a fresh DB session for this thread
    db = SessionLocal()
    try:
        from classifier.offline import OfflineClassifierClient

        classifier = OfflineClassifierClient(
            session=db,
            household_id=household_id,
        )
        fx_client = FXClient()

        pipeline = IngestionPipeline(
            session=db,
            household_id=household_id,
            classifier=classifier,
            fx_client=fx_client,
        )

        for filename, pdf_bytes in file_data:
            file_status = session.files[filename]

            try:
                # Update status: parsing
                file_status.status = "parsing"

                # Run the full pipeline (parse + classify + write)
                file_status.status = "classifying"
                result = pipeline.ingest(pdf_bytes, filename)

                # Update with results
                file_status.issuer = result.issuer
                file_status.parsed = result.parsed
                file_status.classified = result.classified
                file_status.flagged = result.flagged
                file_status.duplicates = result.duplicates

                if result.errors > 0:
                    file_status.status = "error"
                    file_status.error_message = "; ".join(result.error_messages)
                else:
                    file_status.status = "done"

            except Exception as e:
                logger.exception("Ingestion failed for %s", filename)
                file_status.status = "error"
                file_status.error_message = str(e)

        db.commit()

    except Exception as e:
        logger.exception("Background ingestion thread error")
        db.rollback()
    finally:
        session.complete = True
        db.close()


# ---------------------------------------------------------------------------
# SSE generator
# ---------------------------------------------------------------------------


import asyncio


async def _sse_generator(session_id: str) -> AsyncGenerator[str, None]:
    """Yield SSE events as ingestion progresses."""
    session = _sessions.get(session_id)
    if session is None:
        yield _sse_event({"error": "Unknown session"})
        return

    # Track which file states we've already sent
    last_sent: dict[str, str] = {}

    while True:
        all_done = True

        for filename, file_status in session.files.items():
            current_state = _file_status_dict(file_status)
            current_key = json.dumps(current_state, sort_keys=True)

            # Only send if state changed
            if last_sent.get(filename) != current_key:
                yield _sse_event(current_state)
                last_sent[filename] = current_key

            if file_status.status not in ("done", "error"):
                all_done = False

        if all_done and session.complete:
            # Send final summary
            total_parsed = sum(f.parsed for f in session.files.values())
            total_flagged = sum(f.flagged for f in session.files.values())
            total_duplicates = sum(f.duplicates for f in session.files.values())
            total_errors = sum(1 for f in session.files.values() if f.status == "error")

            yield _sse_event({
                "type": "complete",
                "total_parsed": total_parsed,
                "total_flagged": total_flagged,
                "total_duplicates": total_duplicates,
                "total_errors": total_errors,
            })
            break

        await asyncio.sleep(0.3)


def _file_status_dict(fs: FileStatus) -> dict:
    """Convert FileStatus to a dict for SSE event data."""
    d: dict = {
        "type": "file_status",
        "filename": fs.filename,
        "status": fs.status,
    }
    if fs.issuer:
        d["issuer"] = fs.issuer
    if fs.status == "done":
        d["parsed"] = fs.parsed
        d["classified"] = fs.classified
        d["flagged"] = fs.flagged
        if fs.duplicates > 0:
            d["duplicates"] = fs.duplicates
    elif fs.status == "error":
        d["message"] = fs.error_message
    return d


def _sse_event(data: dict) -> str:
    """Format a dict as an SSE event string."""
    return f"data: {json.dumps(data)}\n\n"
