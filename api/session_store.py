"""Server-side review session store.

Holds pending edits (responsibility, category changes) in memory keyed
by a session token stored in a browser cookie.  Single-user PoC on
localhost — if the server restarts, session state is lost (acceptable).
"""

import uuid

from fastapi import Request, Response

SESSION_COOKIE = "review_session"

# Store: {session_token: {transaction_id: {"split_method": str, "andrew_amount": str, "kristy_amount": str}}}
_review_sessions: dict[str, dict[str, dict]] = {}

# Split cycle order per discovery: A → K → A/K → A
SPLIT_CYCLE = {"A": "K", "K": "A/K", "A/K": "A"}


def get_session_token(request: Request) -> str:
    """Get or create a session token from the request cookie."""
    token = request.cookies.get(SESSION_COOKIE)
    if token and token in _review_sessions:
        return token
    token = str(uuid.uuid4())
    _review_sessions[token] = {}
    return token


def get_pending_edits(request: Request) -> dict[str, dict]:
    """Return the pending edits dict for this session (may be empty)."""
    token = request.cookies.get(SESSION_COOKIE)
    if token and token in _review_sessions:
        return _review_sessions[token]
    return {}


def set_pending_edit(token: str, txn_id: str, edit: dict) -> None:
    """Store a pending edit for a transaction."""
    if token not in _review_sessions:
        _review_sessions[token] = {}
    _review_sessions[token][txn_id] = edit


def set_session_cookie(response: Response, token: str) -> None:
    """Set the session cookie on the response."""
    response.set_cookie(
        SESSION_COOKIE,
        token,
        httponly=True,
        samesite="lax",
    )


def remove_category_move(token: str, txn_id: str) -> None:
    """Remove a category move from a pending edit, keeping split changes if any."""
    session = _review_sessions.get(token, {})
    if txn_id not in session:
        return
    edit = session[txn_id]
    edit.pop("category_id", None)
    edit.pop("original_category_id", None)
    # If no split change remains either, remove the edit entirely
    if "split_method" not in edit:
        session.pop(txn_id, None)


def get_pending_count(token: str) -> int:
    """Return the number of pending edits in this session."""
    return len(_review_sessions.get(token, {}))


def set_review_correction(
    token: str,
    txn_id: str,
    category_id: str,
    original_category_id: str,
    split_method: str,
    andrew_amount: str,
    kristy_amount: str,
) -> None:
    """Store a review-queue correction for a transaction.

    Merges with any existing pending edit (e.g. a prior responsibility toggle).
    """
    if token not in _review_sessions:
        _review_sessions[token] = {}
    existing = _review_sessions[token].get(txn_id, {})
    existing.update({
        "category_id": category_id,
        "original_category_id": original_category_id,
        "split_method": split_method,
        "andrew_amount": andrew_amount,
        "kristy_amount": kristy_amount,
        "reviewed": True,
    })
    _review_sessions[token][txn_id] = existing


def is_reviewed_in_session(token: str, txn_id: str) -> bool:
    """Check if a transaction has been reviewed/accepted in this session."""
    session = _review_sessions.get(token, {})
    return session.get(txn_id, {}).get("reviewed", False)
