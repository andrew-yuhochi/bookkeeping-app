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
