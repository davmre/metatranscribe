from __future__ import annotations

import hmac
from functools import wraps
from typing import Callable

from flask import current_app, redirect, request, session, url_for


def check_password(candidate: str) -> bool:
    """Constant-time comparison of a submitted password against the configured one."""
    configured = current_app.config.get("WEB_PASSWORD")
    if not configured or candidate is None:
        return False
    return hmac.compare_digest(candidate.encode("utf-8"), configured.encode("utf-8"))


def login_required(view: Callable) -> Callable:
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("authed"):
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped
