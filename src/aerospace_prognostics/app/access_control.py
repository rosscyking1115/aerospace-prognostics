"""Optional access gate for hosted Streamlit console demos."""

from __future__ import annotations

import hmac
import os
from typing import Any

CONSOLE_ACCESS_TOKEN_ENV = "AEROSPACE_PROGNOSTICS_CONSOLE_ACCESS_TOKEN"
CONSOLE_ACCESS_SESSION_KEY = "aerospace_prognostics_console_access_granted"


def configured_console_access_token() -> str:
    """Return the configured console access token, if one is set."""

    return os.getenv(CONSOLE_ACCESS_TOKEN_ENV, "").strip()


def console_access_required(access_token: str | None = None) -> bool:
    """Return whether the console should require an unlock token."""

    token = configured_console_access_token() if access_token is None else access_token
    return bool(token.strip())


def token_matches(candidate: str, expected: str) -> bool:
    """Compare access tokens without leaking early mismatch timing."""

    return hmac.compare_digest(candidate.strip(), expected.strip())


def render_console_access_gate(st: Any, access_token: str | None = None) -> bool:
    """Render a password gate when a hosted-demo token is configured."""

    expected_token = (
        configured_console_access_token() if access_token is None else access_token.strip()
    )
    if not expected_token:
        return True

    st.session_state.setdefault(CONSOLE_ACCESS_SESSION_KEY, False)
    if st.session_state.get(CONSOLE_ACCESS_SESSION_KEY):
        return True

    st.title("Aerospace PHM Console")
    st.subheader("Private demo access")
    entered_token = st.text_input(
        "Access token",
        type="password",
        key="console_access_token",
    )
    if st.button("Unlock", type="primary"):
        if token_matches(entered_token, expected_token):
            st.session_state[CONSOLE_ACCESS_SESSION_KEY] = True
            st.rerun()
        else:
            st.error("Access token was not accepted.")
    return False
