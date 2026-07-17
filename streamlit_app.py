"""Streamlit Community Cloud entry point for the read-only PHM console.

Community Cloud runs this file by default and does not build `Dockerfile.demo`,
so the evidence the console needs is not pre-baked. This entry point:

1. makes the ``src`` layout importable without an editable install,
2. defaults the console to read-only (so a public demo cannot be mutated),
3. provisions the demo workspace once per process, then
4. renders the console.

For local development, run the console module directly instead::

    uv run streamlit run src/aerospace_prognostics/app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# A public hosted demo defaults to read-only; an operator can still override it.
os.environ.setdefault("AEROSPACE_PROGNOSTICS_CONSOLE_READ_ONLY", "true")

import streamlit as st  # noqa: E402

from aerospace_prognostics.app.streamlit_app import (  # noqa: E402
    DEFAULT_DATABASE,
    DEFAULT_WORKSPACE,
    main,
)


@st.cache_resource(show_spinner="Preparing demo evidence…")
def _bootstrap() -> dict[str, object]:
    """Provision the demo workspace once per app process."""

    from aerospace_prognostics.app.bootstrap import ensure_demo_workspace

    return ensure_demo_workspace(DEFAULT_WORKSPACE, DEFAULT_DATABASE)


_bootstrap()
main()
