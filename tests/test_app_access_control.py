from __future__ import annotations

from typing import Any

from aerospace_prognostics.app.access_control import (
    CONSOLE_ACCESS_SESSION_KEY,
    console_access_required,
    render_console_access_gate,
    token_matches,
)


class _FakeStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, Any] = {}
        self.titles: list[str] = []
        self.subheaders: list[str] = []
        self.errors: list[str] = []
        self.text_inputs: dict[str, str] = {}
        self.buttons: dict[str, bool] = {}
        self.rerun_count = 0
        self.widget_keys: list[str] = []

    def title(self, text: str) -> None:
        self.titles.append(text)

    def subheader(self, text: str) -> None:
        self.subheaders.append(text)

    def text_input(self, label: str, **kwargs: Any) -> str:
        key = kwargs.get("key")
        if key is not None:
            self.widget_keys.append(str(key))
        return self.text_inputs.get(label, "")

    def button(self, label: str, **_kwargs: Any) -> bool:
        return self.buttons.get(label, False)

    def error(self, text: str) -> None:
        self.errors.append(text)

    def rerun(self) -> None:
        self.rerun_count += 1


def test_console_access_is_disabled_when_no_token_configured() -> None:
    st = _FakeStreamlit()

    assert not console_access_required("")
    assert render_console_access_gate(st, "") is True

    assert st.titles == []
    assert CONSOLE_ACCESS_SESSION_KEY not in st.session_state


def test_console_access_gate_blocks_until_valid_token() -> None:
    st = _FakeStreamlit()
    st.text_inputs["Access token"] = "wrong"
    st.buttons["Unlock"] = True

    assert console_access_required("demo-secret")
    assert render_console_access_gate(st, "demo-secret") is False

    assert st.titles == ["Aerospace PHM Console"]
    assert st.subheaders == ["Private demo access"]
    assert st.errors == ["Access token was not accepted."]
    assert not st.session_state[CONSOLE_ACCESS_SESSION_KEY]


def test_console_access_gate_accepts_valid_token_and_reruns() -> None:
    st = _FakeStreamlit()
    st.text_inputs["Access token"] = " demo-secret "
    st.buttons["Unlock"] = True

    assert render_console_access_gate(st, "demo-secret") is False

    assert st.session_state[CONSOLE_ACCESS_SESSION_KEY] is True
    assert st.rerun_count == 1
    assert token_matches("demo-secret", " demo-secret ")


def test_console_access_gate_allows_authenticated_session() -> None:
    st = _FakeStreamlit()
    st.session_state[CONSOLE_ACCESS_SESSION_KEY] = True

    assert render_console_access_gate(st, "demo-secret") is True

    assert st.titles == []
