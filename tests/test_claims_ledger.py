"""Keep claims.md honest by construction rather than by maintenance.

The ledger's whole premise is that every published number traces to the thing
that produced it. That premise decays silently: a function gets renamed, the
ledger keeps citing the old name, and the row still *looks* authoritative. The
first audit shipped exactly that bug -- the headline C-MAPSS row cited
``run_cmapss_validation_selected_hgb_policy_default_windows``, which has never
existed (the real symbol is prefixed ``run_all_``).

These tests make that class of error fail CI instead of surviving review.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CLAIMS_PATH = REPO_ROOT / "claims.md"

# Backticked snake_case tokens only. Paths (slashes, dots), CLI commands
# (dashes), shell invocations (spaces) and config keys (brackets) are cited in
# the same column and are deliberately not treated as symbols.
_SYMBOL_PATTERN = re.compile(r"`([a-z_][a-z0-9_]*)`")
_DEFINITION_TEMPLATE = "def {name}("


def _produced_by_cells(claims_markdown: str) -> list[str]:
    """Return the 'Produced by' cell of every ledger row."""

    cells: list[str] = []
    for line in claims_markdown.splitlines():
        if not line.startswith("|"):
            continue
        columns = [column.strip() for column in line.strip().strip("|").split("|")]
        # Ledger rows are: id | claim | produced by | allowed to claim | status
        if len(columns) != 5:
            continue
        if columns[0] in {"#", "---"} or set(columns[0]) <= {"-"}:
            continue
        cells.append(columns[2])
    return cells


def _cited_symbols(claims_markdown: str) -> set[str]:
    symbols: set[str] = set()
    for cell in _produced_by_cells(claims_markdown):
        symbols.update(_SYMBOL_PATTERN.findall(cell))
    return symbols


def _defined_symbols() -> set[str]:
    """Every ``def`` name in tracked Python, via git so untracked files cannot mask a break."""

    listing = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    defined: set[str] = set()
    for relative in listing.stdout.split():
        source = (REPO_ROOT / relative).read_text(encoding="utf-8", errors="ignore")
        defined.update(re.findall(r"^\s*(?:async )?def ([a-z_][a-z0-9_]*)\(", source, re.M))
    return defined


def test_every_symbol_cited_in_the_claims_ledger_exists() -> None:
    claims_markdown = CLAIMS_PATH.read_text(encoding="utf-8")
    cited = _cited_symbols(claims_markdown)
    assert cited, "no symbols extracted from claims.md -- the parser is broken, not the ledger"

    missing = sorted(symbol for symbol in cited if symbol not in _defined_symbols())
    assert not missing, (
        "claims.md cites symbols that do not exist in tracked source: "
        f"{missing}. A ledger row whose 'Produced by' cell does not resolve "
        "cannot substantiate its number."
    )


def test_the_symbol_check_would_catch_a_bad_citation() -> None:
    """Guard against the check passing because it never looks at anything.

    Without this, a parser that silently extracted zero symbols would leave
    ``test_every_symbol_cited_in_the_claims_ledger_exists`` permanently green
    while checking nothing at all.
    """

    fabricated = (
        "| # | Claim | Produced by | Allowed to claim | Status |\n"
        "|---|---|---|---|---|\n"
        "| X1 | some number | `a_function_that_does_not_exist` | nothing | Clean |\n"
    )

    cited = _cited_symbols(fabricated)

    assert cited == {"a_function_that_does_not_exist"}
    assert not cited <= _defined_symbols()


def test_disclosed_items_are_reachable_from_the_rows_that_carry_them() -> None:
    """Every 'see Dn' pointer must land on a real disclosed-but-unresolved item."""

    claims_markdown = CLAIMS_PATH.read_text(encoding="utf-8")
    referenced = set(re.findall(r"see (D\d+)", claims_markdown))
    defined = set(re.findall(r"\*\*(D\d+)\*\*", claims_markdown))

    assert referenced, "no disclosed-item references found -- the parser is broken"
    dangling = sorted(referenced - defined)
    assert not dangling, f"claims.md points at undefined disclosed items: {dangling}"
