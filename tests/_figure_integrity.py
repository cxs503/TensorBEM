"""Guard against verification figures silently vanishing from the pushed tree.

Background (measured 2026-11-11 on ``origin/main`` @ 6600bcc): the platform push
gate drops ``*.png`` outside ``docs/`` (``tp_pusher._EXCLUDE2``), so every
verification bundle that commits figures under ``benchmarks/verified/<case>/
figures/`` arrives on GitHub **without** them.  Eight figure-presence
assertions then fail on ``main`` -- the case records are complete but their
figures are dangling.  A committed binary is therefore not a durable invariant
in this repository; the durable invariant is that each case's figures are
**reproducible from the committed numbers**, with no hand-typed value.

This module re-exports the shared probe from :mod:`tensorbem.figure_integrity`
for backward compatibility with tests that import from ``_figure_integrity``.
The canonical implementation lives in ``src/tensorbem/figure_integrity.py``.
"""

from __future__ import annotations

import pathlib
from typing import Callable, Iterable

# Re-export from the canonical implementation
from tensorbem.figure_integrity import (
    MINIMUM_FIGURE_BYTES,
    CaseIntegrityReport,
    FigureResult,
    check_all_cases,
    check_case_integrity,
    check_report_linkage,
)


def assert_reproducible_figures(
    figures: Iterable[pathlib.Path],
    *,
    minimum: int = 1,
    minimum_bytes: int = 10_000,
    embedded_in: str | None = None,
) -> list[pathlib.Path]:
    """The figures must be regenerable, non-trivial, and referenced by the report.

    ``figures`` are the paths a case's own figure builder just produced into a
    temporary directory from the committed ``result.json``; nothing here reads a
    committed image.  ``minimum_bytes`` keeps a blank/aborted render from
    passing (the renderers are matplotlib vector-into-raster dumps, so a real
    figure is ~10 kB or more).  ``embedded_in`` is the case report text: every
    figure must still be linked from it, otherwise the regenerated plot is not
    the one the record claims.
    """
    paths = [pathlib.Path(p) for p in figures]
    assert len(paths) >= minimum, (len(paths), minimum)
    for path in paths:
        assert path.is_file(), path
        size = path.stat().st_size
        assert size > minimum_bytes, (path, size)
        if embedded_in is not None:
            assert path.name in embedded_in, (path.name, "not linked from the report")
    return paths


def builders_are_callable(builders: Iterable[Callable[..., None]]) -> None:
    """Cheap negative control: a builder list must not be empty or non-callable."""
    items = list(builders)
    assert items
    assert all(callable(item) for item in items), items
