"""Mode-cap truncation guardrails for the finite-depth 3-D solver.

Why these tests exist
---------------------
``mode_count`` is a *cap* on the per-row evanescent-mode requirement
``_modes_for(h, r_min)``, so a pinned value below that requirement silently
under-resolves the near field: the same physical case then returns heave added
mass 9786 (cap 48), 463427 (cap 64) and -2175 (cap 96) on the 1280-panel sphere
of the cross-validation study.  Those three numbers were recorded as an
"eigenseries does not converge" defect; they are a truncation artefact.  These
tests pin the guardrails that make the artefact visible and the automatic mode
count reachable.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem.finite_depth_solver3d import (
    _row_mode_requirements,
    assemble_finite_depth_operators_3d,
    solve_radiation_finite_depth_3d,
)
from tensorbem.geometry3d import submerged_sphere_mesh

NU = 1.0
DEPTH = 5.0
CENTER = np.array([0.0, 0.0, -2.0])


def _mesh(subdivisions: int = 0):
    return submerged_sphere_mesh(subdivisions, radius=1.0, submergence=2.0)


def test_row_requirements_exceed_any_pinned_cap_used_by_the_study():
    """The near field wants >> 96 modes: caps 48/64/96 truncate every row."""
    need = _row_mode_requirements(_mesh(), DEPTH)
    assert need.min() > 96, need.min()  # every row of the study is cap-bound
    assert need.max() >= need.min()


def test_cap_shortfall_is_reported_and_warned():
    with pytest.warns(RuntimeWarning, match="truncates"):
        ops = assemble_finite_depth_operators_3d(
            _mesh(), NU, DEPTH, mode_count=16, warn_on_truncation=True
        )
    assert ops.mode_cap_bound_rows == _mesh().panel_count
    assert ops.mode_shortfall > 1.0
    # the requirement at the nearest pair, divided by the cap
    assert ops.mode_shortfall == pytest.approx(
        float(_row_mode_requirements(_mesh(), DEPTH).max() / 16), rel=1e-12
    )


def test_no_shortfall_at_or_above_the_automatic_requirement():
    ops = assemble_finite_depth_operators_3d(_mesh(), NU, DEPTH)
    assert ops.mode_cap_bound_rows == 0
    assert ops.mode_shortfall == 1.0 or ops.mode_shortfall < 1.0 + 1e-12


def test_automatic_mode_count_reproduces_a_cap_above_the_requirement():
    """Auto == a cap that clears every row: the per-row grouping is exact there."""
    need = int(_row_mode_requirements(_mesh(), DEPTH).max())
    auto = solve_radiation_finite_depth_3d(_mesh(), NU, DEPTH, modes=[2])
    wide = solve_radiation_finite_depth_3d(
        _mesh(), NU, DEPTH, modes=[2], mode_count=2 * need
    )
    assert auto.added_mass[0, 0] == pytest.approx(wide.added_mass[0, 0], rel=1e-10)
    assert auto.damping[0, 0] == pytest.approx(wide.damping[0, 0], rel=1e-10)


def test_truncated_cap_changes_the_answer_by_more_than_the_accuracy_standard():
    """A 16-mode cap is under-resolved and must not be mistaken for convergence."""
    capped = solve_radiation_finite_depth_3d(
        _mesh(), NU, DEPTH, modes=[2], mode_count=16
    )
    auto = solve_radiation_finite_depth_3d(_mesh(), NU, DEPTH, modes=[2])
    assert abs(capped.added_mass[0, 0] / auto.added_mass[0, 0] - 1.0) > 0.15


def test_block_budget_keeps_peak_temporaries_bounded(monkeypatch):
    """The block size honours ``_BLOCK_ELEMENTS`` instead of a fixed row count."""
    from tensorbem import finite_depth_solver3d as fds

    seen: list[int] = []
    original = fds._quadrature_remainder_block

    def spy(rows, *args, **kwargs):
        seen.append(int(np.asarray(rows).size))
        return original(rows, *args, **kwargs)

    monkeypatch.setattr(fds, "_quadrature_remainder_block", spy)
    monkeypatch.setattr(fds, "_BLOCK_ELEMENTS", 2000)
    fds.assemble_finite_depth_operators_3d(_mesh(), NU, DEPTH, mode_count=64)
    n_qp = 6 * 20
    assert seen, "the block helper was never called"
    assert max(seen) * n_qp * 64 <= 2000 * 2 + n_qp * 64  # budget respected up to one row