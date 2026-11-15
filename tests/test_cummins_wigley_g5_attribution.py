"""What the G5 failure of ``CASE-BEM-CUMMINS-WIGLEY`` actually measures.

The gate is a predeclared identity gate (round trip ``B -> K -> (B, A)``, tol
5 %).  These tests pin the *decomposition* established by
``docs/dev_report_m21140_cummins_g5_ainf_attribution.md`` and
``scripts/cummins_g5_ainf_attribution.py``:

* the inverse legs of the round trip are quadrature-exact when they are handed
  the exact kernel, so a residual is never a property of the transform;
* shifting ``A_inf`` by a known ``delta`` moves the G5 statistic to
  ``|delta| / max|A - (A_inf + delta)|`` and leaves the residual *flat* in
  ``omega`` -- the signature that identifies an ``A_inf`` bias rather than a
  band or transform error;
* on the cached panel sweep the flat offset accounts for >90 % of the shipped
  G5 residual on all six DOF, and the bias-free shape residual is <3 %;
* closing the band above ``omega_max`` with the algebraic tail buys <10 %
  relative, so high-frequency truncation is *not* the missing 20-48 %;
* G2 (kernel at zero) cannot fail: it compares the exact piecewise-linear
  forward quadrature against the trapezoid rule of the same nodes.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from cummins_g5_ainf_attribution import (  # noqa: E402
    BAND,
    SWEEP_NPZ,
    g2_vacuity,
    panel_rows,
    synthetic_biased_a_inf,
)
from cummins_kernel_high_frequency_audit import (  # noqa: E402
    kernel_tail_closed,
    round_trip_errors,
    synthetic_arm,
)
from tensorbem import time_domain as td  # noqa: E402

TAU = np.arange(0, 40.0001, 0.02)
_HAVE_SWEEP = SWEEP_NPZ.exists()
_needs_sweep = pytest.mark.skipif(not _HAVE_SWEEP, reason="cached panel sweep not shipped")


def _omega(n: int = 41, low: float = 0.2, high: float = 12.0) -> np.ndarray:
    return np.geomspace(low, high, n)


def test_exact_kernel_round_trip_is_quadrature_exact():
    """With the exact kernel the inverse legs carry no error of their own."""
    arm = synthetic_arm(_omega(), TAU)["exact_kernel"]
    assert arm["worst"] < 2.0e-3, arm["worst"]


def test_a_inf_shift_is_flat_in_omega_and_lands_on_the_g5_statistic():
    arm = synthetic_biased_a_inf(_omega(), TAU, delta=7.5)
    assert arm["damping_error_unshifted"] < 2.0e-3
    # measured statistic equals the predicted |delta|/scale to <1 %
    assert arm["identity_error"] < 1.0e-2 * arm["predicted_residual"]
    # ... and the added-mass residual is flat: this is what makes the panel
    # attribution a measurement (an offset) rather than an inference.
    assert arm["flatness"] < 1.0e-2


def test_shifted_a_inf_reproduces_the_g5_residual_monotonically():
    """Bigger bias -> bigger statistic; the relation is the predeclared scale."""
    small = synthetic_biased_a_inf(_omega(), TAU, delta=1.0)
    large = synthetic_biased_a_inf(_omega(), TAU, delta=7.5)
    assert small["measured_residual"] < large["measured_residual"]
    for arm in (small, large):
        assert abs(arm["measured_residual"] - arm["predicted_residual"]) < 5.0e-3


@_needs_sweep
def test_panel_g5_failure_is_the_a_inf_offset_not_the_transform():
    rows = panel_rows(TAU)
    assert set(rows) == {"surge", "sway", "heave", "roll", "pitch", "yaw"}
    for label, row in rows.items():
        assert row["damping_leg_error"] < 5.0e-3, (label, row["damping_leg_error"])
        assert row["bias_explained_fraction"] > 0.9, (label, row["bias_explained_fraction"])
        assert row["g5_shape_residual"] < 0.03, (label, row["g5_shape_residual"])
        # the shipped gate value really is (offset + shape), not something else
        assert abs(row["g5_shipped"]
                   - (row["bias_over_scale"] + 0.5 * row["g5_shape_residual"])) < 2.0e-2


@_needs_sweep
def test_high_frequency_tail_closure_does_not_explain_g5():
    """A measured negative result: the band above omega_max is not the 20-48 %."""
    data = np.load(SWEEP_NPZ)
    omega = np.asarray(data["omega"], dtype=float)
    damping = np.asarray(data["damping"], dtype=float)[:, 2, 2]
    added = np.asarray(data["added_mass"], dtype=float)[:, 2, 2]
    a_inf = float(td.fit_added_mass_asymptote(omega, added).added_mass_inf)
    shipped = round_trip_errors(td.radiation_memory_kernel(omega, damping, TAU),
                                omega, added, damping, a_inf, BAND)["worst"]
    closed = round_trip_errors(kernel_tail_closed(omega, damping, TAU),
                               omega, added, damping, a_inf, BAND)["worst"]
    assert closed < shipped
    assert (shipped - closed) / shipped < 0.10, (shipped, closed)


@_needs_sweep
def test_g2_kernel_at_zero_gate_cannot_fail():
    """Pinned vacuity: G2 compares the forward quadrature with its own trapezoid rule."""
    rows = g2_vacuity(TAU)
    for label in ("surge", "sway", "heave", "roll", "pitch", "yaw"):
        assert rows[label]["g2_rel_err"] < 1.0e-12, (label, rows[label]["g2_rel_err"])