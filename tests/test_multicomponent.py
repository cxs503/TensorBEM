"""Guards for the multiply-connected (annulus) 2-D Laplace case.

``docs/verification.md`` registers the annulus as *"后续：需多边界组件"*. This
file does two different jobs and keeps them apart on purpose:

**What works, asserted as a pass.** The loop-orientation convention that makes
the concatenated assembly mean anything (outer loop counter-clockwise, inner
loop clockwise, every normal away from the fluid), the exact closed-form traces
``u = A log r + B``, the per-loop flux against ``+/- 2 pi A`` (which is
discretisation-free for a polygonal loop because the chord integrals
telescope), and the reduced boundary-integral identity on the exact traces,
which is the only direct check that both components were assembled with a
consistent normal sense.

**What does not work, asserted as a *failure*.** Refining the mesh does not
bring the unknown trace ``q`` or the reconstructed interior field below 3%:
measured ``q`` is 11-17% and the interior field is ~28% at every level, and the
interior error is a near-constant offset. These two tests therefore assert that
the error is *large*, with a message saying so. They are registered debt, not
acceptance: when the near-capacity mode of the single-layer operator is dealt
with properly, they will flip and force this docstring plus
``docs/multicomponent_domain.md`` to be rewritten. Pinning a failure keeps it
visible; deleting it would let the case be claimed as closed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.benchmarks.annulus_log_potential import (
    run_level,
    run_study,
    wrong_winding_control,
)
from tensorbem.geometry import PanelMesh
from tensorbem.multicomponent import (
    Annulus,
    annulus_mesh,
    check_loop_orientation,
    log_potential,
    log_potential_boundary_data,
    log_potential_normal_derivative,
    loop_fluxes,
    solve_annulus_dirichlet,
)
from tensorbem.solver import assemble_matrices

STUDY = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "multicomponent"
    / "outputs"
    / "study.json"
)


def _flipped_inner(annulus: Annulus) -> PanelMesh:
    vertices = annulus.mesh.vertices.copy()
    inner = np.arange(annulus.outer_panels, annulus.panel_count)
    vertices[inner] = vertices[inner][::-1]
    return PanelMesh(vertices)


def test_annulus_mesh_orients_each_loop_away_from_the_fluid() -> None:
    """Outer loop CCW, inner loop CW: both normal sets point out of the fluid."""
    annulus = annulus_mesh(32, 16, outer_radius=1.0, inner_radius=0.5)
    assert annulus.panel_count == 48
    vertices = annulus.mesh.vertices
    outer = vertices[annulus.outer_slice]
    inner = vertices[annulus.inner_slice]
    outer_area = 0.5 * float(
        np.sum(
            outer[:, 0] * np.roll(outer[:, 1], -1)
            - outer[:, 1] * np.roll(outer[:, 0], -1)
        )
    )
    inner_area = 0.5 * float(
        np.sum(
            inner[:, 0] * np.roll(inner[:, 1], -1)
            - inner[:, 1] * np.roll(inner[:, 0], -1)
        )
    )
    assert outer_area > 0.0
    assert inner_area < 0.0
    # ``PanelMesh.area`` is a shoelace sum over the *concatenated* vertex list,
    # so it also picks up the bridging edge from the last inner vertex back to
    # the first outer one. It stays positive (which is what PanelMesh needs),
    # but it is not the annulus area -- the per-loop areas above are.
    assert annulus.mesh.area > 0.0
    # Geometric normal test, independent of the shoelace sign.
    normals = annulus.mesh.normals
    centres = annulus.mesh.centers
    outer_radial = np.sum(normals[annulus.outer_slice] * centres[annulus.outer_slice], axis=1)
    inner_radial = np.sum(normals[annulus.inner_slice] * centres[annulus.inner_slice], axis=1)
    assert np.all(outer_radial > 0.0), "outer normals must point away from the centre"
    assert np.all(inner_radial < 0.0), "inner normals must point into the hole"
    check_loop_orientation(annulus)  # must not raise


def test_wrong_winding_is_rejected_before_the_solve() -> None:
    """A clockwise inner loop is refused with the reason, not silently solved."""
    annulus = annulus_mesh(16, 8)
    bad = Annulus(
        mesh=_flipped_inner(annulus),
        outer_panels=annulus.outer_panels,
        inner_panels=annulus.inner_panels,
        outer_radius=annulus.outer_radius,
        inner_radius=annulus.inner_radius,
    )
    with pytest.raises(ValueError, match="clockwise"):
        check_loop_orientation(bad)
    with pytest.raises(ValueError, match="clockwise"):
        solve_annulus_dirichlet(bad)


def test_the_rejected_winding_would_otherwise_fail_silently() -> None:
    """Why the guard exists: the wrong winding solves cleanly and lies.

    The system stays square and its residual is at rounding level, so no
    residual-based check can catch the flip. Only the orientation guard (or a
    reference solution, i.e. exactly the verification this file is about) can.
    """
    control = wrong_winding_control(32, reference_length=0.5)
    assert control["relative_residual_l2"] < 1.0e-12
    assert control["q_relative_l2_error"] > 0.5


def test_exact_traces_satisfy_the_boundary_integral_identity() -> None:
    """Both components assembled with one consistent normal sense.

    Feeding the closed-form ``u`` and ``q = +/- A/r`` back into
    ``S q - (I/2 + H) u`` must leave a residual that is small compared with the
    magnitude of both terms; if the inner loop's normals were flipped the
    residual would be the size of ``q`` itself.
    """
    annulus = annulus_mesh(64, 32)
    single, double = assemble_matrices(annulus.mesh)
    values = log_potential_boundary_data(annulus)
    q = log_potential_normal_derivative(annulus)
    residual = 0.5 * values + double @ values - single @ q
    scale = np.sqrt(np.mean((single @ q) ** 2))
    assert np.sqrt(np.mean(residual**2)) < 0.05 * scale


def test_closed_form_flux_keeps_the_sign_of_each_loop() -> None:
    """The analytic ``q`` gives a positive outer flux and a negative inner one.

    The *chord integral* of ``du/dn`` is ``2 pi A`` for any panel count, but the
    panel-*midpoint* rule in :func:`loop_fluxes` evaluates ``A/r`` at the point
    of the chord closest to the centre, i.e. at the largest ``q`` on that chord,
    so it overestimates both loops by a few percent. The guard therefore checks
    the sign of each loop and that the two loops cancel to within 10% of their
    own magnitude; the quantitative flux criterion on the *solved* trace lives in
    ``test_per_loop_flux_meets_the_three_percent_criterion``.
    """
    annulus = annulus_mesh(16, 8, inner_phase=0.3)
    fluxes = loop_fluxes(annulus, log_potential_normal_derivative(annulus))
    assert fluxes["outer"] > 0.0
    assert fluxes["inner"] < 0.0
    magnitude = 0.5 * (abs(fluxes["outer"]) + abs(fluxes["inner"]))
    assert abs(fluxes["net"]) < 0.10 * magnitude


def test_per_loop_flux_meets_the_three_percent_criterion() -> None:
    """The measured per-loop flux is inside 3% from 96 panels on."""
    row = run_level(32, reference_length=0.5)
    assert abs(row["flux_outer_rel_error"]) < 0.03
    assert abs(row["flux_inner_rel_error"]) < 0.03
    assert row["flux_net_over_exact"] < 1.0e-2


@pytest.mark.parametrize(
    "quantity, floor",
    [
        ("q_relative_l2_error", 0.03),
        ("interior_field_relative_l2_error", 0.03),
    ],
)
def test_accuracy_above_three_percent_is_registered_debt(quantity: str, floor: float) -> None:
    """Registered HONEST FAIL: refinement does not reach 3% on these traces.

    Measured at 96 panels: ``q`` 14.5%, interior field 29.2%. The interior error
    is a near-constant offset, the signature of the near-capacity mode of the
    single-layer operator on a doubly-connected boundary rather than of panel
    resolution. Do not relax ``floor`` -- fix the mode, then rewrite this test
    and ``docs/multicomponent_domain.md`` together.
    """
    row = run_level(32, reference_length=0.5)
    assert row[quantity] > floor, (
        f"{quantity} is now {row[quantity]:.3e}, at or below the 3% criterion: the "
        "registered limitation appears fixed -- update docs/multicomponent_domain.md, "
        "the module docstring and this test together instead of deleting the guard"
    )


def test_committed_study_reproduces_from_the_module() -> None:
    """The documented numbers cannot drift away from the code."""
    assert STUDY.exists(), f"missing committed study artifact: {STUDY}"
    committed = json.loads(STUDY.read_text())
    assert committed["schema"] == "tensorbem.annulus_log_potential.study/1"
    fresh = run_study((8, 16), reference_length=committed["reference_length"])
    for old, new in zip(committed["rows"], fresh["rows"]):
        for key in (
            "panel_count",
            "condition_number",
            "q_relative_l2_error",
            "flux_outer_rel_error",
            "flux_inner_rel_error",
            "interior_field_relative_l2_error",
        ):
            assert new[key] == pytest.approx(old[key], rel=1e-9, abs=1e-12), key
    assert log_potential(np.e) == pytest.approx(1.0, rel=1e-15)
