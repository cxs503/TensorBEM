"""Operator-level regression tests for the 6-node quadratic panel path (m7526).

These tests exist because the shipped suite for :mod:`tensorbem.panel_quadratic`
was entirely *self-referential*: partition of unity, positive diagonals and a
small linear-solve residual all pass for a wrong operator.  The three defects
found on 2026-11-09 are pinned here against closed-form references:

A. ``quadratic_shape_gradients_ref`` had wrong rows for the two
   edge shape functions ``N5``/``N6`` (Jacobian therefore wrong, so every
   integral over a curved element was wrong);
B. ``duffy_self_quadrature`` permuted the singular point to the wrong vertex for
   ``apex_node`` 1 and 2, and edge-midpoint collocation was routed through the
   non-singular rule;
C. ``assemble_quadratic_matrices`` summed only element-local rows, so the matrix
   was a patch operator rather than the boundary operator (analytic image
   residual 2.84 instead of decreasing with ``h``).

Analytic references used below (unit sphere, kernel ``G = 1/(4 pi |x-y|)``):

* ``int_T N_k dS = (0, 0, 0, A/3, A/3, A/3)`` -- quadratic mass matrix;
* ``S 1 = R`` -- potential of a unit-density spherical shell;
* ``S q = R q / (2n+1)`` for the trace of a harmonic polynomial of degree ``n``;
* ``g = z`` has the exact single-layer density ``q = 3 z``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.panel_quadratic import (
    duffy_self_quadrature,
    gauss_triangle_rule,
    icosphere_quadratic,
    quadratic_shape_functions,
    quadratic_shape_gradients_ref,
    assemble_quadratic_matrices,
    solve_laplace_dirichlet_quadratic,
)
from tensorbem.benchmarks import quadratic_panel_verification as study

FOUR_PI = 4.0 * math.pi


@pytest.fixture(scope="module")
def coarse():
    """80-element quadratic icosphere plus its assembled single-layer matrix."""
    mesh = icosphere_quadratic(1, radius=1.0)
    S, D = assemble_quadratic_matrices(mesh, quadrature_order=8, self_order=16)
    return mesh, S, D


# ---------------------------------------------------------------------------
# A. shape-function gradients
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("L1,L2", [(0.2, 0.3), (0.5, 0.25), (0.1, 0.6), (0.7, 0.2)])
def test_shape_gradients_match_finite_differences(L1, L2):
    """The analytic gradient table must be the derivative of the shape functions."""
    step = 1e-6
    analytic = quadratic_shape_gradients_ref(L1, L2)
    for k in range(6):
        fd_L1 = (
            quadratic_shape_functions(L1 + step, L2)[k]
            - quadratic_shape_functions(L1 - step, L2)[k]
        ) / (2 * step)
        fd_L2 = (
            quadratic_shape_functions(L1, L2 + step)[k]
            - quadratic_shape_functions(L1, L2 - step)[k]
        ) / (2 * step)
        assert abs(analytic[k, 0] - fd_L1) < 1e-8
        assert abs(analytic[k, 1] - fd_L2) < 1e-8


def test_mass_matrix_of_a_flat_element_is_analytic():
    """``int_T N_k dS`` is (0, 0, 0, A/3, A/3, A/3) for a flat 6-node triangle."""
    nodes = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.0],
    ])
    from tensorbem.panel_quadratic import QuadraticTriangle

    elem = QuadraticTriangle(nodes)
    L, w = gauss_triangle_rule(8)
    mass = np.zeros(6)
    for q in range(len(w)):
        _, det = elem.jacobian(L[q, 0], L[q, 1])
        mass += w[q] * quadratic_shape_functions(L[q, 0], L[q, 1]) * det
    area = 0.5
    assert np.allclose(mass[:3], 0.0, atol=1e-12)
    assert np.allclose(mass[3:], area / 3.0, atol=1e-12)


# ---------------------------------------------------------------------------
# B. Duffy singular quadrature
# ---------------------------------------------------------------------------
def test_duffy_self_integral_matches_the_closed_form_on_a_flat_triangle():
    """Exact reference: ``int_T dS / |x - y|`` from the right-angle vertex.

    For the right isoceles triangle ``(0,0), (1,0), (0,1)`` and the target at the
    right angle, polar coordinates give ``2 ln(1 + sqrt(2)) = 1.7627472``.  With
    ``sum_k N_k = 1`` the six self integrals must sum to ``2 ln(1+sqrt2)/(4 pi)``.
    The two acute vertices are equal by the ``x <-> y`` symmetry and differ from
    the right-angle value, which is what pins the apex permutation: a shifted
    permutation reproduces the wrong one of the three values.
    """
    from tensorbem.panel_quadratic import QuadraticTriangle

    elem = QuadraticTriangle(np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.0],
    ]))
    exact = 2.0 * math.log(1.0 + math.sqrt(2.0)) / FOUR_PI
    totals = [float(elem.integrate_single_layer_self(apex, order=32).sum()) for apex in range(6)]
    assert abs(totals[0] - exact) < 1e-8
    assert abs(totals[1] - totals[2]) < 1e-10
    assert abs(totals[1] - totals[0]) > 1e-2
    assert all(t > 0.0 and math.isfinite(t) for t in totals)


def test_duffy_weights_cover_the_element():
    """One vertex patch or two midpoint patches tile the reference triangle."""
    for apex in range(6):
        _, weights = duffy_self_quadrature(apex, order=6)
        assert abs(weights.sum() - 0.5) < 1e-12


# ---------------------------------------------------------------------------
# C. assembled operator against the analytic sphere anchors
# ---------------------------------------------------------------------------
def test_double_layer_matrix_is_the_documented_zero_stub(coarse):
    """``D`` is returned as zeros; this is a registered stub, not an assembly."""
    _, _, D = coarse
    assert not D.any()


def test_operator_image_of_unit_density_is_the_shell_potential(coarse):
    """``S 1 = R``: potential of a unit-density shell of radius R = 1."""
    mesh, S, _ = coarse
    row = S @ np.ones(mesh.node_count)
    assert np.max(np.abs(row - 1.0)) < 5e-3


def test_operator_image_of_harmonics_matches_the_analytic_eigenvalues(coarse):
    """``S q = R q / (2n+1)`` for the trace of a degree-n harmonic polynomial."""
    mesh, S, _ = coarse
    for n, (name, fn) in study.HARMONICS.items():
        q = fn(mesh.vertices)
        image = S @ q
        exact = q / (2 * n + 1)
        assert np.linalg.norm(image - exact) / np.linalg.norm(q) < 5e-3, name


def test_patch_only_assembly_fails_the_analytic_anchor(coarse):
    """Mutation guard for defect C: dropping the off-element rows is detectable.

    Zeroing every row/column entry whose nodes share no element reproduces the
    pre-fix matrix; it must *fail* the analytic identity by a wide margin.
    """

    mesh, S, _ = coarse
    shares = np.zeros(S.shape, dtype=bool)
    for e in range(mesh.element_count):
        idx = mesh.faces[e]
        shares[np.ix_(idx, idx)] = True
    patch_only = np.where(shares, S, 0.0)
    q = mesh.vertices[:, 2]                      # n = 1 harmonic
    resid = np.linalg.norm(patch_only @ q - q / 3.0) / np.linalg.norm(q)
    assert resid > 0.1, "patch-only assembly should be grossly wrong"
    resid_full = np.linalg.norm(S @ q - q / 3.0) / np.linalg.norm(q)
    assert resid_full < 5e-3


def test_dirichlet_solve_recovers_the_analytic_density(coarse):
    """``g = z`` on the unit sphere has the exact density ``q = 3 z``."""
    mesh, S, _ = coarse
    g = mesh.vertices[:, 2]
    result = solve_laplace_dirichlet_quadratic(mesh, g, quadrature_order=8, self_order=16)
    q_exact = 3.0 * g
    rel = np.linalg.norm(result.nodal_values - q_exact) / np.linalg.norm(q_exact)
    assert rel < 5e-3
    assert result.relative_residual_l2 < 1e-10


def test_study_artifact_matches_a_fresh_run():
    """``docs/quadratic_panel_verification/study.json`` must be reproducible."""
    import json
    import pathlib

    path = pathlib.Path("docs/quadratic_panel_verification/study.json")
    if not path.exists():  # artifact lives in the repo root of the checkout
        pytest.skip("study.json artifact not present in this checkout")
    published = json.loads(path.read_text())
    fresh = study.run_study(max_sub=1)
    lv_pub = published["levels"][0]
    lv_new = fresh["levels"][0]
    for key in ("nodes", "elements"):
        assert lv_pub[key] == lv_new[key]
    assert abs(
        lv_pub["solves"]["n=1 (g=z)"]["density_rel_err"]
        - lv_new["solves"]["n=1 (g=z)"]["density_rel_err"]
    ) < 1e-12
    assert abs(lv_pub["area"] - lv_new["area"]) < 1e-12