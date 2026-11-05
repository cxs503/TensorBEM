"""Pins the audit of the Wigley wave-resistance "pressure path" (mission m3100).

``docs/wigley_wave_resistance.md`` carried a 103-284 % two-path disagreement as
an honest FAIL attributed to the thin-ship approximation (B/L not small enough).
The measurements in ``docs/m3100/wigley_pressure_path_audit.md`` show that
attribution is wrong:

* for a fore-aft symmetric hull the integrand the shipped path sums,
  ``dphi_dx * n_x dS``, is *odd in x* (``dphi_dx`` even, ``n_x dS = -b_x dx dz``
  odd), so its exact value over the symmetric interval is zero -- this is the
  d'Alembert statement that the first-order pressure carries no resistance;
* the panelling cannot cancel it either, because the panel centroid abscissae
  ``-L/2 + h(3i+2)/3`` admit no mirror partner (``3(i+j)+4 = 3L/h`` has no
  integer solution), so a fraction ~1 of the residual is centroid aliasing that
  decays like ``h``.

These tests keep that statement honest and stop the documentation from drifting
back to the "thin-ship approximation" explanation.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from tensorbem.wigley import (
    michell_kochin_function,
    wave_resistance_kochin,
    wave_resistance_pressure,
    wigley_hull_mesh,
)

L, B, T = 2.0, 0.2, 0.125
RHO, G = 1025.0, 9.80665
DOC = pathlib.Path(__file__).resolve().parents[1] / "docs" / "wigley_wave_resistance.md"


def _dphi_dx(x: np.ndarray, z: np.ndarray, U: float, n_quad: int = 256) -> np.ndarray:
    """Wave-part x-velocity on the centreplane, exactly as the shipped path builds it."""
    k0 = G / U**2
    k_max = k0 * max(10.0, 20.0 / (k0 * T))
    nodes, weights = leggauss(n_quad)
    k = 0.5 * (k_max - k0) * (nodes + 1.0) + k0
    w = 0.5 * (k_max - k0) * weights
    H = np.array([michell_kochin_function(L, B, T, kk) for kk in k], dtype=complex)
    ph = (1j * k * H * np.exp(np.outer(z, k)) * np.exp(1j * np.outer(x, k))).real
    return (U / math.pi) * (ph * w).sum(axis=1)


def _near_field_exact_grid(U: float, n_x: int = 48, n_z: int = 24, n_quad: int = 256) -> float:
    """Rw = -rho U int int dphi_dx n_x dS on an x-symmetric trapezoid grid."""
    xs = np.linspace(-L / 2, L / 2, n_x)
    zs = np.linspace(-T, 0.0, n_z)
    w_x = np.full(n_x, L / (n_x - 1))
    w_x[[0, -1]] = 0.5 * L / (n_x - 1)
    w_z = np.full(n_z, T / (n_z - 1))
    w_z[[0, -1]] = 0.5 * T / (n_z - 1)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    px = _dphi_dx(X.ravel(), Z.ravel(), U, n_quad).reshape(X.shape)
    b_x = -8.0 * B * X / L**3 * (1.0 - (Z / T) ** 2)
    return 2.0 * RHO * U * float((w_x[:, None] * w_z[None, :] * px * b_x).sum())


@pytest.mark.parametrize("froude", [0.25, 0.30, 0.35])
def test_exact_pressure_integral_vanishes(froude: float) -> None:
    """The integrand below the shipped sum integrates to zero by x-parity."""
    U = froude * math.sqrt(G * L)
    rw_k = wave_resistance_kochin(L, B, T, U, RHO, G)
    rel = abs(_near_field_exact_grid(U)) / rw_k
    assert rel < 1e-10, f"Fr={froude}: exact near-field integral is {rel:.2e} of Rw_Kochin"


def test_wave_part_velocity_is_even_in_x() -> None:
    """dphi_dx(x) == dphi_dx(-x): the parity that kills the first-order pressure."""
    U = 0.30 * math.sqrt(G * L)
    x = np.array([0.0, 0.17, 0.43, 0.71, 0.96])
    z = np.array([-0.6 * T] * x.size)
    fwd = _dphi_dx(x, z, U)
    bwd = _dphi_dx(-x, z, U)
    assert np.allclose(fwd, bwd, rtol=1e-10, atol=1e-14 * np.max(np.abs(fwd)))


def test_hull_panelling_has_no_mirror_partners() -> None:
    """Explains why the odd integrand cannot cancel pairwise on the triangulation."""
    for n_x, n_z in ((8, 4), (16, 8)):
        c = wigley_hull_mesh(L, B, T, n_x, n_z).centroids
        tol = 1e-9 * L
        paired = 0
        for i in range(c.shape[0]):
            d = np.hypot(c[:, 0] + c[i, 0], c[:, 2] - c[i, 2])
            if np.any(d < tol):
                paired += 1
        # 3(i+j)+4 = 3 L/h has no integer solution -> no exact mirror partners
        assert paired == 0, f"{n_x}x{n_z}: {paired} panels unexpectedly have a mirror partner"


def test_shipped_pressure_path_decays_to_zero_with_refinement() -> None:
    """A convergent discretisation tends to the integral; this one tends to 0."""
    U = 0.30 * math.sqrt(G * L)
    rw_k = wave_resistance_kochin(L, B, T, U, RHO, G)
    rels = []
    for n_x, n_z in ((8, 4), (16, 8), (32, 16)):
        rw_p = wave_resistance_pressure(wigley_hull_mesh(L, B, T, n_x, n_z), L, B, T, U, RHO, G)
        assert rw_p < 0.0, "the artefact has a negative sign; a real wave resistance cannot"
        rels.append(abs(rw_p) / rw_k)
    assert rels[0] > rels[1] > rels[2], f"residual not monotone: {rels}"
    assert rels[2] < 0.15, f"residual at 32x16 is {rels[2]:.3f} of Rw_Kochin"
    assert rels[0] < 0.6, f"residual at 8x4 is {rels[0]:.3f} of Rw_Kochin"


@pytest.mark.parametrize("b_over_l", [0.10, 0.0125])
def test_thin_ship_limit_does_not_recover_kochin(b_over_l: float) -> None:
    """If the gap were a finite-B/L effect, the ratio would approach 1 as B/L -> 0."""
    U = 0.30 * math.sqrt(G * L)
    b = b_over_l * L
    t = b / 1.6
    rw_k = wave_resistance_kochin(L, b, t, U, RHO, G)
    rw_p = wave_resistance_pressure(wigley_hull_mesh(L, b, t, 16, 8), L, b, t, U, RHO, G)
    ratio = rw_p / rw_k
    assert ratio < 0.0 or abs(ratio) > 1.0, (
        f"B/L={b_over_l}: ratio {ratio:+.3f} sits near the thin-ship target, "
        "which would contradict the audit"
    )


def test_documentation_no_longer_blames_the_thin_ship_approximation() -> None:
    """Doc gate: the honest attribution must be the measured one."""
    text = DOC.read_text()
    assert "centroid" in text.lower() or "parity" in text.lower(), (
        "docs/wigley_wave_resistance.md must record the parity/centroid attribution"
    )
    assert "no resistance at first order" in text or "identically zero" in text
    # the old, measured-false explanation must be gone from the two-path section
    two_path = text[text.index("## Two-Path Comparison"):]
    assert "both paths **do not agree within 5%** because" not in two_path