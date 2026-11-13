"""Case CASE-BEM-NEUMANN-KELVIN: steady NK far-field vs Michell thin-ship Cw(Fr).

Regression pins for the far-field (Havelock/Kochin) operator of
:mod:`tensorbem.nk_panel3d`, mission 17246.

The defect these tests exist for: until m17246 ``wave_resistance_far`` carried a
*dimensionless* default prefactor ``0.5`` although the exact steady-Kelvin value
is ``kappa/(2 pi g) = 1/(2 pi U^2)``.  The missing ``1/U^2`` made the "implied
prefactor" of the thin-ship reduction drift like ``Fr^-2`` (the 4.8x drift
reported in ``docs/m10003/REPORT.md``) instead of being Froude independent.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem import nk_panel3d as nk
from tensorbem import wigley

L, B, T = 10.0, 1.0, 0.625
RHO, G = 1025.0, 9.80665
S_REF = L * T


def _speed(fr: float) -> float:
    return fr * math.sqrt(G * L)


def _thin_ship_far(fr: float, nx: int = 60, nz: int = 30, n_theta: int = 2001):
    mesh = nk.centreplane_panels(L, B, T, nx, nz)
    x = mesh.centroids[:, 0]
    z = mesh.centroids[:, 2]
    b_x = 0.5 * B * (-8.0 * x / L**2) * (1.0 - (z / T) ** 2)
    sigma = -2.0 * _speed(fr) * b_x
    kappa = G / _speed(fr) ** 2
    rw, _, _ = nk.wave_resistance_far(
        sigma, mesh, kappa, rho=RHO, g=G, n_theta=n_theta, mirror=False
    )
    return rw


def test_kochin_prefactor_is_kappa_over_2pi_g():
    """Exact constant ``kappa/(2 pi g) = 1/(2 pi U^2)``; it carries 1/U^2."""
    kappa = 0.625
    got = nk.kochin_prefactor(kappa, G)
    assert got == pytest.approx(kappa / (2.0 * math.pi * G), rel=0.0)
    u = math.sqrt(G / kappa)
    assert got == pytest.approx(1.0 / (2.0 * math.pi * u * u), rel=1e-12)
    with pytest.raises(ValueError):
        nk.kochin_prefactor(0.0, G)


def test_far_field_operator_reproduces_michell_thin_ship():
    """NK steady far field applied to the thin-ship sheet == Michell (ref), <2%."""
    fr = 0.30
    u = _speed(fr)
    ref = wigley.michell_wave_resistance_canonical(L, B, T, u, RHO, G)
    got = _thin_ship_far(fr)
    assert abs(got - ref) / ref < 0.02


def test_far_field_prefactor_has_no_froude_drift():
    """The got/ref ratio must not drift with Froude number (the m10003 defect)."""
    ratios = []
    for fr in (0.20, 0.40):
        u = _speed(fr)
        ref = wigley.michell_wave_resistance_canonical(L, B, T, u, RHO, G)
        # 120 x 60 centreplane panels: the panel quadrature error at Fr=0.20 is
        # ~0.5 %, so a 2 % drift threshold isolates the prefactor, not the mesh.
        ratios.append(_thin_ship_far(fr, nx=120, nz=60, n_theta=1001) / ref)
    drift = abs(ratios[1] - ratios[0]) / ratios[0]
    assert drift < 0.02, f"ratio drift {drift:.3%} between Fr=0.20 and Fr=0.40"


def test_thin_ship_reference_reports_michell_agreement():
    fr = 0.30
    ts = nk.thin_ship_reference(L, B, T, _speed(fr), nx=60, nz=30)
    assert ts["s_ref"] == pytest.approx(S_REF)
    assert ts["rw_michell"] == pytest.approx(
        wigley.michell_wave_resistance_canonical(L, B, T, _speed(fr), RHO, G)
    )
    assert ts["rel_error_far_vs_michell"] < 0.02
    assert ts["cw_michell"] == pytest.approx(ts["rw_michell"] / (0.5 * RHO * _speed(fr) ** 2 * S_REF))


def test_far_field_is_quadratic_in_source_strength():
    """Rw is quadratic in sigma -> the operator is a genuine energy functional."""
    mesh = nk.centreplane_panels(L, B, T, 30, 15)
    rng = np.random.default_rng(7)
    sigma = rng.normal(size=mesh.n_panels)
    kappa = 0.625
    a = nk.wave_resistance_far(sigma, mesh, kappa, n_theta=501, mirror=False)[0]
    b = nk.wave_resistance_far(2.0 * sigma, mesh, kappa, n_theta=501, mirror=False)[0]
    assert b / a == pytest.approx(4.0, rel=1e-10)