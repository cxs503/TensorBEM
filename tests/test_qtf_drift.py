"""Pins for the second-order difference-frequency QTF (:mod:`tensorbem.qtf_drift`).

The chain of trust:

* the free-surface closure of the OC4-class hull is pinned against the closed
  form of *the same facetted* geometry (machine precision) and against the
  published DeepCwind displacement (3 %),
* the Newman matrix is pinned as an exact interpolant of the diagonal,
* the direct QTF's diagonal is pinned against
  :func:`tensorbem.drift.near_field_mean_drift_3d` on a real diffraction solve
  -- the diagonal of the QTF *is* the verified mean-drift assembly, so this is a
  regression guard on the prefactors of both terms,
* the pair matrix is pinned by its hermitian symmetry and by the Parseval
  identity ``Int S_F dmu = variance``,
* the arbitration metrics are pinned on a synthetic QTF whose off-diagonal
  behaviour is known exactly (zero-bandwidth limit and bandwidth growth).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.benchmarks.qtf_drift_study import (
    DOC_BEGIN,
    DOC_END,
    band_subset,
    jonswap_on_grid,
    render_doc_block,
)
from tensorbem.drift import bottom_mounted_cylinder_mesh
from tensorbem.free_surface_solver3d import solve_diffraction_3d
from tensorbem.qtf_drift import (
    DriftQTF,
    SEMISUB_PUBLISHED_DISPLACEMENT,
    diagonal_drift_coefficients,
    four_column_semisub_mesh,
    newman_bandwidth_study,
    newman_qtf,
    qtf_from_diagonal,
    quadratic_qtf,
    sea_state_drift_force,
    semisub_analytic_displacement,
    semisub_mesh_displacement,
    slow_drift_response_spectrum,
    waterline_elevations,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_JSON = REPO_ROOT / "docs" / "qtf_drift_oc4" / "study.json"
REPORT = REPO_ROOT / "docs" / "qtf_drift_oc4.md"

SMALL_MESH = dict(
    panels_per_column=8, panels_vertical=2, pontoon_theta=6, pontoon_axial=2
)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def test_mesh_volume_matches_facetted_closed_form() -> None:
    """The mesh volume equals the analytic volume of the *same* facetted hull."""
    spec = SMALL_MESH
    mesh = four_column_semisub_mesh(**spec)
    analytic = semisub_analytic_displacement(
        n_theta=spec["panels_per_column"], pontoon_theta=spec["pontoon_theta"]
    )
    volume = semisub_mesh_displacement(mesh)
    assert volume > 0.0
    assert volume == pytest.approx(analytic["displacement"], rel=1.0e-12)


def test_smooth_geometry_reproduces_published_displacement() -> None:
    """Smooth dimensions reproduce the DeepCwind 13 473 m^3 displacement."""
    smooth = semisub_analytic_displacement()
    ratio = smooth["displacement"] / SEMISUB_PUBLISHED_DISPLACEMENT
    assert 0.95 < ratio < 1.05
    # waterplane area of a 6.5 m central plus three 12 m columns
    assert smooth["waterplane_area"] == pytest.approx(
        math.pi * 3.25**2 + 3.0 * math.pi * 6.0**2, rel=1.0e-12
    )
    assert 350.0 < smooth["waterplane_area"] < 400.0


def test_semisub_mesh_has_four_waterline_loops() -> None:
    """Four surface-piercing columns -> four waterline loops of n_theta edges."""
    spec = SMALL_MESH
    mesh = four_column_semisub_mesh(**spec)
    wl = waterline_elevations(
        mesh, np.ones(mesh.panel_count, dtype=np.complex128), wavenumber=0.05
    )
    assert wl["count"] == 4 * spec["panels_per_column"]
    # all wetted panels stay at or below the calm waterplane
    assert float(np.max(np.asarray(mesh.vertices)[:, 2])) == 0.0
    assert float(np.min(np.asarray(mesh.vertices)[:, 2])) < -10.0


# ---------------------------------------------------------------------------
# Newman matrix
# ---------------------------------------------------------------------------
def test_newman_is_exact_on_the_diagonal_and_symmetric() -> None:
    diag = np.array([[1.0, -0.5], [4.0, 2.0], [-2.5, 0.25]])
    matrix = newman_qtf(diag)
    assert matrix.shape == (3, 3, 2)
    for i in range(3):
        assert np.allclose(matrix[i, i], diag[i])
    assert np.allclose(matrix, np.transpose(matrix, (1, 0, 2)))


def _synthetic_qtf(n: int = 7, spread: float = 1.2) -> DriftQTF:
    """QTF whose off-diagonal decays away from the diagonal by construction."""
    omegas = np.linspace(0.5, 0.9, n)
    diag = np.linspace(2.0e4, 6.0e4, n)  # monotone mean drift
    matrix = np.zeros((n, n, 2), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            delta = abs(omegas[i] - omegas[j])
            weight = math.exp(-spread * delta)
            value = weight * 0.5 * (diag[i] + diag[j]) + (1.0 - weight) * min(diag[i], diag[j])
            matrix[i, j, 0] = value
            matrix[i, j, 1] = 0.2 * value
    return DriftQTF(
        omegas=omegas,
        wavenumbers=omegas**2 / 9.81,
        matrix=matrix,
        diagonal=np.real(matrix[np.arange(n), np.arange(n)]),
        route="direct",
    )


def test_newman_reproduces_the_regular_wave_mean_drift() -> None:
    """Both routes give A^2 T_ii for a single component: Newman is exact there."""
    qtf = _synthetic_qtf(5)
    newman = qtf_from_diagonal(qtf.diagonal, qtf.omegas)
    amp = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    direct = sea_state_drift_force(qtf, amp)
    approx = sea_state_drift_force(newman, amp)
    expected = qtf.diagonal[0]
    assert np.allclose(direct["mean_force"], expected)
    assert np.allclose(approx["mean_force"], expected)
    assert direct["force_rms"] == pytest.approx(0.0, abs=1.0e-9)


def test_newman_error_vanishes_for_zero_bandwidth() -> None:
    """A regular wave has no off-diagonal pairs: the error must be exactly zero."""
    qtf = _synthetic_qtf(6)
    sub = _sub(qtf, np.array([2, 3]))
    rows = newman_bandwidth_study(sub, [{"amplitude": np.array([1.0, 0.0]), "spectral_width": 0.0, "relative_bandwidth": 0.0, "hs": 1.0, "tp": 10.0}])
    assert rows[0]["rms_relative_error"] == pytest.approx(0.0, abs=1.0e-12)


def test_newman_error_grows_with_bandwidth() -> None:
    """The arbitration metric must respond to the spectral width, monotonically."""
    qtf = _synthetic_qtf(9)
    widths = (0.05, 0.10, 0.20)
    rows = []
    for offset, width in enumerate(widths):
        idx = np.arange(3 - offset, 6 + offset)
        sub = _sub(qtf, idx)
        amp = np.asarray(jonswap_on_grid(sub.omegas, 5.0, 12.0)["amplitude"])
        rows.append(
            newman_bandwidth_study(
                sub,
                [{"amplitude": amp, "spectral_width": width, "relative_bandwidth": 2 * width, "hs": 5.0, "tp": 12.0}],
            )[0]
        )
    errors = [r["rms_relative_error"] for r in rows]
    assert errors[0] < errors[1] < errors[2]
    assert errors[0] > 0.0
    assert all(r["mean_relative_error"] == pytest.approx(0.0, abs=1.0e-12) for r in rows)


def _sub(qtf: DriftQTF, idx) -> DriftQTF:
    idx = np.asarray(idx, dtype=int)
    matrix = np.asarray(qtf.matrix)[np.ix_(idx, idx)]
    return DriftQTF(
        omegas=np.asarray(qtf.omegas)[idx],
        wavenumbers=np.asarray(qtf.wavenumbers)[idx],
        matrix=matrix,
        diagonal=np.real(matrix[np.arange(idx.size), np.arange(idx.size)]),
    )


# ---------------------------------------------------------------------------
# spectra
# ---------------------------------------------------------------------------
def test_spectrum_integral_equals_the_force_variance() -> None:
    """Parseval: the binned one-sided spectrum must carry the whole variance."""
    qtf = _synthetic_qtf(7)
    amp = np.linspace(0.4, 1.2, 7)
    result = sea_state_drift_force(qtf, amp)
    assert result["spectrum_integral"] == pytest.approx(result["force_variance"], rel=1.0e-12)
    assert result["force_rms"] == pytest.approx(math.sqrt(result["force_variance"]), rel=1.0e-12)
    assert result["force_rms"] > 0.0


def test_jonswap_grid_carries_the_requested_height() -> None:
    omegas = np.linspace(0.3, 1.2, 40)
    sea = jonswap_on_grid(omegas, 8.0, 11.0)
    m0 = float(np.sum(0.5 * sea["amplitude"] ** 2))
    assert 4.0 * math.sqrt(m0) == pytest.approx(8.0, rel=0.05)
    band = band_subset(sea, 11.0, 0.2)
    assert band["component_count"] >= 3
    assert band["hs_discrete"] < 8.0  # a truncation can only lose energy
    assert band["spectral_width"] > 0.0


def test_band_subset_narrows_monotonically() -> None:
    omegas = np.linspace(0.3, 1.2, 40)
    sea = jonswap_on_grid(omegas, 8.0, 11.0)
    widths = [band_subset(sea, 11.0, half)["spectral_width"] for half in (0.1, 0.2, 0.4)]
    assert widths[0] < widths[1] < widths[2]


# ---------------------------------------------------------------------------
# slow drift
# ---------------------------------------------------------------------------
def test_slow_drift_response_limits() -> None:
    mu = np.array([0.0, 0.05, 0.1, 0.2])
    spectrum = {"mu": mu, "spectrum": np.ones_like(mu), "bin_width": 0.05}
    result = slow_drift_response_spectrum(
        spectrum, mass=1.0e7, added_mass=5.0e6, damping=5.0e5, stiffness=1.0e5
    )
    # static limit: unit force -> 1/K of displacement
    assert result["rao"][0] == pytest.approx(1.0 / 1.0e5, rel=1.0e-12)
    assert result["natural_frequency"] == pytest.approx(
        math.sqrt(1.0e5 / 1.5e7), rel=1.0e-12
    )
    peak = float(result["mu"][int(np.argmax(result["rao"]))])
    assert peak <= result["natural_frequency"] * 1.5
    assert result["std_motion"] > 0.0


# ---------------------------------------------------------------------------
# real diffraction solve: the diagonal identity
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cylinder_case():
    mesh = bottom_mounted_cylinder_mesh(5.0, 10.0, n_theta=12, n_z=4)
    wavenumbers = (0.20, 0.24)
    results = [solve_diffraction_3d(mesh, k) for k in wavenumbers]
    qtf = quadratic_qtf(mesh, wavenumbers, diffraction_results=results)
    return mesh, wavenumbers, results, qtf


def test_qft_diagonal_equals_the_verified_mean_drift(cylinder_case) -> None:
    """The QTF diagonal *is* the Pinkster near-field assembly, to 1e-8."""
    mesh, _, results, qtf = cylinder_case
    reference = diagonal_drift_coefficients(mesh, results)[:, :2]
    scale = np.max(np.abs(reference))
    residual = np.max(np.abs(np.asarray(qtf.diagonal) - reference)) / scale
    assert residual < 1.0e-8


def test_qft_matrix_is_hermitian(cylinder_case) -> None:
    """T(w_i, w_j) = conj T(w_j, w_i) follows from the quadratic product."""
    _, _, _, qtf = cylinder_case
    matrix = np.asarray(qtf.matrix)
    residual = np.max(np.abs(matrix - np.conj(np.transpose(matrix, (1, 0, 2)))))
    assert residual / np.max(np.abs(matrix)) < 1.0e-10


def test_qft_off_diagonal_is_not_negligible(cylinder_case) -> None:
    """Guard against a silently zeroed off-diagonal block."""
    _, wavenumbers, _, qtf = cylinder_case
    matrix = np.asarray(qtf.matrix)
    off = matrix[0, 1, 0]
    assert abs(off) > 0.05 * abs(matrix[0, 0, 0])
    assert wavenumbers[0] < wavenumbers[1]


def test_qft_is_smooth_across_the_diagonal(cylinder_case) -> None:
    """Off-diagonal smoothness: the pair QTF must interpolate the diagonals.

    This is the physical requirement Newman's approximation rests on.  Two
    neighbouring wavenumbers (ka = 1.0 and 1.2 on the cylinder) must give an
    off-diagonal that sits close to the arithmetic mean of the two diagonals; a
    QTF that fails this is noise, not physics.
    """
    _, _, _, qtf = cylinder_case
    matrix = np.asarray(qtf.matrix)
    t00 = float(np.real(matrix[0, 0, 0]))
    t11 = float(np.real(matrix[1, 1, 0]))
    t01 = float(np.real(matrix[0, 1, 0]))
    mean = 0.5 * (t00 + t11)
    assert abs(t01 - mean) / abs(mean) < 0.03
    # the quadrature part must stay small compared with the in-phase part
    assert abs(float(np.imag(matrix[0, 1, 0]))) < 0.05 * abs(mean)


# ---------------------------------------------------------------------------
# committed artifacts stay consistent with the code
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not STUDY_JSON.exists(), reason="study artifact not generated")
def test_committed_report_block_matches_artifact() -> None:
    study = json.loads(STUDY_JSON.read_text())
    block = render_doc_block(study)
    text = REPORT.read_text()
    assert DOC_BEGIN in text and DOC_END in text
    committed = text[text.index(DOC_BEGIN) : text.index(DOC_END) + len(DOC_END)]
    assert committed.strip() == block.strip()
