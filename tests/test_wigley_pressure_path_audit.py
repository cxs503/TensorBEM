"""Executable form of the Wigley pressure-path root-cause audit.

The audit settles an open *honest FAIL* (``docs/wigley_wave_resistance.md``): the
legacy pressure path disagrees with the verified Kochin path by 103-284 % and is
negative everywhere, and the document used to attribute that to the thin-ship
approximation.  These tests pin the four facts that make the attribution
impossible and the diagnosis unambiguous:

1. the legacy result is regulated by an arbitrary wavenumber cut-off, so it has
   no limit (``test_pressure_path_is_cutoff_dependent``);
2. the sampled integral *is* converged in quadrature, so the defect is the
   integrand, not quadrature error (``test_quadrature_is_converged``);
3. no sign change and no single constant factor can map one path onto the other
   (``test_no_constant_factor_maps_the_paths``);
4. the field the legacy integrand builds violates the steady free-surface
   condition by O(1) at *every* quadrature node, because a steady free wave in
   this theory lives on ``k = k0`` in two dimensions and on the Kelvin wake
   curve with its ``1/sqrt(k0^2 k^4 - k^2)`` Jacobian in three
   (``test_legacy_field_violates_the_free_surface_condition``).

They also keep the committed artefact reproducible: ``docs/wigley_pressure_audit/
summary.md`` must equal ``render_markdown(json.load(results.json))`` byte for
byte, so no number in the manuscript is transcribed by hand.

Kept deliberately cheap: one 512-panel and one 8192-panel mesh, no dense solves.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.benchmarks import wigley_pressure_audit as audit
from tensorbem.wigley import (
    michell_wave_resistance,
    wave_resistance_pressure,
    wigley_hull_mesh,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = REPO_ROOT / "docs" / "wigley_pressure_audit" / "results.json"
SUMMARY_PATH = REPO_ROOT / "docs" / "wigley_pressure_audit" / "summary.md"
FIGURE_PATH = REPO_ROOT / "docs" / "figures" / "wigley_pressure_audit.png"

L, B, T = audit.DEFAULT_L, audit.DEFAULT_B, audit.DEFAULT_T
RHO, G = audit.DEFAULT_RHO, audit.DEFAULT_G


@pytest.fixture(scope="module")
def committed() -> dict:
    if not RESULT_PATH.exists():  # pragma: no cover - artefact is committed
        pytest.skip("run python -m tensorbem.benchmarks.wigley_pressure_audit first")
    return json.loads(RESULT_PATH.read_text())


def test_case_is_the_documented_wigley_hull():
    props = audit.michell_wave_resistance(L, B, T, 0.30 * math.sqrt(G * L), RHO, G)
    assert props > 0.0
    assert audit.DEFAULT_L / audit.DEFAULT_B == pytest.approx(10.0)


def test_pressure_path_is_cutoff_dependent():
    """A physical near-field quantity cannot be set by an arbitrary k_max."""
    sweep = audit.cutoff_sweep(factors=(10.0, 80.0))
    values = [row["Rw_pressure"] for row in sweep["cutoff"]]
    assert all(value < 0.0 for value in values)
    spread = abs(values[1] - values[0]) / abs(sweep["Rw_kochin"])
    assert spread > 0.05
    # and it does not settle as the cut-off is raised further
    wide = audit.cutoff_sweep(factors=(80.0, 160.0))
    wide_values = [row["Rw_pressure"] for row in wide["cutoff"]]
    assert abs(wide_values[1] - wide_values[0]) / abs(wide["Rw_kochin"]) > 1e-2


def test_quadrature_is_converged():
    """The legacy result is a converged evaluation of the wrong integrand."""
    sweep = audit.node_sweep(counts=(128, 256, 512))
    values = [row["Rw_pressure"] for row in sweep["nodes"]]
    for value in values[1:]:
        assert value == pytest.approx(values[0], rel=1e-6)


def test_no_constant_factor_maps_the_paths():
    sweep = audit.froude_sweep()
    ratios = [row["ratio_to_kochin"] for row in sweep["rows"]]
    assert max(ratios) < 0.0  # all negative: no sign flip can repair it
    assert max(ratios) - min(ratios) > 0.1  # and no constant factor either
    assert sweep["min_relative_difference"] > 1.0


def test_mesh_refinement_drives_the_pressure_path_to_zero():
    sweep = audit.mesh_sweep(levels=((8, 4), (64, 32)))
    first, last = sweep["levels"][0], sweep["levels"][-1]
    assert abs(last["Rw_pressure"]) < 0.2 * abs(first["Rw_pressure"])
    assert last["relative_difference"] > 0.9  # converges to zero, not to Kochin


def test_legacy_field_violates_the_free_surface_condition():
    """The O(1) residual is the root cause, and it is analytic."""
    k0 = audit._k0(G, 0.30 * math.sqrt(G * L))
    # two-dimensional steady dispersion: the only admissible wavenumber is k0
    assert audit.eigval_standard(k0, k0) == pytest.approx(0.0, abs=1e-12)
    for factor in (2.0, 5.0, 10.0):
        assert abs(audit.eigval_standard(factor * k0, k0)) > 0.4 * factor * k0
    # the operator as written in the legacy docstring admits no real wavenumber
    for factor in (0.5, 1.0, 2.0, 10.0):
        assert audit.eigval_documented(factor * k0, k0) > 0.0

    residual = audit.fs_residual(froude=0.30)
    assert residual["fs_residual_standard"] > 0.5
    assert residual["fs_residual_documented"] == pytest.approx(1.0, abs=1e-9)
    # Gauss-Legendre nodes never touch the grid endpoints, and the dispersion
    # curve is a single point, so no node can be a free wave
    assert residual["nodes_on_dispersion_curve"] == 0


def test_dispersion_profile_is_zero_only_at_lambda_one():
    profile = audit.dispersion_mismatch_profile()
    lam = np.asarray(profile["lambda"])
    res = np.asarray(profile["relative_residual"])
    assert res[0] == pytest.approx(0.0, abs=1e-12)
    assert lam[0] == pytest.approx(1.0)
    assert np.all(np.diff(res) > 0.0)
    assert res[-1] > 0.8


def test_kochin_path_is_independent_of_any_cutoff_and_node_count():
    U = 0.30 * math.sqrt(G * L)
    base = michell_wave_resistance(L, B, T, U, RHO, G)
    for n_quad in (192, 384, 768):
        assert michell_wave_resistance(L, B, T, U, RHO, G, n_quad=n_quad) == pytest.approx(
            base, rel=1e-6
        )


def test_k_max_factor_is_validated_and_legacy_default_reproduced():
    mesh = wigley_hull_mesh(L, B, T, 8, 4)
    U = 0.30 * math.sqrt(G * L)
    with pytest.raises(ValueError):
        wave_resistance_pressure(mesh, L, B, T, U, RHO, G, k_max_factor=1.0)
    k0 = audit._k0(G, U)
    legacy_factor = audit._legacy_k_max_factor(k0, T)
    default = wave_resistance_pressure(mesh, L, B, T, U, RHO, G)
    explicit = wave_resistance_pressure(
        mesh, L, B, T, U, RHO, G, k_max_factor=legacy_factor
    )
    assert default == pytest.approx(explicit, rel=1e-15, abs=0.0)


def test_committed_verdict_is_the_diagnosis(committed):
    verdict = committed["verdict"]
    assert verdict["pressure_path_cutoff_dependent"] is True
    assert verdict["pressure_path_mesh_nonconvergent"] is True
    assert verdict["pressure_path_negative_everywhere"] is True
    assert verdict["no_constant_factor"] is True
    assert verdict["field_off_dispersion_curve"] is True
    assert verdict["quadrature_nodes_on_dispersion_curve"] == 0
    assert verdict["is_formulation_error_not_thin_ship_limit"] is True


def test_committed_tables_match_the_committed_numbers(committed):
    """Numbers in the committed JSON are the ones the audit recomputes."""
    cutoff = committed["cutoff_sweep"]
    recomputed = audit.cutoff_sweep(factors=tuple(
        row["k_max_factor"] for row in cutoff["cutoff"]
    ))
    for stored, fresh in zip(cutoff["cutoff"], recomputed["cutoff"]):
        assert stored["Rw_pressure"] == pytest.approx(fresh["Rw_pressure"], rel=1e-12)
    assert committed["froude_sweep"]["all_negative"] is True
    assert committed["froude_sweep"]["ratio_spread"] == pytest.approx(
        committed["froude_sweep"]["ratio_max"] - committed["froude_sweep"]["ratio_min"],
        rel=1e-12,
    )


def test_summary_markdown_is_regenerated_from_the_json(committed):
    assert SUMMARY_PATH.exists()
    assert SUMMARY_PATH.read_text() == audit.render_markdown(committed)


def test_figure_is_committed():
    assert FIGURE_PATH.exists()
    assert FIGURE_PATH.stat().st_size > 20_000