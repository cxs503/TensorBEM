"""Verification of the stored CASE-BEM-HULME-HEMISPHERE case-pool results.

This test reads the committed ``results.json`` from the case-pool output
directory and verifies:

1. The verdict is PASS.
2. All four gates (G1-G4) pass.
3. The worst relative error on the case mesh is below the 5% gate.
4. The reference truncation ladder shows geometric convergence.
5. Key numbers at ka=1.0 match the published Hulme (1982) table within tolerance.
6. The figure artefact exists.

This is a *lightweight* test: it does not re-run any BEM solves, only reads
and validates the stored artefacts.  It serves as a regression guard so that
any future change to the case-pool driver or the reference implementation
that breaks the stored evidence is caught immediately.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

CASEPOOL_DIR = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "verified"
    / "casepool_bem_hulme_hemisphere"
)
RESULTS = CASEPOOL_DIR / "results.json"
FIGURE = CASEPOOL_DIR / "figure.png"
GATE = 0.05  # 5% predeclared tolerance


@pytest.fixture(scope="module")
def results():
    assert RESULTS.is_file(), f"results.json missing: {RESULTS}"
    return json.loads(RESULTS.read_text())


@pytest.fixture(scope="module")
def rows(results):
    return results.get("rows", [])


# --- Verdict and gates ----------------------------------------------------

def test_verdict_is_pass(results):
    """The case-pool verdict must be PASS."""
    assert results["verdict"]["verdict"] == "PASS"


def test_all_gates_pass(results):
    """All four predeclared gates must pass."""
    criteria = results["verdict"]["criteria"]
    for name, crit in criteria.items():
        assert crit["pass"] is True, f"{name} failed: {crit}"


def test_worst_error_below_gate(rows):
    """Worst relative error on the case mesh (sub=4) must be < 5%."""
    quantities = (
        "A11_dimless", "B11_dimless", "A33_dimless",
        "B33_dimless", "Fx_dimless", "Fz_dimless",
    )
    case_rows = [r for r in rows if r["kind"] == "band" and r["subdivisions"] == 4]
    assert len(case_rows) == 6, f"Expected 6 ka values, got {len(case_rows)}"

    worst = max(
        abs(r[f"rel_error_{q}"]) for r in case_rows for q in quantities
    )
    assert worst < GATE, f"Worst error {worst:.4e} exceeds gate {GATE}"


def test_worst_error_at_ka2_is_b33(rows):
    """The worst error should be B33 at ka=2.0 (highest frequency, coarsest resolution)."""
    quantities = (
        "A11_dimless", "B11_dimless", "A33_dimless",
        "B33_dimless", "Fx_dimless", "Fz_dimless",
    )
    case_rows = [r for r in rows if r["kind"] == "band" and r["subdivisions"] == 4]
    ka2_row = next(r for r in case_rows if r["ka"] == 2.0)
    worst_q = max(quantities, key=lambda q: abs(ka2_row[f"rel_error_{q}"]))
    assert worst_q == "B33_dimless", f"Expected B33_dimless to be worst, got {worst_q}"
    assert abs(ka2_row["rel_error_B33_dimless"]) < 0.012  # ~1.15%


# --- Reference values at ka=1.0 (Hulme Table 1/2) -------------------------

def test_ka1_heave_added_mass(rows):
    """A33 at ka=1.0 matches Hulme Table 1 (0.4284) within 1%."""
    case_rows = [r for r in rows if r["kind"] == "band" and r["subdivisions"] == 4]
    ka1 = next(r for r in case_rows if r["ka"] == 1.0)
    ref = ka1["reference"]["A33_dimless"]
    assert abs(ref - 0.4284) < 0.001, f"A33 ref {ref} vs 0.4284"
    assert abs(ka1["A33_dimless"] - 0.4284) / 0.4284 < 0.01


def test_ka1_heave_damping(rows):
    """B33 at ka=1.0 matches Hulme Table 1 (0.2484) within 1%."""
    case_rows = [r for r in rows if r["kind"] == "band" and r["subdivisions"] == 4]
    ka1 = next(r for r in case_rows if r["ka"] == 1.0)
    ref = ka1["reference"]["B33_dimless"]
    assert abs(ref - 0.2484) < 0.001, f"B33 ref {ref} vs 0.2484"
    assert abs(ka1["B33_dimless"] - 0.2484) / 0.2484 < 0.01


def test_ka1_surge_added_mass(rows):
    """A11 at ka=1.0 matches Hulme Table 2 (0.5740) within 1%."""
    case_rows = [r for r in rows if r["kind"] == "band" and r["subdivisions"] == 4]
    ka1 = next(r for r in case_rows if r["ka"] == 1.0)
    ref = ka1["reference"]["A11_dimless"]
    assert abs(ref - 0.5740) < 0.001, f"A11 ref {ref} vs 0.5740"
    assert abs(ka1["A11_dimless"] - 0.5740) / 0.5740 < 0.01


def test_ka1_surge_damping(rows):
    """B11 at ka=1.0 matches Hulme Table 2 (0.3535) within 1%."""
    case_rows = [r for r in rows if r["kind"] == "band" and r["subdivisions"] == 4]
    ka1 = next(r for r in case_rows if r["ka"] == 1.0)
    ref = ka1["reference"]["B11_dimless"]
    assert abs(ref - 0.3535) < 0.001, f"B11 ref {ref} vs 0.3535"
    assert abs(ka1["B11_dimless"] - 0.3535) / 0.3535 < 0.01


# --- Reference truncation ladder ------------------------------------------

def test_reference_ladder_geometric_convergence(results):
    """The 96->144 term drift must be geometrically smaller than 48->96."""
    ladder = results["reference_provenance"]["truncation_ladder"]
    for ka_str, vals in ladder.items():
        d1 = vals["48->96"]
        d2 = vals["96->144"]
        assert d2 < d1, f"ka={ka_str}: 96->144 drift {d2} >= 48->96 drift {d1}"
        # tail ratio should be ~0.19 (geometric series)
        ratio = d2 / d1 if d1 > 0 else 0
        assert 0.15 < ratio < 0.25, f"ka={ka_str}: tail ratio {ratio} out of range"


def test_reference_drift_below_gate(results):
    """The worst 96->144 drift must be at least 20x below the 5% gate."""
    ladder = results["reference_provenance"]["truncation_ladder"]
    worst_drift = max(v["96->144"] for v in ladder.values())
    assert worst_drift * 20 < GATE, f"Worst drift {worst_drift:.2e} * 20 >= {GATE}"


# --- Geometry precondition ------------------------------------------------

def test_geometry_precondition(rows):
    """Mesh volume and waterplane area must be within 2% of exact."""
    case_rows = [r for r in rows if r["kind"] == "band" and r["subdivisions"] == 4]
    for r in case_rows:
        vol_err = abs(r["mesh_volume"] / r["nabla_exact"] - 1.0)
        wp_err = abs(r["mesh_waterplane_area"] / (math.pi * 1.0**2) - 1.0)
        assert vol_err < 0.02, f"Volume error {vol_err:.4e} at ka={r['ka']}"
        assert wp_err < 0.02, f"Waterplane error {wp_err:.4e} at ka={r['ka']}"


# --- Artefact existence ---------------------------------------------------

def test_figure_artifact_exists():
    """The comparison figure must exist."""
    assert FIGURE.is_file(), f"figure.png missing: {FIGURE}"
    assert FIGURE.stat().st_size > 10000, f"figure.png too small: {FIGURE.stat().st_size}"


def test_environment_artifact_exists():
    """The environment.json must exist."""
    env_path = CASEPOOL_DIR / "environment.json"
    assert env_path.is_file(), f"environment.json missing: {env_path}"
    env = json.loads(env_path.read_text())
    assert "python" in env
    assert "numpy" in env
    assert "case_code" in env
    assert env["case_code"] == "BEM-HULME-HEMISPHERE"


# --- Convergence: coarse vs case mesh -------------------------------------

def test_refinement_reduces_error(rows):
    """The case mesh (1024 panels) must have lower worst error than coarse (256)."""
    quantities = (
        "A11_dimless", "B11_dimless", "A33_dimless",
        "B33_dimless", "Fx_dimless", "Fz_dimless",
    )
    case_rows = [r for r in rows if r["kind"] == "band" and r["subdivisions"] == 4]
    coarse_rows = [r for r in rows if r["kind"] == "band" and r["subdivisions"] == 3]

    worst_case = max(abs(r[f"rel_error_{q}"]) for r in case_rows for q in quantities)
    worst_coarse = max(abs(r[f"rel_error_{q}"]) for r in coarse_rows for q in quantities)

    assert worst_case < worst_coarse, (
        f"Case mesh worst error {worst_case:.4e} >= coarse {worst_coarse:.4e}"
    )
