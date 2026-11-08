"""Reading pins for ``docs/kkr_causality_band_study.md`` (m6404).

The m3664 study script that generated the document was never committed, so
its 119 number tokens were the largest unguarded paper-level contract in the
repository (``doc_claim_audit`` tier ``UNGUARDED``).  These tests close that
debt: every result table is recomputed from the current tree by
:mod:`tensorbem.benchmarks.kkr_band_doc_pins` and compared against the
document text.  A future edit that changes a printed digit turns this file
red; a future change to the transform or the Ursell reference that moves a
value does the same.

The floor grid (16 KKR checks on the analytic pair, ka up to 30) is the slow
part (~2 min); the study record is computed once at module scope and shared
by every test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorbem.benchmarks import kkr_band_doc_pins as pins

DOC = Path(__file__).resolve().parents[1] / "docs" / "kkr_causality_band_study.md"


@pytest.fixture(scope="module")
def study():
    return pins.study()


@pytest.fixture(scope="module")
def failures(study):
    return pins.check_document(DOC, study)


def test_document_exists_and_is_the_guarded_contract():
    assert DOC.name == pins.DOC_NAME
    assert DOC.is_file(), "the guarded document is missing from docs/"


def test_floor_grid_table_matches_document(study, failures):
    """§3.1: all 16 (band x samples) rows reproduce to the printed 4 decimals."""
    bad = [f for f in failures if f.startswith("§3.1")]
    assert not bad, "\n".join(bad)
    keys = [(r["ka_max"], r["points"]) for r in study["floor_grid"]["rows"]]
    assert keys == [(k, p) for k in pins.GRID_KA_MAX for p in pins.GRID_POINTS]


def test_high_frequency_limit_matches_document(study, failures):
    """§3.2: A_inf, its extrapolation residual, and the displaced-mass ratio."""
    bad = [f for f in failures if f.startswith("§3.2")]
    assert not bad, "\n".join(bad)
    limit = study["high_frequency_limit"]
    # physical sanity of the recomputed limit itself
    assert 0.8 < limit["added_mass_inf_over_displaced"] < 1.0
    assert limit["extrapolation_max_residual"] < 1.0  # kg/m, sub-kg fit residual


def test_panel_audit_table_matches_document(study, failures):
    """§3.3: KKR residuals and Ursell errors of analytic/32/64-panel solutions."""
    bad = [f for f in failures if f.startswith("§3.3")]
    assert not bad, "\n".join(bad)
    by_sol = {r["solution"]: r for r in study["panel_audit"]["rows"]}
    # 32 -> 64 panels halves the panel error (constant-panel discretisation)
    assert by_sol["panels_64"]["damping_error_vs_ursell"] < by_sol["panels_32"]["damping_error_vs_ursell"]
    assert by_sol["analytic_multipole"]["added_mass_error_vs_ursell"] == 0.0


def test_gain_sweep_table_matches_document(study, failures):
    """§3.4: the detection threshold ladder, including the clean-solution residual."""
    bad = [f for f in failures if f.startswith("§3.4")]
    assert not bad, "\n".join(bad)
    gains = study["gain_sweep"]
    # the clean (exactly causal) solution carries the floor, not zero
    assert 0.1 < gains["clean_residual"] < 0.4
    # and the +50 % gain must stand clearly above it
    assert gains["rows"][-1]["ratio_to_clean"] > 1.5


def test_verdict_rows_match(study, failures):
    """§6: the verdict rows that restate §3.1/§3.2/§3.3 numbers."""
    bad = [f for f in failures if f.startswith("§6")]
    assert not bad, "\n".join(bad)


def test_study_json_round_trip(study, tmp_path):
    """The full study record is JSON-serialisable (artifact contract)."""
    payload = json.dumps(study, sort_keys=True)
    assert json.loads(payload)["verdict"]["floor_default_band"] == pytest.approx(
        0.1904, abs=5.5e-5
    )
    out = tmp_path / "kkr_band_study.json"
    out.write_text(payload)
    assert json.loads(out.read_text())["floor_grid"]["ka_min"] == 0.05


def test_markdown_renderer_reproduces_document_rows(study):
    """The rendered tables must appear verbatim in the document (no drift)."""
    rendered = pins.format_markdown(study)
    document = DOC.read_text(encoding="utf-8")
    missing = [
        line
        for line in rendered.splitlines()
        if line.startswith("|") and line not in document
    ]
    assert not missing, "document table drifted from the code:\n" + "\n".join(missing)
