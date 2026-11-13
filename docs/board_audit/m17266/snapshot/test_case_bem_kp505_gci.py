"""Pins for case ``CASE-BEM-KP505-GCI`` (KP505 three-density GCI, r = 2).

The pins are deliberately cheap: the three-density ladder itself (D3 = 256
bound panels, 96 wake steps, ~50 s and ~6.4 GB) is *not* re-solved here.  What
is pinned is everything the verdict depends on that can be checked in seconds:

* the GCI procedure itself (manufactured sequence with an exact order and an
  exact limit, library vs inline closed form, oscillatory-triplet gating),
* the artefact's internal consistency (every ``gci_fine_pct`` recomputes from
  the stored three levels; the ``< 3 %`` flags match the numbers),
* one real lattice level (D1 at the design advance) against its recorded value,
  so a silent change in the solver or in the case configuration turns red.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorbem.benchmarks import kp505_three_density_gci as case
from tensorbem.uq import grid_convergence_index

CASE_JSON = Path(case.DEFAULT_OUTDIR) / "case.json"


# ---------------------------------------------------------------------------
# reference procedure (G7)
# ---------------------------------------------------------------------------
def test_reference_procedure_verifies_to_roundoff():
    v = case.verify_reference_procedure()
    assert v["all_pass"], [c for c in v["checks"] if not c["pass"]]
    assert v["safety_factor"] == 1.25          # Roache, three-grid study
    assert v["refinement_ratio"] == 2.0


def test_closed_form_matches_library_on_case_numbers():
    closed = case.roache_gci_closed_form(0.158243, 0.170180, 0.170944)
    lib = grid_convergence_index(0.158243, 0.170180, 0.170944,
                                 refinement_ratio=2.0)
    assert closed["observed_order"] == pytest.approx(lib.observed_order, rel=1e-12)
    assert closed["gci_fine"] == pytest.approx(lib.gci_fine, rel=1e-12)
    assert closed["gci_medium"] == pytest.approx(lib.gci_medium, rel=1e-12)


def test_anti_convergent_triplet_gives_negative_order_and_is_not_admitted():
    # fine/medium/coarse of K_T at J = 0.70: the coarse pair is *closer* than
    # the fine pair, i.e. the increments grow with refinement -> p < 0.
    row = case._gci(0.158243, 0.170180, 0.170944, "KT")
    assert row["observed_order"] < 0.0
    assert row["monotone"] is True           # same sign, yet not converging
    assert row["asymptotic_admitted"] is False
    assert row["under_case_tol_3pct"] is False


def test_oscillatory_triplet_is_flagged_and_never_admitted():
    row = case._gci(1.0, 1.1, 1.05, "KT")
    assert row["oscillatory"] is True
    assert row["monotone"] is False
    assert row["asymptotic_admitted"] is False
    assert math.isfinite(row["gci_fine_pct"])


def test_well_behaved_manufactured_triplet_is_admitted():
    A, C, p = 0.5, 3.0, 1.7
    f = {h: A + C * h ** p for h in (0.01, 0.02, 0.04)}
    row = case._gci(f[0.01], f[0.02], f[0.04], "KT")
    assert row["observed_order"] == pytest.approx(p, rel=1e-10)
    assert row["gci_fine_pct"] < 3.0
    assert row["monotone"] and row["asymptotic_admitted"]


# ---------------------------------------------------------------------------
# one real lattice level (solver regression pin); ~0.1 s
# ---------------------------------------------------------------------------
def test_level_d1_matches_recorded_value(tmp_path):
    cell = case.evaluate(0.70, n_radial=8, n_chord=2, wake_steps=24,
                         cache_dir=tmp_path)
    assert cell["KT"] == pytest.approx(0.170944, abs=5e-7)
    assert cell["10KQ"] == pytest.approx(0.280106, abs=1e-5)
    assert cell["eta0"] == pytest.approx(0.67990, abs=5e-5)
    assert cell["n_panels_per_blade"] == 16


# ---------------------------------------------------------------------------
# artefact consistency (skipped when the artefact has not been generated)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CASE_JSON.is_file(), reason="case artefact not generated")
def test_artefact_is_internally_consistent():
    payload = json.loads(CASE_JSON.read_text(encoding="utf-8"))
    cells = payload["cells"]
    for row in payload["gci"]:
        triple = sorted([c for c in cells if c["J"] == row["J"]],
                        key=lambda c: c["n_panels_per_blade"], reverse=True)
        again = case._gci(triple[0][row["quantity"]], triple[1][row["quantity"]],
                          triple[2][row["quantity"]], row["quantity"])
        assert again["gci_fine_pct"] == pytest.approx(row["gci_fine_pct"], rel=1e-12)
        assert again["observed_order"] == pytest.approx(row["observed_order"], rel=1e-12)
        # the /3 %/ flag in the artefact must follow the number, not a verdict
        assert row["under_case_tol_3pct"] == (row["gci_fine_pct"] <= 3.0)

    # the reference-procedure block must still pass, and the anchor row must be
    # quoted verbatim from the committed table (no re-typed numbers).
    assert payload["reference_verification"]["all_pass"]
    assert payload["case"]["anchor"]["row_J0.7_verbatim"].split()[0] == "0.7"
    assert payload["verdict"]["tolerance_pct"] == 3.0

    # every strategy knob that must be held is recorded
    assert payload["design"]["refinement_ratio"] == 2.0
    assert [l["n_radial"] for l in payload["design"]["levels"]] == [8, 16, 32]

def test_asymptotic_ratio_is_an_identity_not_a_test():
    """Pin the methodological caveat the case report leans on.

    For one three-grid triplet the shipped ``asymptotic_ratio`` reduces to
    ``f_fine / f_medium`` exactly, for *any* observed order -- including a
    negative one.  It must therefore never be used alone to admit a GCI.
    """
    for fine, medium, coarse in ((0.158243, 0.170180, 0.170944),
                                 (1.30, 1.10, 1.04),
                                 (0.028089, 0.030808, 0.031264)):
        res = grid_convergence_index(fine, medium, coarse, refinement_ratio=2.0)
        assert res.asymptotic_ratio == pytest.approx(fine / medium, rel=1e-12)
