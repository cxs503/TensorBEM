"""Gates for the three-axis attribution study (`capytaine_attribution`).

The study answers a question the cross-code comparison cannot: when TensorBEM
and capytaine disagree by 8.75 %, *which* code is wrong?  These tests pin the
answer to artefacts instead of prose:

1. the attribution rule is a pure function with the documented semantics;
2. the recorded cross-code FAIL is read back unchanged -- attribution may never
   quietly upgrade it;
3. the analytic axis publishes a truncation residual far below the band, so it
   can serve as truth;
4. every cell of the frozen study is attributed, and in the cells where the
   cross-code band is exceeded both one-sided errors point in opposite
   directions and stay in band;
5. the committed document is byte-identical to `render_document(results.json)`
   (no hand-transcribed number can survive);
6. a cheap solve reproduces the refinement trend (residual against the analytic
   axis shrinks from 16 to 64 panels).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.benchmarks import capytaine_attribution as attr

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / attr.ARTIFACT_ROOT / attr.RESULT_FILE


@pytest.fixture(scope="module")
def study() -> dict:
    if not STUDY.exists():  # pragma: no cover - artefact missing
        pytest.skip("attribution artefact missing")
    return json.loads(STUDY.read_text())


# --------------------------------------------------------------------------
# 1. the rule itself
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "e_tb, e_ref, expected",
    [
        (0.01, 0.02, "BOTH_WITHIN_BAND"),      # both inside the band
        (0.05, 0.049, "BOTH_WITHIN_BAND"),      # band edge is inclusive
        (0.20, 0.01, "TENSORBEM"),              # our solver is the outlier
        (0.20, 0.25, "REFERENCE"),              # the foreign code is the outlier
        (0.20, 0.20, "TENSORBEM"),              # tie above band blames our code
        (0.06, 0.06, "TENSORBEM"),              # conservative tie-breaking
    ],
)
def test_classify_semantics(e_tb: float, e_ref: float, expected: str) -> None:
    assert attr.classify(e_tb, e_ref) == expected


def test_classify_is_band_parameterised() -> None:
    # a cell that is a defect at 5 % is not one at 25 %: the band is the only
    # thing that changed, so the rule must follow it.
    assert attr.classify(0.10, 0.05, band=0.05) == "TENSORBEM"
    assert attr.classify(0.10, 0.05, band=0.25) == "BOTH_WITHIN_BAND"


def test_classify_symmetric_in_truth_not_in_blame() -> None:
    assert attr.classify(0.30, 0.01) == "TENSORBEM"
    assert attr.classify(0.01, 0.30) == "REFERENCE"


# --------------------------------------------------------------------------
# 2. the honest FAIL survives
# --------------------------------------------------------------------------
def test_cross_code_fail_is_not_overwritten(study: dict) -> None:
    cross = study["cross_code"]
    frozen = json.loads((ROOT / attr.CROSSCHECK_RESULTS).read_text())
    assert cross["verdict"] == frozen["comparison"]["verdict"] == "FAIL"
    assert cross["worst_relative"] == pytest.approx(
        frozen["comparison"]["worst_relative"], rel=0, abs=0
    )
    assert cross["overwritten"] is False
    assert cross["failures"] == list(frozen["comparison"]["failures"])


def test_finer_mesh_closes_the_cross_code_gap(study: dict) -> None:
    """The FAIL was measured on 256 panels.  At 1024 the same metric is inside the
    band -- which is the claim being made (resolution, not a code change), so it
    must be true and it must be a *shrink* relative to the recorded number."""
    cross = study["cross_code"]
    assert study["attribution"]["tensorbem_faces"] == study["tensorbem"]["levels"][-1]["n_faces"]
    assert study["attribution"]["worst_e_cross"] <= study["band"]
    assert study["attribution"]["worst_e_cross"] < cross["worst_relative"]


# --------------------------------------------------------------------------
# 3. the truth axis grades itself
# --------------------------------------------------------------------------
def test_analytic_residual_is_negligible(study: dict) -> None:
    residual = study["analytic"]["worst_residual_relative"]
    assert residual < 1e-3, residual
    # and sharper than the smallest deviation it is asked to judge
    smallest = min(
        min(abs(c["e_tb"]), abs(c["e_ref"])) for c in study["attribution"]["cells"]
    )
    assert residual < 0.1 * smallest


def test_analytic_axis_converges_in_terms() -> None:
    """More multipole terms must move the value monotonically toward a limit."""
    vals = [attr._hulme_values(1.0, n)["heave"]["damping"] for n in (48, 96, 192)]
    diffs = [vals[1] - vals[0], vals[2] - vals[1]]
    assert diffs[0] != 0.0 and diffs[1] != 0.0
    assert abs(diffs[1]) < abs(diffs[0]), (vals, diffs)
    # the sequence must contract: 48 vs 192 differ by well under 0.1 %
    assert abs(vals[2] - vals[0]) / vals[2] < 1e-3


# --------------------------------------------------------------------------
# 4. every cell attributed, and the failing cells explained
# --------------------------------------------------------------------------
def test_all_cells_attributed(study: dict) -> None:
    att = study["attribution"]
    assert len(att["cells"]) == len(study["nu_grid"]) * 2 * 2  # nu x dof x quantity
    assert sum(att["counts"].values()) == len(att["cells"])
    assert att["counts"].get("TENSORBEM", 0) == 0, att["failures"]["TENSORBEM"]
    assert att["verdict_analytic"] == "PASS"


def test_recorded_failing_cells_are_in_band_on_both_sides(study: dict) -> None:
    """Every cell the recorded cross-code study fails on must be excused by the
    analytic axis: both one-sided errors in band and of opposite sign."""
    band = study["band"]
    att = study["attribution"]
    cells = {(round(c["nu"], 3), c["dof"], c["quantity"]): c for c in att["cells"]}
    matched = 0
    for line in study["cross_code"]["failures"]:
        parts = line.split()
        key = (round(float(parts[1].split("=")[1]), 3), parts[2], parts[3].rstrip(":"))
        if key not in cells:
            continue
        cell = cells[key]
        matched += 1
        assert abs(cell["e_tb"]) <= band, cell
        assert abs(cell["e_ref"]) <= band, cell
        assert cell["attribution"] == "BOTH_WITHIN_BAND", cell
        assert math.copysign(1.0, cell["e_tb"]) != math.copysign(1.0, cell["e_ref"]), cell
    assert matched == len(study["cross_code"]["failures"]) > 0


def test_refinement_improves_the_residual(study: dict) -> None:
    rows = study["refinement"]["tensorbem"]
    worst = [r["worst_vs_analytic"] for r in rows]
    assert worst[-1] < worst[0]
    for a, b in zip(worst, worst[1:]):
        assert b < a, worst  # monotone in the panel count
    order = study["refinement"]["tensorbem_apparent_order"]
    assert order is not None and 0.5 < order < 2.5, order


def test_hemisphere_volume_closes_with_refinement(study: dict) -> None:
    """The mesh geometry is graded before the physics is trusted: the faceted
    hemisphere under-fills the exact volume and must approach it as h -> 0."""
    rows = study["tensorbem"]["levels"]
    deficits = [abs(r["volume_relative"]) for r in rows]
    assert deficits[-1] < 1e-2, rows           # finest level within 1 % of 2/3 pi R^3
    for a, b in zip(deficits, deficits[1:]):
        assert b < a, deficits                  # and closing monotonically


# --------------------------------------------------------------------------
# 5. the document cannot disagree with the numbers
# --------------------------------------------------------------------------
def test_document_is_rendered_from_the_artefact(study: dict) -> None:
    doc = ROOT / attr.DOC_PATH
    assert doc.exists(), doc
    assert doc.read_text() == attr.render_document(study)


def test_document_quotes_the_verdicts(study: dict) -> None:
    text = (ROOT / attr.DOC_PATH).read_text()
    att = study["attribution"]
    assert f"{100.0 * att['worst_e_tb']:.2f} %" in text
    assert f"{100.0 * study['cross_code']['worst_relative']:.2f} %" in text
    assert "**FAIL**" in text or "FAIL stands unchanged" in text


# --------------------------------------------------------------------------
# 6. cheap live reproduction of the refinement trend
# --------------------------------------------------------------------------
def test_live_refinement_trend_is_reproducible() -> None:
    """Two coarse levels solved here (16 and 64 panels, ~1 s) must show the same
    direction as the frozen study: refining moves the answer toward Hulme."""
    analytic = attr.analytic_axis(terms=(96, 144, 192))
    coarse = attr.sweep_tensorbem((1, 2))
    residuals = [
        attr.refinement({"levels": [row], "reference_faces": row["n_faces"]},
                        [{"n_faces": 1, "values": row["values"]}], analytic)
        ["tensorbem"][0]["worst_vs_analytic"]
        for row in coarse["levels"]
    ]
    assert residuals[1] < residuals[0], residuals
    assert not np.isnan(residuals).any()