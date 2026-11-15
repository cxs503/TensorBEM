"""Round-3 guards for CASE-BEM-HULME-HEMISPHERE: the ka > 2 band on artifacts.

Every assertion below is recomputed from the committed artifacts, not read off a
verdict field, so a relaxed gate or a hand-edited conclusion fails the test.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

CASE = (Path(__file__).resolve().parents[1] / "benchmarks" / "verified"
        / "casepool_bem_hulme_hemisphere")
CURRENT = CASE / "band_current_code" / "results.json"
COMMITTED = CASE / "band_extension" / "results.json"
REFINE = CASE / "band_edge_refinement" / "results.json"
ROUND1 = CASE / "evidence" / "np246_contradiction.json"
RESOLUTION = CASE / "evidence" / "band_edge_resolution.json"
GATE = 0.05


def _load(path: Path):
    if not path.exists():
        pytest.skip(f"artifact missing: {path.name}")
    return json.loads(path.read_text())


def _band_max(cell) -> float:
    return max(abs(cell[k]) for k in cell if k.startswith("rel_error_"))


def _extension_cells(payload):
    return sorted([c for c in payload["cells"] if c["kind"] == "extension"],
                  key=lambda c: c["ka"])


def test_every_band_cell_records_how_its_vector_was_obtained():
    payload = _load(CURRENT)
    for cell in payload["cells"]:
        assert cell.get("radiation_solve_method") in {"lstsq", "solve", "qr"}, cell["name"]
        assert cell.get("radiation_relative_residual") is not None, cell["name"]
        assert cell["radiation_relative_residual"] < 1e-2, cell["name"]
        assert cell.get("diffraction_relative_residual") is not None, cell["name"]
        assert math.isfinite(cell["condition_number"]) and cell["condition_number"] > 0


def test_provenance_is_complete_in_the_artifact_schema_that_is_actually_used():
    """The ka>2 artifacts use git_head/environment; no reader may assume
    ``metadata``.  Host, numpy version, git head and per-module hashes must be
    present, otherwise the run cannot be audited at all."""
    for path in (CURRENT, COMMITTED):
        payload = _load(path)
        meta = payload.get("metadata") or {}
        env = payload.get("environment") or {}
        assert (meta.get("host") or env.get("host")), path.name
        assert (meta.get("numpy") or env.get("numpy")), path.name
        assert (meta.get("git_commit") or payload.get("git_head")), path.name
        sha = meta.get("source_sha256") or payload.get("source_sha256")
        assert sha and "src/tensorbem/free_surface3d.py" not in sha
        assert "src/tensorbem/free_surface_solver3d.py" in sha, path.name


def test_two_current_code_runs_agree_cell_by_cell():
    a = {c["name"]: c for c in _load(COMMITTED)["cells"]}
    b = {c["name"]: c for c in _load(CURRENT)["cells"]}
    shared = sorted(set(a) & set(b))
    assert len(shared) >= 9
    for name in shared:
        keys = [k for k in a[name] if k.startswith("rel_error_")]
        worst = max(abs(a[name][k] - b[name][k]) for k in keys)
        assert worst < 1e-9, (name, worst)


def test_credited_band_stops_before_the_first_failing_ka():
    """ka in (2, 3.5] passes on the case mesh; ka = 4.0 does not, and it is not
    disguised as a pass by a widened tolerance."""
    ext = _extension_cells(_load(CURRENT))
    credited = [c["ka"] for c in ext if _band_max(c) < GATE]
    assert credited == [pytest.approx(2.2), pytest.approx(2.58),
                        pytest.approx(3.0), pytest.approx(3.5)]
    failing = [c["ka"] for c in ext if _band_max(c) >= GATE]
    assert failing == [pytest.approx(4.0)]
    worst_credited = max(_band_max(c) for c in ext if c["ka"] <= 3.5)
    assert worst_credited < GATE


def test_pre_audit_catastrophic_cells_recover_under_current_code():
    round1 = _load(ROUND1)
    current = {c["name"]: c for c in _load(CURRENT)["cells"]}
    name = f"extension_ka{round1['ka']:g}_n{round1['panels']}_chief_on"
    cell = current[name]
    for quantity, row in round1["rows"].items():
        assert abs(row["err_band_extension"]) > 0.5  # the crash being explained
        assert abs(cell[f"rel_error_{quantity}_dimless"]) < GATE, quantity


def test_band_edge_error_falls_with_mesh_refinement():
    """Three rungs if the 4096-panel job returned, two otherwise -- a rising
    error would mean the refinement statement is unsupported."""
    committed = _load(COMMITTED)
    coarse = {c["ka"]: c for c in committed["cells"] if c["kind"] == "ladder_coarse"}
    fine = {c["ka"]: c for c in _load(CURRENT)["cells"] if c["kind"] == "extension"}
    ka = max(coarse)
    ladder = [("256", _band_max(coarse[ka])), ("1024", _band_max(fine[ka]))]
    if REFINE.exists():
        for r in json.loads(REFINE.read_text())["cells"]:
            if abs(r["ka"] - ka) < 1e-9:
                ladder.append((str(r["panels"]), _band_max(r)))
    ladder.sort(key=lambda kv: int(kv[0]))
    assert len(ladder) >= 2
    for (_, e0), (_, e1) in zip(ladder, ladder[1:]):
        assert e1 < e0, ladder


def test_resolution_artifact_agrees_with_the_cells_it_cites():
    payload = _load(RESOLUTION)
    cells = {c["name"]: c for c in json.loads(CURRENT.read_text())["cells"]}
    for row in payload["band_extension_cells"]:
        assert row["band_max"] == pytest.approx(_band_max(cells[row["name"]]))
    be = payload["band_edge"]
    assert be["first_failing_ka"] == pytest.approx(4.0)
    assert pytest.approx(3.5) in be["credited_ka_on_case_mesh"]
    # the verdict must not claim a passing ka beyond the failing one
    assert payload["overclaimed_ka"] == []