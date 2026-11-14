"""Tests for the staged / sharded extended-band pipeline of the Cummins-Wigley case.

The structural defect these tests pin down is not numerical: rounds 2-5 of the
anchor ``CASE-BEM-CUMMINS-WIGLEY`` each lost their extended-band panel sweep to
one sequential job that ran past the scheduler budget, so the G1-G5 gates were
never evaluated on the predeclared [0.10, 20] rad/s band (see
``docs/dev_report_m20462_cummins_wigley_band_closure.md`` §5).  A pipeline that
can only be run end-to-end cannot be resumed, and a job that dies at 90 % leaves
nothing.  ``scripts/run_cummins_wigley_closure_full.py`` therefore grew a
``--stage solve|gates`` split with ``--shard/--shards``; these tests check the
three properties that make the split trustworthy:

1. the shards partition the grid exactly (every node solved once, none lost);
2. the solving stage only solves its own slice, and resumes from the cache;
3. the gates stage never interpolates and never re-solves -- a missing node is
   a refusal, not a silently narrowed band.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_cummins_wigley_closure_full.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("cw_closure_full", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pipeline():
    return _load_module()


def test_shards_partition_the_grid_exactly(pipeline):
    nodes = [float(w) for w in pipeline.case.EXTENDED_GRID]
    for shards in (1, 3, 8, len(nodes), len(nodes) + 3):
        union = []
        for shard in range(shards):
            union.extend(pipeline._shard_slice(nodes, shard, shards))
        assert sorted(union) == sorted(nodes), f"shards={shards} lost or duplicated a node"
        assert len(union) == len(set(union))


def test_shard_bounds_are_rejected(pipeline):
    nodes = [1.0, 2.0]
    for shard, shards in ((-1, 2), (2, 2), (0, 0)):
        with pytest.raises(SystemExit):
            pipeline._shard_slice(nodes, shard, shards)


def _fake_solve(calls: list[float]):
    def solve_point(level, omega, **kwargs):
        calls.append(float(omega))
        identity = np.eye(6)
        return {
            "level": level,
            "omega": float(omega),
            "added_mass": identity.tolist(),
            "damping": identity.tolist(),
            "relative_residual": 1e-9,
            "rotation_center": [0.0, 0.0, -0.09375],
            "density": 1025.0,
            "gravity": 9.80665,
            "wall_seconds": 0.0,
        }

    return solve_point


def test_solve_stage_only_solves_its_slice_and_resumes(pipeline, tmp_path, monkeypatch):
    calls: list[float] = []
    monkeypatch.setattr(pipeline.case, "solve_point", _fake_solve(calls))
    nodes = [float(w) for w in pipeline.case.EXTENDED_GRID]
    shards = 4
    seen: list[float] = []
    for shard in range(shards):
        calls.clear()
        report = pipeline.run_solve_stage(
            pipeline.argparse.Namespace(
                level="medium", outdir=str(tmp_path), grid="extended",
                shard=shard, shards=shards, resonance=None,
            )
        )
        expected = pipeline._shard_slice(nodes, shard, shards)
        assert sorted(calls) == sorted(expected)
        assert report["slice_nodes"] == len(expected)
        seen.extend(calls)
    assert sorted(seen) == sorted(nodes)

    # resume: a second pass over shard 0 must not re-solve anything
    calls.clear()
    report = pipeline.run_solve_stage(
        pipeline.argparse.Namespace(
            level="medium", outdir=str(tmp_path), grid="extended",
            shard=0, shards=shards, resonance=None,
        )
    )
    assert calls == []
    assert report["solved_now"] == 0
    assert report["cached_total"] == len(nodes)
    written = sorted(p.name for p in (tmp_path / "points").glob("medium_omega_*.json"))
    assert len(written) == len(nodes)


def test_gates_stage_refuses_a_gap_instead_of_narrowing_the_band(
    pipeline, tmp_path, monkeypatch
):
    calls: list[float] = []
    monkeypatch.setattr(pipeline.case, "solve_point", _fake_solve(calls))
    nodes = [float(w) for w in pipeline.case.EXTENDED_GRID]
    # cache all but the top node
    for omega in nodes[:-1]:
        (tmp_path / "points").mkdir(parents=True, exist_ok=True)
        payload = _fake_solve(calls)("medium", omega)
        (tmp_path / "points" / f"medium_omega_{omega:07.3f}.json").write_text(
            json.dumps(payload)
        )
    calls.clear()
    with pytest.raises(SystemExit) as excinfo:
        pipeline.run_gate_stage(
            pipeline.argparse.Namespace(
                level="medium", outdir=str(tmp_path), out="results.json",
                grid="closure", shard=0, shards=1, resonance=7.38,
            )
        )
    message = str(excinfo.value)
    assert "not cached" in message and f"{nodes[-1]}" in message
    assert calls == [], "the gates stage must never re-solve"
    assert not (tmp_path / "results.json").exists()


def test_closure_grid_grows_only_by_the_resonance_cluster(pipeline):
    base = set(round(float(w), 9) for w in pipeline.case.EXTENDED_GRID)
    closure = [round(float(w), 9) for w in pipeline.case.closure_grid(7.38)]
    assert base <= set(closure)
    extra = sorted(set(closure) - base)
    # RESONANCE_STEP spacing inside +-RESONANCE_HALF_WIDTH, merged with the grid
    step = pipeline.case.RESONANCE_STEP
    assert max(np.diff(extra)) <= step + 1e-9
    assert min(extra) >= 7.38 - pipeline.case.RESONANCE_HALF_WIDTH - 1e-9
    assert max(extra) <= 7.38 + pipeline.case.RESONANCE_HALF_WIDTH + 1e-9