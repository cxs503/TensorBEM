"""Gate ``G0`` (solver admissibility) of CASE-BEM-CUMMINS-WIGLEY.

G1-G5 are *consistency* statements about one sweep, so all five are blind to the
one defect that invalidates all five: a node whose BEM solve never converged.
The committed artifact ``docs/m20021/sweep_medium.npz`` is the counter-example
that made this gate necessary -- 41/41 nodes above the case pool's 1 % solver
limit, and a negative heave damping ``B33 = -129.9`` at ``w = 5.2912`` rad/s that
was read as physics until the residual exposed it as a bad node solve.

These tests pin:

1. the tolerance is the case pool's ``C0`` limit, predeclared, not tuned;
2. the two committed bands are admissible and the diverged artifact is not;
3. a single poisoned node moves a gate by 5-36 pp while the verdict string does
   not change -- G0 is the only gate that names it;
4. a sweep with no recorded residual is *skipped* (``passed`` is ``None``), never
   passed, so an old artifact cannot borrow the gate;
5. the closure pipeline refuses to cache a diverged node and refuses to gate a
   cache that already contains one.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pytest

from tensorbem.benchmarks import cummins_wigley_kk_causality as case

REPO = pathlib.Path(__file__).resolve().parents[1]
CASE_DIR = REPO / "benchmarks" / "verified" / "casepool_bem_cummins_wigley"
DIVERGED = REPO / "docs" / "m20021" / "sweep_medium.npz"
SCRIPT = REPO / "scripts" / "run_cummins_wigley_closure_full.py"


def sweep_from_npz(path: pathlib.Path, level_key: str = "medium") -> dict:
    payload = np.load(path)
    n_sections, n_levels = case.MESH_LEVELS[level_key]
    return {
        "level": f"n_sections={n_sections},n_levels={n_levels}",
        "level_key": level_key,
        "rotation_center": list(case.ROTATION_CENTER),
        "omega": np.asarray(payload["omega"], dtype=float).tolist(),
        "added_mass": np.asarray(payload["added_mass"], dtype=float).tolist(),
        "damping": np.asarray(payload["damping"], dtype=float).tolist(),
        "relative_residual": np.asarray(
            payload["solver_relative_residual"], dtype=float).tolist(),
        "wall_seconds": [0.0] * payload["omega"].size,
        "density": float(case.DENSITY),
        "gravity": float(case.GRAVITY),
    }


@pytest.fixture(scope="module")
def committed() -> dict:
    return sweep_from_npz(CASE_DIR / "sweep_medium.npz")


def _poison(sweep: dict, omega_node: float, *,
            damping_scale: float = 0.4, residual: float = 1.245) -> dict:
    """Copy with one node's heave damping scaled (stays positive) and a bad residual."""
    copy = json.loads(json.dumps(sweep))
    omega = np.asarray(copy["omega"], dtype=float)
    index = int(np.argmin(np.abs(omega - omega_node)))
    damping = np.asarray(copy["damping"], dtype=float)
    damping[index, 2, 2] *= damping_scale
    copy["damping"] = damping.tolist()
    copy["relative_residual"][index] = residual
    assert damping[index, 2, 2] > 0.0, "the poison must stay admissible to the kernel"
    return copy


def test_tolerance_is_the_predeclared_case_pool_solver_limit():
    assert case.SOLVER_RESIDUAL_TOL == 0.01
    assert case.GATES["G0_solver_validity"]["tol"] == case.SOLVER_RESIDUAL_TOL
    # listed before the consistency gates: it is evaluated first
    assert next(iter(case.GATES)) == "G0_solver_validity"


def test_diverged_evidence_artifact_is_inadmissible():
    """The real counter-example: 41/41 nodes over the 1 % limit."""
    sweep = sweep_from_npz(DIVERGED)
    admissibility = case.residual_admissibility(sweep)
    assert admissibility["nodes"] == 41
    assert admissibility["n_offenders"] == 41
    assert admissibility["passed"] is False
    assert admissibility["worst_residual"] > 1.2
    assert admissibility["worst_omega"] == pytest.approx(5.29116, abs=1e-4)

    # and the five consistency gates cannot even run on it: the kernel builder
    # refuses the negative damping deep inside the evaluation, with no node
    # attribution -- which is why the guard has to sit *before* the gates.
    with pytest.raises(ValueError, match="negative radiation damping"):
        case.gates_from_sweep(sweep, include_cross_solver=False)


def test_committed_band_is_admissible(committed):
    admissibility = case.residual_admissibility(committed)
    assert admissibility["n_offenders"] == 0
    assert admissibility["worst_residual"] < case.SOLVER_RESIDUAL_TOL
    analysis = case.gates_from_sweep(committed, include_cross_solver=False)
    assert analysis["admissible"] is True
    assert analysis["interpretable"] is True
    assert analysis["gates"]["G0_solver_validity"]["passed"] is True
    assert "G0_solver_validity" not in analysis["failed"]


def test_a_poisoned_node_is_invisible_to_the_consistency_gates(committed):
    """One wrong node: the five gates change by pp, the verdict string does not."""
    poison = _poison(committed, 5.2912, damping_scale=0.4)
    clean = case.gates_from_sweep(committed, include_cross_solver=False,
                                 include_residual_gate=False)
    dirty = case.gates_from_sweep(poison, include_cross_solver=False,
                                  include_residual_gate=False)
    assert dirty["failed"] == clean["failed"], (
        "the consistency gates must be shown to be insensitive here"
    )
    assert "G0_solver_validity" not in dirty["gates"]
    moved = abs(dirty["gates"]["G4_free_decay"]["value"]
                - clean["gates"]["G4_free_decay"]["value"])
    assert moved > 0.05, f"perturbation too weak to be evidence ({moved:.3f})"
    # G4 moves by more than half its own tolerance while nothing complains
    assert moved > 0.5 * case.GATES["G4_free_decay"]["tol"]

    guarded = case.gates_from_sweep(poison, include_cross_solver=False)
    assert guarded["admissible"] is False
    assert guarded["interpretable"] is False
    assert "G0_solver_validity" in guarded["failed"]
    g0 = guarded["gates"]["G0_solver_validity"]
    assert g0["offenders"] and g0["offenders"][0]["residual"] == pytest.approx(1.245)
    assert g0["worst_omega"] == pytest.approx(5.29116, abs=1e-4)


def test_sweep_without_residuals_is_skipped_not_passed(committed):
    stripped = dict(committed)
    stripped.pop("relative_residual")
    admissibility = case.residual_admissibility(stripped)
    assert admissibility["passed"] is None
    assert admissibility["unavailable"] == admissibility["nodes"]
    analysis = case.gates_from_sweep(stripped, include_cross_solver=False)
    g0 = analysis["gates"]["G0_solver_validity"]
    assert g0["passed"] is None and "skipped" in g0
    assert "G0_solver_validity" not in analysis["failed"]


# ---------------------------------------------------------------------------
# the pipeline end: poison never enters the cache, never reaches a verdict
# ---------------------------------------------------------------------------


def _load_pipeline():
    spec = importlib.util.spec_from_file_location("cw_closure_full_g0", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fake_solve(residual: float, calls: list[float] | None = None):
    def solve_point(level, omega, **kwargs):
        if calls is not None:
            calls.append(float(omega))
        identity = np.eye(6)
        return {
            "level": level,
            "omega": float(omega),
            "added_mass": identity.tolist(),
            "damping": identity.tolist(),
            "relative_residual": residual,
            "rotation_center": [0.0, 0.0, -0.09375],
            "density": 1025.0,
            "gravity": 9.80665,
            "wall_seconds": 0.0,
        }

    return solve_point


def test_solve_stage_refuses_to_cache_a_diverged_node(tmp_path, monkeypatch):
    pipeline = _load_pipeline()
    monkeypatch.setattr(pipeline.case, "solve_point", _fake_solve(1.245))
    with pytest.raises(SystemExit) as excinfo:
        pipeline.run_solve_stage(
            pipeline.argparse.Namespace(
                level="medium", outdir=str(tmp_path), grid="extended",
                shard=0, shards=1, resonance=None,
            )
        )
    message = str(excinfo.value)
    assert "did not converge" in message and "residual" in message
    assert not list((tmp_path / "points").glob("medium_omega_*.json")), (
        "a diverged node must never reach the cache the gates read"
    )
    rejected = list((tmp_path / "rejected").glob("medium_omega_*.json"))
    assert len(rejected) == 1, "the failure must be on record"
    assert json.loads(rejected[0].read_text())["relative_residual"] == 1.245


def test_gates_stage_refuses_a_cache_with_a_diverged_node(tmp_path, monkeypatch):
    pipeline = _load_pipeline()
    nodes = [float(w) for w in pipeline.case.EXTENDED_GRID]
    (tmp_path / "points").mkdir(parents=True, exist_ok=True)
    bad = nodes[len(nodes) // 2]
    for omega in nodes:
        payload = _fake_solve(1.2 if omega == bad else 1e-9)("medium", omega)
        (tmp_path / "points" / f"medium_omega_{omega:07.3f}.json").write_text(
            json.dumps(payload))
    with pytest.raises(SystemExit) as excinfo:
        pipeline.run_gate_stage(
            pipeline.argparse.Namespace(
                level="medium", outdir=str(tmp_path), out="results.json",
                grid="extended", shard=0, shards=1, resonance=None,
            )
        )
    message = str(excinfo.value)
    assert "solver-inadmissible" in message
    assert f"{bad:.3f}" in message
    assert not (tmp_path / "results.json").exists(), (
        "no verdict may be published from diverged nodes"
    )