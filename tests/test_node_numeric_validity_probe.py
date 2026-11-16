"""Tests for ``tools/node_numeric_validity_probe.py`` (decision rule + doc pin).

The tool decides whether a compute node reproduces the committed
CASE-BEM-CUMMINS-WIGLEY coarse radiation sweep.  The decision rule has to be
mechanical, so it is tested on synthetic sweeps: no BEM solve here (the real
probe is dispatched to the cluster, see ``docs/casepool_bem_cummins_wigley/
NODE_NUMERIC_VALIDITY.md``).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "node_numeric_validity_probe.py"
DOC_DIR = REPO_ROOT / "docs" / "casepool_bem_cummins_wigley"

DOFS = ("surge", "sway", "heave", "roll", "pitch", "yaw")


def _load_tool():
    spec = importlib.util.spec_from_file_location("node_numeric_validity_probe", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


probe_tool = _load_tool()


def _synthetic(n_omega: int = 5, *, amplitude: float = 1.0) -> dict[str, np.ndarray]:
    """A positive-definite-damping sweep with a smooth diagonal signature."""
    omega = np.logspace(np.log10(0.2), np.log10(12.0), n_omega)
    added = np.zeros((n_omega, 6, 6))
    damping = np.zeros((n_omega, 6, 6))
    for k, w in enumerate(omega):
        for dof in range(6):
            value = amplitude * (dof + 1) * np.exp(-((w - 3.0) ** 2) / 8.0)
            added[k, dof, dof] = 10.0 * value
            damping[k, dof, dof] = value
        added[k, 0, 4] = added[k, 4, 0] = 0.1 * amplitude
    return {
        "omega": omega,
        "added_mass": added,
        "damping": damping,
        "residual": np.full(n_omega, 1.0e-3),
    }


def _write(path: Path, sweep: dict[str, np.ndarray]) -> Path:
    np.savez(
        path,
        omega=sweep["omega"],
        added_mass=sweep["added_mass"],
        damping=sweep["damping"],
        solver_relative_residual=sweep["residual"],
    )
    return path


def test_identical_sweeps_match() -> None:
    sweep = _synthetic()
    result = probe_tool.decide(sweep, sweep, dof_names=DOFS)
    assert result["decision"] == "MATCH"
    assert result["amplitude"]["added_mass_rel"] == pytest.approx(0.0, abs=1e-15)
    assert [row["dof"] for row in result["amplitude"]["per_dof_damping_rel"]] == list(DOFS)


def test_rounding_level_difference_is_near_not_match() -> None:
    reference = _synthetic()
    probe = {key: value.copy() for key, value in reference.items()}
    probe["added_mass"] *= 1.0 + 1.0e-4
    result = probe_tool.decide(probe, reference, dof_names=DOFS)
    assert result["decision"] == "NEAR"
    assert 1.0e-6 < result["amplitude"]["added_mass_rel"] <= 1.0e-3


def test_rescaled_damping_diverges() -> None:
    """The negative control: a 30 % amplitude change is not the same solve."""
    reference = _synthetic()
    probe = {key: value.copy() for key, value in reference.items()}
    probe["damping"] *= 1.3
    result = probe_tool.decide(probe, reference, dof_names=DOFS)
    assert result["decision"] == "DIVERGE"
    assert result["amplitude"]["damping_rel"] == pytest.approx(0.3, rel=1e-6)


def test_unconverged_solve_is_flagged_before_anything_else() -> None:
    reference = _synthetic()
    probe = {key: value.copy() for key, value in reference.items()}
    probe["residual"] = np.full(probe["residual"].size, 1.253)  # the n315 value
    result = probe_tool.decide(probe, reference, dof_names=DOFS)
    assert result["decision"] == "NOT_CONVERGED"
    assert "1.253" in result["reasons"][0]


def test_negative_diagonal_damping_is_non_physical() -> None:
    """n315 reported -0.947 (surge) / -1.46 (pitch): energy-creating damping."""
    reference = _synthetic()
    probe = {key: value.copy() for key, value in reference.items()}
    probe["damping"][2, 0, 0] = -0.947
    probe["damping"][2, 4, 4] = -1.46
    result = probe_tool.decide(probe, reference, dof_names=DOFS)
    assert result["decision"] == "NON_PHYSICAL"
    assert result["validity"]["diagonal_damping_min"] == pytest.approx(-1.46)


def test_negative_off_diagonal_damping_is_not_penalised() -> None:
    """Only the diagonal is a radiation coefficient; coupling terms may be negative."""
    reference = _synthetic()
    probe = {key: value.copy() for key, value in reference.items()}
    probe["damping"][1, 0, 4] = -2.0
    probe["damping"][1, 4, 0] = -2.0
    result = probe_tool.decide(probe, reference, dof_names=DOFS)
    assert result["decision"] == "DIVERGE"  # amplitudes moved, but not NON_PHYSICAL


def test_mismatched_grids_are_refused_not_interpolated() -> None:
    reference = _synthetic()
    probe = {key: value.copy() for key, value in reference.items()}
    probe["omega"] = probe["omega"] * (1.0 + 1.0e-6)
    with pytest.raises(ValueError, match="frequency grids differ"):
        probe_tool.decide(probe, reference, dof_names=DOFS)


def test_compare_files_round_trips_through_npz(tmp_path: Path) -> None:
    reference = _synthetic()
    probe = {key: value.copy() for key, value in _synthetic().items()}
    probe["residual"] = np.full(probe["residual"].size, 6.4e-3)
    probe_ref = _write(tmp_path / "probe.npz", probe)
    ref_path = _write(tmp_path / "reference.npz", reference)
    result = probe_tool.compare_files(probe_ref, ref_path)
    assert result["decision"] == "MATCH"
    assert result["thresholds"]["solver_residual_max"] == 1.0e-2


def test_linalg_verdict_catches_a_broken_solve() -> None:
    report = probe_tool.linalg_probe(n=24)
    assert probe_tool.linalg_verdict(report)["passed"] is True
    report["problems"]["solve_cond_1e3"]["rel_residual"] = 1.0e-3
    verdict = probe_tool.linalg_verdict(report)
    assert verdict["passed"] is False
    assert any("solve_cond_1e3.rel_residual" in item for item in verdict["failures"])


@pytest.mark.skipif(
    not (DOC_DIR / "probe_sweep_meta.json").exists(),
    reason="node numeric-validity evidence not committed",
)
def test_committed_docs_report_matches_the_committed_probe_json() -> None:
    """The evidence doc and the README status block are generated, so they cannot drift."""
    rendered = probe_tool.render_report(DOC_DIR)
    committed = (DOC_DIR / "NODE_NUMERIC_VALIDITY.md").read_text(encoding="utf-8")
    assert rendered == committed

    status = probe_tool.render_status_block(DOC_DIR)
    readme = (DOC_DIR / "README.md").read_text(encoding="utf-8")
    assert status in readme
    assert probe_tool.STATUS_BEGIN in readme


@pytest.mark.skipif(
    not (DOC_DIR / "probe_sweep_meta.json").exists(),
    reason="node numeric-validity evidence not committed",
)
def test_committed_probe_artifact_has_the_reference_grid() -> None:
    meta = json.loads((DOC_DIR / "probe_sweep_meta.json").read_text(encoding="utf-8"))
    assert meta["grid_matches_case_default"] is True
    assert meta["panel_count"] == 192  # coarse 12x4, the disputed n315 configuration
    payload = np.load(DOC_DIR / "probe_sweep_coarse.npz")
    assert payload["added_mass"].shape == (meta["n_omega"], 6, 6)
    assert np.all(np.isfinite(payload["added_mass"]))

def _synthetic_probe_dir(tmp_path: Path) -> Path:
    """A probe directory with the same JSON schema as a real cluster run."""
    linalg = probe_tool.linalg_probe(n=24)
    linalg["environment"] = {"platform": "synthetic", "machine": "none", "python": "0", "numpy": "0"}
    linalg["verdict"] = probe_tool.linalg_verdict(linalg)
    (tmp_path / "probe_linalg.json").write_text(json.dumps(linalg), encoding="utf-8")

    reference = _synthetic()
    probe = {key: value.copy() for key, value in reference.items()}
    probe["residual"] = np.full(probe["residual"].size, 6.4e-3)
    _write(tmp_path / "probe_sweep_coarse.npz", probe)
    _write(tmp_path / "committed.npz", reference)
    _write(tmp_path / "disputed.npz", reference)
    meta = {
        "kind": "sweep_probe",
        "hostname": "synthetic-host",
        "platform": "synthetic",
        "machine": "none",
        "python": "0",
        "numpy": "0",
        "label": "coarse",
        "mesh": [12, 4],
        "panel_count": 192,
        "max_panel_scale": 0.01,
        "band_rad_s": [0.2, 12.0],
        "n_omega": int(reference["omega"].size),
        "grid_matches_case_default": True,
        "solver_relative_residual_max": 6.4e-3,
        "diagonal_damping_min": 1e-3,
        "diagonal_damping_min_per_dof": [1e-3] * 6,
        "elapsed_s": 1.0,
        "wall_s": 2.0,
        "artifact": "synthetic",
    }
    (tmp_path / "probe_sweep_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    for name in probe_tool.REFERENCE_LABELS:
        (tmp_path / name).write_text(
            json.dumps(probe_tool.decide(probe, reference, dof_names=DOFS)), encoding="utf-8"
        )
    return tmp_path


def test_report_and_status_block_render_from_probe_json(tmp_path: Path) -> None:
    probe_dir = _synthetic_probe_dir(tmp_path)
    report = probe_tool.render_report(probe_dir)
    assert "synthetic-host" in report
    assert "| **MATCH** |" in report  # both comparisons land in the decision table
    assert probe_tool.COMMITTED_REFERENCE.name in str(probe_tool.COMMITTED_REFERENCE)
    assert "Error judgement" in report
    status = probe_tool.render_status_block(probe_dir)
    assert probe_tool.STATUS_BEGIN in status and probe_tool.STATUS_END in status
    assert "`MATCH`" in status


def test_write_report_injects_the_block_into_a_readme(tmp_path: Path) -> None:
    probe_dir = _synthetic_probe_dir(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(
        "intro\n" + probe_tool.STATUS_BEGIN + "\nstale\n" + probe_tool.STATUS_END + "\noutro\n",
        encoding="utf-8",
    )
    probe_tool.write_report(probe_dir, tmp_path / "DOC.md", readme)
    text = readme.read_text(encoding="utf-8")
    assert "stale" not in text
    assert "synthetic-host" in text
    assert text.startswith("intro") and text.rstrip().endswith("outro")
    assert "## 1. Question and decision" in (tmp_path / "DOC.md").read_text(encoding="utf-8")
