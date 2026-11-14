"""Guards for the stage-wise cell fingerprint and for cross-artifact consistency.

Two distinct things are asserted here.

1. **The fingerprint tool is a faithful instrument** (``scripts/hulme_cell_fingerprint.py``):
   it must reproduce ``hulme_hemisphere_case.run_cell`` bit-for-bit on a cheap
   mesh, its comparison must report "no divergence" for two identical
   fingerprints and must report the first diverging *stage* (not just a
   different final number) when one stage is perturbed, and it must record
   enough provenance to tell two machines apart.

2. **Committed artefacts that claim to describe the same cell agree** -- or the
   disagreement is registered, with its measured size, in
   ``KNOWN_CROSSENV_DIVERGENCE`` below.  This is the guard that was missing when
   ``inhouse_ladder_ka2_rung4_node.json`` (numpy 2.0.1, cluster node) and
   ``results.json`` (numpy 2.4.6, control plane) were both committed for
   ``ka = 2`` / 1024 panels: the same mesh, the same reported condition number
   and a heave damping that differs by 2.05x.  Neither file is edited here: the
   registry makes the divergence *visible and bounded* (delta and environment
   are pinned, so it cannot silently grow), and it fails loudly once the
   divergence disappears so the registry cannot outlive the defect.

Run: ``python -m pytest tests/test_hulme_cell_fingerprint.py -q`` (~3 s, one
256-panel and one 64-panel solve; no 1024-panel solve).
"""
from __future__ import annotations

import copy
import json
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
CASEPOOL = REPO / "benchmarks" / "verified" / "casepool_bem_hulme_hemisphere"
QUANTITIES = ("A11_dimless", "B11_dimless", "A33_dimless", "B33_dimless", "Fx_dimless", "Fz_dimless")

# Relative agreement demanded between two artefacts that describe the same cell.
# 1e-6 is far above float64 roundoff for these magnitudes (~1e-15) and far below
# any physics tolerance (5e-2), so only a genuinely different computation trips it.
SAME_CELL_TOL = 1e-6

# Registered cross-environment divergences: cell -> (max |relative| delta of the
# six quantities, what differs, evidence).
# See docs/bug_bem_numpy_version_drift.md and
# docs/case_bem_hulme_inhouse_ladder_env_dependence.md.
# Key = (ka, subdivisions, chief). ``chief=False`` rows are a different
# configuration of the same mesh, so they are grouped separately on purpose.
KNOWN_CROSSENV_DIVERGENCE = {
    (2.0, 4, True): {
        "max_rel_delta": 0.5105,
        "rel_tol": 0.02,  # registered value +/- this; a bigger drift is a new defect
        "differs_in": "numpy 2.4.6 (control plane) vs 2.0.1 (cluster node n315)",
        "evidence": "inhouse_ladder_ka2_rung4_node.json vs results.json / inhouse_ladder_ka2.json",
    },
    (2.0, 5, True): {
        # 4096 panels were only ever run on the node side (cj2750) -- there is no
        # control-plane 4096 rung to compare against, so this entry documents the
        # absence rather than a measured cross-environment delta.
        "max_rel_delta": None,
        "rel_tol": None,
        "differs_in": "node-only rung (numpy 2.0.1); no control-plane counterpart exists",
        "evidence": "inhouse_ladder_ka2_rung5.json",
    },
}


def _load_fingerprint_module():
    path = REPO / "scripts" / "hulme_cell_fingerprint.py"
    spec = importlib.util.spec_from_file_location("hulme_cell_fingerprint_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fingerprint_module():
    return _load_fingerprint_module()


@pytest.fixture(scope="module")
def cheap_pair(fingerprint_module):
    """Two fingerprints of the same 256-panel cell (the cheapest non-trivial mesh)."""
    first = fingerprint_module.fingerprint(2.0, 3)
    second = fingerprint_module.fingerprint(2.0, 3)
    return first, second


# --- 1. the instrument -----------------------------------------------------
def test_fingerprint_reproduces_run_cell(cheap_pair):
    """The instrument must agree with the library entry point it instruments."""
    from tensorbem.benchmarks.hulme_hemisphere_case import run_cell

    payload, _ = cheap_pair
    cell = run_cell(2.0, 3)
    for key in QUANTITIES:
        assert payload["results"][key] == pytest.approx(cell[key], rel=0.0, abs=1e-13), (
            f"fingerprint {key} disagrees with run_cell on the same 256-panel cell"
        )
    assert payload["mesh"]["panel_count"] == cell["panels"]


def test_fingerprint_is_repeatable_and_self_comparison_is_clean(fingerprint_module, cheap_pair):
    first, second = cheap_pair
    rows, same = fingerprint_module.compare(first, second)
    assert same, [r for r in rows if r["status"] != "same"]
    # Bit-level identity of the assembled operator is the property that makes a
    # cross-machine diff meaningful, so assert the hashes, not just the norms.
    assert (
        first["operators"]["system_matrix"]["sha256_rounded"]
        == second["operators"]["system_matrix"]["sha256_rounded"]
    )


def test_comparison_localises_the_diverging_stage(fingerprint_module, cheap_pair):
    """A perturbed *results* stage must be reported there, not smeared everywhere."""
    first, _ = cheap_pair
    tampered = copy.deepcopy(first)
    tampered["results"]["B33_dimless"] *= 2.0
    rows, same = fingerprint_module.compare(first, tampered)
    assert not same
    diverging = [r for r in rows if r["status"] == "DIFFERS" and not r.get("downstream_of_divergence")]
    assert diverging, "comparison failed to flag the tampered scalar"
    assert {r["stage"] for r in diverging} == {"results"}
    assert {r["key"] for r in diverging} == {"B33_dimless"}
    # Upstream stages must stay clean: that is what makes the report causal.
    upstream = {r["stage"] for r in rows if r["stage"] in ("mesh", "operators", "spectrum")}
    assert all(
        r["status"] == "same" for r in rows if r["stage"] in upstream
    )


def test_comparison_localises_an_assembly_divergence(fingerprint_module, cheap_pair):
    """Perturbing the operator must be blamed on the operator stage."""
    first, _ = cheap_pair
    tampered = copy.deepcopy(first)
    tampered["operators"]["system_matrix"]["max_abs"] *= 1.01
    rows, same = fingerprint_module.compare(first, tampered)
    assert not same
    diverging = [r for r in rows if r["status"] == "DIFFERS" and not r.get("downstream_of_divergence")]
    assert [r["stage"] for r in diverging] == ["operators"]


def test_fingerprint_records_provenance_and_conditioning(cheap_pair):
    payload, _ = cheap_pair
    env = payload["environment"]
    prov = payload["provenance"]
    spec = payload["spectrum"]
    assert env["numpy"] == np.__version__
    assert prov["free_surface_solver3d_sha256"], "solver source hash missing"
    assert prov["tensorbem_file"], "the imported TensorBEM path is not recorded"
    # A well-posed cell: full rank against the lstsq cutoff, finite conditioning.
    assert spec["effective_rank"] == spec["n_cols"]
    assert spec["n_cols_rank_deficient"] == 0
    assert math.isfinite(spec["cond_of_solved_system"])


# --- 2. committed artefacts ------------------------------------------------
def _ladder_rows() -> dict[tuple[float, int, bool], list[tuple[str, dict]]]:
    """Every committed (ka, subdivisions, chief) row, keyed by cell, with its source."""
    cells: dict[tuple[float, int, bool], list[tuple[str, dict]]] = {}

    def add(source: str, rows) -> None:
        for row in rows:
            if not all(k in row for k in QUANTITIES):
                continue
            key = (
                round(float(row["ka"]), 6),
                int(row["subdivisions"]),
                bool(row.get("chief", True)),
            )
            cells.setdefault(key, []).append((source, row))

    results_path = CASEPOOL / "results.json"
    if results_path.is_file():
        add("results.json", json.loads(results_path.read_text())["rows"])
    for name in ("inhouse_ladder_ka2.json", "inhouse_ladder_ka2_rung4_node.json",
                 "inhouse_ladder_ka2_rung5.json"):
        path = CASEPOOL / name
        if path.is_file():
            add(name, json.loads(path.read_text())["rows"])
    return cells


def test_same_cell_artefacts_agree_or_are_registered():
    """The registry must match reality: every disagreement is known, none is new."""
    cells = _ladder_rows()
    assert cells, "no ladder artefacts found -- the inventory test would be vacuous"

    seen_divergent: set[tuple[float, int, bool]] = set()
    for key, entries in sorted(cells.items()):
        if len(entries) < 2:
            continue
        worst = 0.0
        worst_where = ""
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                (src_a, row_a), (src_b, row_b) = entries[i], entries[j]
                for q in QUANTITIES:
                    a, b = float(row_a[q]), float(row_b[q])
                    rel = abs(a - b) / max(abs(a), abs(b), 1e-300)
                    if rel > worst:
                        worst, worst_where = rel, f"{q} {src_a} vs {src_b}"
        if worst <= SAME_CELL_TOL:
            assert key not in KNOWN_CROSSENV_DIVERGENCE or KNOWN_CROSSENV_DIVERGENCE[key][
                "max_rel_delta"
            ] is None, (
                f"cell {key} now agrees across artefacts ({worst:.2e}) but is still registered "
                f"as a known cross-environment divergence -- update the registry and the doc"
            )
            continue
        assert key in KNOWN_CROSSENV_DIVERGENCE, (
            f"cell ka={key[0]} subdivisions={key[1]} chief={key[2]} differs by {worst:.3%} ({worst_where}) "
            f"between committed artefacts and is NOT registered as a known cross-environment "
            f"divergence: two files claim the same mesh and disagree about the physics"
        )
        entry = KNOWN_CROSSENV_DIVERGENCE[key]
        seen_divergent.add(key)
        assert entry["max_rel_delta"] is not None, f"cell {key} diverges but is registered as node-only"
        assert abs(worst - entry["max_rel_delta"]) <= entry["rel_tol"], (
            f"cell {key} divergence changed size: measured {worst:.4f}, registered "
            f"{entry['max_rel_delta']:.4f} +/- {entry['rel_tol']:.4f} ({worst_where})"
        )
    assert seen_divergent == {
        k for k, v in KNOWN_CROSSENV_DIVERGENCE.items() if v["max_rel_delta"] is not None
    }, "registry lists a divergence that the committed artefacts do not show"


def test_registered_divergence_is_the_documented_one():
    """Pin *which* copies disagree at ka=2/1024 -- the defect must not migrate."""
    cells = _ladder_rows()
    entries = dict((src, row) for src, row in cells[(2.0, 4, True)])
    control = entries["inhouse_ladder_ka2.json"]
    node = entries["inhouse_ladder_ka2_rung4_node.json"]
    # Same mesh claimed by both sides -- this is what makes it a reproducibility
    # defect rather than a resolution difference.
    assert control["mesh_volume"] == pytest.approx(node["mesh_volume"], rel=1e-12)
    assert control["condition_number"] == pytest.approx(node["condition_number"], rel=1e-12)
    assert control["panels"] == node["panels"] == 1024
    # The divergent quantity is the heave *imaginary* part (damping): the node
    # copy is ~half the control-plane value.
    ratio = control["B33_dimless"] / node["B33_dimless"]
    assert 1.9 < ratio < 2.2, f"registered divergence changed character: B33 ratio {ratio:.3f}"
    # ... and the node copy is the one that misses the Hulme reference by ~50%.
    ref = control["reference"]["B33_dimless"]
    assert abs(node["B33_dimless"] - ref) / ref > 0.4
    assert abs(control["B33_dimless"] - ref) / ref < 0.05

# --- 3. the localised defect, as committed evidence -------------------------
FINGERPRINT_DIR = CASEPOOL / "fingerprints"


@pytest.mark.parametrize(
    "name,expected_healthy",
    [("ctrl_ka2_s4.json", True), ("node_ka2_s4.json", False)],
)
def test_committed_fingerprints_record_the_solve_health_split(name, expected_healthy):
    """The defect is pinned by *where* it lives: same matrix, different residual."""
    path = FINGERPRINT_DIR / name
    if not path.is_file():
        pytest.skip(f"fingerprint artefact missing: {path}")
    payload = json.loads(path.read_text())
    # Read the raw residual rather than the derived health flag so that
    # fingerprints written before the flag existed stay comparable.
    residual = payload["solve_radiation"]["relative_residual"]
    healthy = residual <= 1e-2
    assert healthy is expected_healthy, (
        f"{name}: relative_residual {residual:.3e} contradicts the committed health verdict"
    )


def test_committed_fingerprints_share_the_assembled_operator():
    """If the two environments ever assemble *different* matrices, this framing dies."""
    ctrl_path = FINGERPRINT_DIR / "ctrl_ka2_s4.json"
    node_path = FINGERPRINT_DIR / "node_ka2_s4.json"
    if not (ctrl_path.is_file() and node_path.is_file()):
        pytest.skip("fingerprint artefacts missing")
    ctrl = json.loads(ctrl_path.read_text())
    node = json.loads(node_path.read_text())
    for stage in ("mesh", "operators"):
        for key in ctrl[stage]:
            if not isinstance(ctrl[stage][key], dict) or "sha256_rounded" not in ctrl[stage][key]:
                continue
            assert ctrl[stage][key]["sha256_rounded"] == node[stage][key]["sha256_rounded"], (
                f"{stage}.{key} differs between the control plane and the node: the "
                f"documented root cause (linear-solve-only divergence) no longer holds"
            )
    # ... while the solve does not agree: that is the actual defect.
    assert ctrl["solve_radiation"]["relative_residual"] < 1e-3
    assert node["solve_radiation"]["relative_residual"] > 1e-2
    assert ctrl["provenance"]["free_surface_solver3d_sha256"] == node["provenance"][
        "free_surface_solver3d_sha256"
    ]
