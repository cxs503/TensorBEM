"""Environment provenance of the in-house Hulme ladder, and the E10 gate.

Background
----------
``docs/case_bem_hulme_inhouse_ladder_env_dependence.md`` (m21674) reported the
4096-panel rung as a *scale breakdown* and a 2.05x heave-damping difference as a
*numpy version drift*.  Both readings rest on one assumption that no artifact
recorded: that the rungs being compared were produced by the same code.  The
node rungs were written 2026-11-14, the solve-stage guard that catches a wrong
complex solve landed 2026-11-15, and neither file carries a revision -- so the
attribution could not be audited after the fact.

What is asserted here (all cheap, no 1024-panel solve):

1. a ladder row now carries the code revision that produced it, and a merged
   ladder summarises the numeric environment of every rung *and* replicate;
2. two rows for the same panel count are accepted only when they come from
   different numeric environments, and are then stored as ``replicates`` (a
   cross-environment control) rather than as a second refinement step; a genuine
   duplicate is still an error;
3. ``environment_consistency`` admits a mixed ladder only when an overlapping
   panel count was recomputed in both environments and agrees to rounding;
4. the E10 gate is present in the analysis payload and reports the committed
   ladder honestly (it is mixed, and the overlap that would control the mixture
   does not exist yet);
5. the drift probe's kernel fingerprints are deterministic, so a cross-machine
   comparison cannot be confounded by the instrument itself.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CASEPOOL = REPO / "benchmarks" / "verified" / "casepool_bem_hulme_hemisphere"
QUANTITIES = ("A11_dimless", "B11_dimless", "A33_dimless", "B33_dimless")


def _load_script(name: str):
    path = REPO / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ladder_module():
    return _load_script("hulme_hemisphere_inhouse_ladder.py")


def _row(panels: int, sub: int, numpy: str, revision: str, host: str, b33: float) -> dict:
    return {
        "subdivisions": sub,
        "panels": panels,
        "host": host,
        "numpy": numpy,
        "revision": revision,
        "condition_number": 2.3986,
        "radiation_relative_residual": 1.0e-9,
        **{q: 1.0 + 0.1 * i for i, q in enumerate(QUANTITIES)},
        "B33_dimless": b33,
    }


def _artifact(path: Path, rows: list[dict], *, numpy: str, revision: str, host: str) -> Path:
    payload = {
        "case_code": "BEM-HULME-HEMISPHERE",
        "ka": 2.0,
        "quantities": list(QUANTITIES),
        "environment": {"host": host, "numpy": numpy, "revision": {"commit": revision}},
        "rows": rows,
        "failures": [],
    }
    path.write_text(json.dumps(payload))
    return path


def test_merged_ladder_records_replicates_instead_of_a_second_rung(ladder_module, tmp_path):
    control = _artifact(
        tmp_path / "control.json",
        [_row(1024, 4, "2.4.6", "revA", "control", 0.104249)],
        numpy="2.4.6",
        revision="revA",
        host="control",
    )
    node = _artifact(
        tmp_path / "node.json",
        [_row(1024, 4, "2.0.1", "revA", "node315", 0.1042490000001)],
        numpy="2.0.1",
        revision="revA",
        host="node315",
    )
    merged = ladder_module.merge_ladders(control, node)
    assert len(merged["rows"]) == 1, "a replicate must not become a refinement step"
    row = merged["rows"][0]
    assert len(row["replicates"]) == 1
    rep = row["replicates"][0]
    assert rep["numpy"] == "2.0.1" and rep["host"] == "node315"
    assert rep["delta_vs_rung"]["B33_dimless"] < 1e-9
    env = merged["environment_consistency"]
    assert env["is_homogeneous"] is False
    assert env["numpy"] == ["2.4.6"] and env["replicate_numpy"] == ["2.0.1"]
    assert env["admissible"] is True, env["reason"]


def test_merge_rejects_a_genuine_duplicate(ladder_module, tmp_path):
    a = _artifact(
        tmp_path / "a.json",
        [_row(1024, 4, "2.4.6", "revA", "control", 0.104249)],
        numpy="2.4.6",
        revision="revA",
        host="control",
    )
    with pytest.raises(ValueError, match="duplicate rung"):
        ladder_module.merge_ladders(a, a)


def test_mixed_ladder_without_overlap_is_inadmissible():
    from tensorbem.benchmarks.hemisphere_independent_convergence import (
        environment_consistency,
    )

    rows = [
        {"panels": 256, "numpy": "2.4.6", "revision": "revA", "host": "control"},
        {"panels": 4096, "numpy": "2.0.1", "revision": "revA", "host": "node330"},
    ]
    env = environment_consistency(rows)
    assert env["is_homogeneous"] is False
    assert env["overlaps"] == []
    assert env["admissible"] is False
    assert "recomputed" in env["reason"]


def test_mixed_ladder_is_admissible_only_when_the_overlap_agrees():
    from tensorbem.benchmarks.hemisphere_independent_convergence import (
        ENV_AGREEMENT_BAND,
        environment_consistency,
    )

    def rows(delta: float) -> list[dict]:
        return [
            {
                "panels": 1024,
                "numpy": "2.4.6",
                "revision": "revA",
                "host": "control",
                "replicates": [
                    {
                        "host": "node315",
                        "numpy": "2.0.1",
                        "revision": "revA",
                        "delta_vs_rung": {"B33_dimless": delta},
                    }
                ],
            },
            {"panels": 4096, "numpy": "2.0.1", "revision": "revA", "host": "node330"},
        ]

    controlled = environment_consistency(rows(1.0e-13))
    assert controlled["admissible"] is True, controlled["reason"]
    assert controlled["worst_replicate_delta"] == pytest.approx(1.0e-13)

    drifting = environment_consistency(rows(0.51))
    assert drifting["admissible"] is False
    assert drifting["worst_replicate_delta"] == pytest.approx(0.51)
    assert 0.51 > ENV_AGREEMENT_BAND


def test_e10_gate_reports_the_committed_ladder_honestly():
    from tensorbem.benchmarks.hemisphere_independent_convergence import analyse

    payload = analyse()
    gate = payload["gates"]["E10_ladder_environment_homogeneous"]
    ladder = json.loads((CASEPOOL / "inhouse_ladder_ka2.json").read_text())
    hosts = {str(p["host"]) for p in ladder.get("provenance", []) if p.get("host")}
    # The committed artifact mixes the control plane with node330 (file-level
    # provenance) and carries no overlapping panel count, so E10 must fail --
    # and it must say *why*, not merely that the ladder is not homogeneous.
    assert hosts == {"ec6ae0b42a0c", "node330"}, hosts
    assert gate["is_homogeneous"] is False
    assert gate["overlaps"] == []
    assert gate["pass"] is False, "a mixed ladder with no overlap control must not pass E10"
    # the rungs themselves predate per-row environment provenance: the gate has
    # to disclose that they cannot testify, not quietly treat them as one build
    assert gate["n_rungs_unknown_numpy"] >= 1
    assert gate["n_rungs_unknown_revision"] >= 1


def test_drift_probe_kernel_fingerprints_are_deterministic():
    probe = _load_script("hulme_numpy_drift_probe.py")
    first = probe.kernel_fingerprint()
    second = probe.kernel_fingerprint()
    for key in (
        "bessel_j0_sha",
        "bessel_y1_sha",
        "struve_h0_sha",
        "wave_function_F_sha",
        "laguerre_nodes_sha",
        "gauss_legendre_nodes_sha",
    ):
        assert first[key] == second[key], f"{key} is not reproducible on one machine"
    # the probe must be informative: the wave integral is a non-trivial number
    assert all(abs(v) > 0.0 for v in first["wave_function_F_probe"])