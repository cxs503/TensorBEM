"""Executable acceptance criterion for the heavy tier of the declared-number ledger.

The five heavy-tier entries (ellipsoid 1280-panel added-mass errors on the three
axes, Ursell 256-panel added-mass and damping errors) were registered in
``tensorbem.benchmarks.declared_numbers`` but never directly reproduced: a
1280-panel three-dimensional assembly costs tens of minutes, so the m280
verification audit deferred them to "a later seat with budget".

Mission m2845 ran the heavy tier end to end (commit 29236f0, numpy 2.4.6,
loongarch64) and committed the machine-readable evidence to
``artifacts/declared-numbers-heavy.json``.  Re-running the benchmarks is
expensive, so this test pins the *evidence* instead: it re-judges every row of
the committed artifact against the registry without launching a single
benchmark.  If a future code change drifts a heavy number, the correct
response is to re-run the heavy tier and refresh the artifact -- the drift
will then show up here as a red test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorbem.benchmarks.declared_numbers import (
    DECLARED_NUMBERS,
    _judge,
)

ARTIFACT = Path(__file__).resolve().parents[1] / "docs" / "heavy_tier_evidence" / "declared-numbers-heavy.json"

HEAVY_KEYS = (
    "ellipsoid_added_mass_1280_x",
    "ellipsoid_added_mass_1280_y",
    "ellipsoid_added_mass_1280_z",
    "ursell_added_mass_256",
    "ursell_damping_256",
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    assert ARTIFACT.is_file(), (
        f"missing heavy-tier evidence artifact {ARTIFACT}; regenerate it with "
        "python -m tensorbem.benchmarks.declared_numbers --tiers heavy "
        f"--output {ARTIFACT}"
    )
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_artifact_covers_exactly_the_heavy_tier(artifact):
    rows = artifact["checks"]
    assert [row["key"] for row in rows] == list(HEAVY_KEYS)
    registry_heavy = {e.key for e in DECLARED_NUMBERS if e.tier == "heavy"}
    assert {row["key"] for row in rows} == registry_heavy


def test_all_rows_pass(artifact):
    for row in artifact["checks"]:
        assert row["verdict"] == "PASS", (
            f"{row['key']}: measured {row['measured']:.6e} vs declared "
            f"{row['declared']:.6e} (deviation {row['relative_deviation']:.3e})"
        )
    assert artifact["summary"]["verdict"] == "PASS"
    assert artifact["summary"]["drifted"] == 0


def test_measured_values_rejudge_to_pass(artifact):
    """Independently re-judge each row from the registry, not from the artifact."""
    registry = {e.key: e for e in DECLARED_NUMBERS}
    for row in artifact["checks"]:
        entry = registry[row["key"]]
        deviation, verdict = _judge(entry, row["measured"])
        assert verdict == "PASS", f"{row['key']}: re-judged {verdict}"
        assert abs(deviation - row["relative_deviation"]) < 1e-12


def test_heavy_errors_are_below_three_percent(artifact):
    """Platform acceptance bar: every heavy quantity within 3% of its reference."""
    for row in artifact["checks"]:
        assert row["measured"] < 3.0e-2, (
            f"{row['key']}: error {row['measured']:.3e} exceeds the 3% bar"
        )


def test_worst_deviation_is_rounding_level(artifact):
    """The declared numbers are quoted to 5 significant figures, so a faithful
    re-computation must agree to well below 0.1% -- anything larger means the
    artifact was produced by different code than the registry."""
    worst = artifact["summary"]["worst_relative_deviation"]
    assert worst < 1.0e-3


def test_provenance_is_recorded(artifact):
    env = artifact["environment"]
    for field in ("python", "numpy", "platform", "commit"):
        assert env.get(field), f"environment provenance missing {field!r}"
    assert env["commit"], "commit provenance must not be empty"


def test_ursell_256_converges_toward_reference(artifact):
    """Refining 128 -> 256 panels must move the absolute added mass and damping
    strictly closer to the Ursell multipole reference (the panel family
    converges to the reference from above)."""
    fine = json.loads((ARTIFACT.parent / "ursell-256.json").read_text(encoding="utf-8"))
    coarse = json.loads((ARTIFACT.parent / "ursell-128.json").read_text(encoding="utf-8"))
    for key in ("panel_added_mass", "panel_damping"):
        reference = fine["reference_solution_added_mass"] if "mass" in key else fine["reference_solution_damping"]
        assert abs(fine[key] - reference) < abs(coarse[key] - reference), (
            f"{key}: 256-panel value {fine[key]:.4f} is not closer to the "
            f"reference {reference:.4f} than the 128-panel value {coarse[key]:.4f}"
        )
