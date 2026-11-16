"""Round-8 guards: is the flat-panel error the spherical-patch area deficit?

Anchor ``CASE-BEM-SUBMERGED-SPHERE-HAVELOCK`` (ref Havelock 1955 frequency-domain multipole,
tolerance < 5 %).  Round 6 registered the "err proportional to the area deficit" mechanism and
P1/P2/P3 but could only test it on committed *flat* data; round 7 wrote the per-ka T1/T2 rows.
These guards assert that ``scripts/havelock_round8_adjudicate.py`` derives its verdicts from
those rows under the predeclared rules (R-P1 / R-P2 / R-P3 / R-A4 for the case mesh when it
exists), and that nothing published before is repainted on the way.

T3 (1280 panels) is *not* measured in this round: its block is absent, ``not_run`` says so, and
the tests skip rather than pretend.  Hand-written numbers must never be able to turn these green.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "docs" / "case_bem_submerged_sphere_havelock"
ROWS = CASE_DIR / "measure_correction" / "numbers_t1t2_rows.json"
CASE_T3 = CASE_DIR / "measure_correction" / "numbers_round6.json"
MECHANISM = ROOT / "docs" / "case_bem_submerged_sphere_havelock_round6" / "mechanism_scaling.json"
GATED = CASE_DIR / "numbers.json"
PUBLISHED = CASE_DIR / "measure_correction" / "verdicts.json"
ADJ = ROOT / "docs" / "case_bem_submerged_sphere_havelock_round8" / "adjudicate.json"

COEFFICIENTS = ("a33", "b33", "a11", "b11")
DAMPING = ("b33", "b11")
GATE = 0.05
FACTOR_BAND = (0.5, 2.0)
TIER_KEYS = {"T1": "T1:economy_s2_band", "T2": "T2:coarsest_s1_band"}


def _load(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"round-8 artefact missing ({path.relative_to(ROOT)})")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def adj() -> dict:
    return _load(ADJ)


@pytest.fixture(scope="module")
def rows_by_tier() -> dict[str, list[dict]]:
    doc = _load(ROWS)
    return {label: doc["tiers"][key]["rows"] for label, key in TIER_KEYS.items()}


@pytest.fixture(scope="module")
def deficits() -> dict[int, float]:
    mech = _load(MECHANISM)
    return {int(k): float(v) for k, v in mech["deficit_by_subdivision"].items()}


def test_measured_tiers_are_covered_and_not_duplicated(adj: dict, rows_by_tier: dict) -> None:
    assert set(TIER_KEYS) <= set(adj["tiers"])
    for label, rows in rows_by_tier.items():
        cells = {(round(float(r["ka"]), 6), r["measure"]) for r in rows}
        assert len(cells) == len(rows), f"{label}: duplicated (ka, measure) cell"
        assert {r["measure"] for r in rows} == {"flat", "patch", "uniform"}
        assert {int(r["n_panels"]) for r in rows} == {int(adj["tiers"][label]["n_panels"])}


def test_measured_band_max_is_recomputable(adj: dict, rows_by_tier: dict) -> None:
    for label, rows in rows_by_tier.items():
        for measure in ("flat", "patch", "uniform"):
            expected = max(float(r["band_max_rel_error"]) for r in rows if r["measure"] == measure)
            stored = adj["tiers"][label]["measured_band_max"][measure]["band_max_rel_error"]
            assert float(stored) == pytest.approx(expected, rel=1e-12), (label, measure)


def test_rp1_halving_rule_is_applied(adj: dict, rows_by_tier: dict) -> None:
    for label, rows in rows_by_tier.items():
        max_flat = max(float(r[f"{c}_rel_error"]) for r in rows if r["measure"] == "flat" for c in COEFFICIENTS)
        max_patch = max(float(r[f"{c}_rel_error"]) for r in rows if r["measure"] == "patch" for c in COEFFICIENTS)
        assert bool(adj["tiers"][label]["R_P1"]["holds"]) == bool(max_patch * 2.0 <= max_flat)


def test_rp2_deficit_proportionality_follows_from_rows(adj: dict, rows_by_tier: dict, deficits: dict) -> None:
    for label, rows in rows_by_tier.items():
        block = adj["tiers"][label]
        deficit = float(block["R_P2"]["deficit"])
        assert deficit == pytest.approx(deficits[int(rows[0]["subdivision"])], rel=1e-12)
        by_ka: dict[float, dict[str, dict[str, float]]] = {}
        for r in rows:
            by_ka.setdefault(round(float(r["ka"]), 6), {})[r["measure"]] = {
                c: float(r[f"{c}_rel_error"]) for c in COEFFICIENTS
            }
        expected = {
            (ka, c): (cell["flat"][c] - cell["patch"][c]) / deficit
            for ka, cell in by_ka.items()
            for c in DAMPING
        }
        for row in block["R_P2"]["rows"]:
            key = (round(float(row["ka"]), 6), row["quantity"])
            assert float(row["reduction_frac_of_deficit"]) == pytest.approx(expected[key], rel=1e-9)
        assert bool(block["R_P2"]["holds"]) == all(v > 0.5 for v in expected.values())
        assert bool(block["R_P2"]["holds"]) == (not block["R_P2"]["counterexamples"])


def test_rp3_uniform_control_verdict_follows_from_rows(adj: dict, rows_by_tier: dict) -> None:
    for label, rows in rows_by_tier.items():
        block = adj["tiers"][label]
        n_beats = sum(
            1
            for r_patch in rows
            if r_patch["measure"] == "patch"
            for r_unif in rows
            if r_unif["measure"] == "uniform"
            and round(float(r_unif["ka"]), 6) == round(float(r_patch["ka"]), 6)
            for c in DAMPING
            if float(r_patch[f"{c}_rel_error"]) <= float(r_unif[f"{c}_rel_error"])
        )
        assert int(block["R_P3"]["n_patch_beats_uniform"]) == n_beats
        assert bool(block["R_P3"]["holds"]) == bool(n_beats > len(DAMPING) * len(block["band"]) / 2.0)


def test_gate_unmodified_and_verdicts_recomputed(adj: dict, rows_by_tier: dict) -> None:
    assert float(adj["tolerance"]) == pytest.approx(GATE)
    for label, rows in rows_by_tier.items():
        for measure, entry in adj["tiers"][label]["verdicts_per_measure"].items():
            sel = [r for r in rows if r["measure"] == measure]
            n_bad = sum(1 for r in sel for c in COEFFICIENTS if float(r[f"{c}_rel_error"]) >= GATE)
            assert int(entry["n_items_over_gate"]) == n_bad
            assert entry["verdict"] == ("PASS" if n_bad == 0 else "FAIL")


def test_published_and_gated_results_are_not_repainted(adj: dict) -> None:
    gated = _load(GATED)
    assert gated["verdict"]["case_mesh"] == "PASS"
    assert gated["verdict"]["economy_mesh"] == "FAIL"  # honest FAIL kept
    assert adj["no_softening_A5"]["untouched"]
    published = _load(PUBLISHED)
    t1 = published["verdicts"]["T1:economy_s2_band"]
    assert t1["flat"]["verdict"] == "FAIL" and int(t1["flat"]["n_items_over_gate"]) == 7
    assert t1["patch"]["verdict"] == "PASS"
    mechanism = _load(MECHANISM)
    assert mechanism["status"] == "prediction_registered_pending_measurement"


def test_case_mesh_block_is_measured_or_declared_not_run(adj: dict) -> None:
    """T3 is either adjudicated from real rows or honestly listed as not_run -- never faked."""
    if CASE_T3.exists():
        rows = json.loads(CASE_T3.read_text())["tiers"]["T3:case_s3_band_flat_patch_uniform"]["rows"]
        assert int(rows[0]["n_panels"]) == 1280
        block = adj["tiers"]["T3"]
        max_patch = max(float(r[f"{c}_rel_error"]) for r in rows if r["measure"] == "patch" for c in COEFFICIENTS)
        assert float(block["measured_band_max"]["patch"]["band_max_rel_error"]) == pytest.approx(max_patch, rel=1e-12)
        per = block["R_A4_numeric_prediction"]["per_coefficient"]
        for c, entry in per.items():
            ratio = float(entry["measured"]) / float(entry["predicted"])
            assert float(entry["measured_over_predicted"]) == pytest.approx(ratio, rel=1e-12)
            assert bool(entry["within_factor_2"]) == bool(FACTOR_BAND[0] <= ratio <= FACTOR_BAND[1])
    else:
        assert "T3" not in adj["tiers"]
        assert any(item.startswith("T3 case-mesh") for item in adj["not_run"])