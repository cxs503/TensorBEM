"""Consistency gates for the m3181 three-dimensional free-surface deliverable.

The milestone ships four artefacts that quote the same numbers:

* the registry ``tensorbem.benchmarks.declared_numbers`` (groups
  ``hulme_hemisphere`` / ``hulme_chief`` / ``submerged_sphere``),
* the audit JSON produced by a real run (``docs/m3181/outputs/...json``),
* the generated report (``dev_report_m3181_three_d_free_surface_registry.md``),
* the three figures (``docs/m3181/figures/*.png``).

Nothing here re-runs a solver: the recomputation itself is guarded by
``tests/test_declared_numbers.py`` in the default ``pytest`` run. These tests
guard that the four artefacts cannot drift *from each other* -- the failure mode
that makes a report quote a number the code no longer produces. The report test
regenerates the markdown from the JSON byte-for-byte, so a hand-edited value in
the report is a red test rather than a silent lie.
"""

from __future__ import annotations

import importlib.util
import json
import math
import struct
from pathlib import Path

import pytest

from tensorbem.benchmarks import declared_numbers as registry

ROOT = Path(__file__).resolve().parents[1]
M3181 = ROOT / "docs" / "m3181"
AUDIT_JSON = M3181 / "outputs" / "three_d_free_surface_registry.json"
REPORT = ROOT / "dev_report_m3181_three_d_free_surface_registry.md"
FIGURES = M3181 / "figures"

GROUPS = ("hulme_hemisphere", "hulme_chief", "submerged_sphere")
FIGURE_STEMS = (
    "fig_registry_3d_free_surface",
    "fig_three_d_convergence_registry",
    "fig_chief_regularization",
)
THRESHOLD = 3.0e-2


def _load_script(name: str):
    """Import ``docs/m3181/<name>.py`` without adding it to the package."""
    path = M3181 / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"m3181_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def payload() -> dict:
    assert AUDIT_JSON.exists(), f"missing audit artefact: {AUDIT_JSON}"
    return json.loads(AUDIT_JSON.read_text(encoding="utf-8"))


def _registry_rows() -> list:
    return [entry for entry in registry.DECLARED_NUMBERS if entry.group in GROUPS]


# --- the registry side -----------------------------------------------------


def test_the_three_documented_tables_are_all_registered():
    """Tables 10-12 of the benchmark document must be fully registered."""
    rows = _registry_rows()
    per_group = {group: [row for row in rows if row.group == group] for group in GROUPS}
    assert {group: len(found) for group, found in per_group.items()} == {
        "hulme_hemisphere": 31,
        "hulme_chief": 6,
        "submerged_sphere": 25,
    }
    keys = [row.key for row in rows]
    assert len(keys) == len(set(keys)), "duplicate registry keys"
    for row in rows:
        assert row.tier in registry.TIERS
        assert "section 4.1" in row.source, row.key
        assert "marine_bem_benchmarks.md" in row.source, row.key
        assert row.tolerance > 0.0


def test_every_registered_cell_names_the_same_group_as_the_audit_script():
    driver = _load_script("run_three_d_free_surface")
    assert tuple(driver.GROUPS) == GROUPS
    assert set(driver.TABLE_OF_GROUP) == set(GROUPS)


# --- the artefact side -----------------------------------------------------


def test_audit_json_is_cell_for_cell_the_registry(payload):
    """No hand editing: the stored JSON must be the registry, recomputed once."""
    assert payload["audit"] == "m3181_three_d_free_surface_registry"
    rows = payload["registry"]["checks"]
    registered = {row.key: row for row in _registry_rows()}
    assert {row["key"] for row in rows} == set(registered)
    for row in rows:
        entry = registered[row["key"]]
        assert row["group"] == entry.group
        assert row["declared"] == pytest.approx(entry.declared, rel=0.0, abs=0.0)
        assert row["tolerance"] == entry.tolerance
        assert row["comparison"] == entry.kind
        assert row["verdict"] == "PASS", row["key"]
        assert math.isfinite(row["measured"])
        assert abs(row["relative_deviation"]) <= row["tolerance"], row["key"]
    summary = payload["registry"]["summary"]
    assert summary["checked"] == len(rows)
    assert summary["drifted"] == 0
    assert summary["worst_relative_deviation"] < 1.5e-2


def test_audit_json_is_strict_json():
    text = AUDIT_JSON.read_text(encoding="utf-8")
    for token in ("NaN", "Infinity"):
        assert token not in text, f"non-strict JSON token {token!r} in the audit artefact"


def test_recorded_studies_cover_the_three_solves(payload):
    """Three parametric settings, i.e. three solves for sixty-two quoted numbers."""
    assert len(payload["studies"]) == 3
    for group in GROUPS:
        assert any(key.startswith(group + "|") for key in payload["studies"])


def _levels(payload: dict, group: str) -> list[dict]:
    builder = _load_script("build_report")
    return builder._levels(payload, group)  # noqa: SLF001 - same helper under test


@pytest.mark.parametrize("group", ["hulme_hemisphere", "submerged_sphere"])
def test_refinement_is_monotone_in_every_registered_error(payload, group):
    levels = _levels(payload, group)
    assert [int(level["panel_count"]) for level in levels] == sorted(
        int(level["panel_count"]) for level in levels
    )
    assert len(levels) == 3
    for key in (
        "heave_added_mass_rel_error",
        "heave_damping_rel_error",
        "surge_added_mass_rel_error",
        "excitation_force_z_rel_error",
    ):
        errors = [level[key] for level in levels]
        assert all(0.0 < error < 1.0 for error in errors), (group, key, errors)
        assert errors == sorted(errors, reverse=True), (group, key, errors)
        # a working BEM has to do better than a factor 2 per level of 4x panels
        assert errors[0] / errors[-1] > 4.0, (group, key, errors)


def test_observed_orders_are_second_order(payload):
    for group in ("hulme_hemisphere", "submerged_sphere"):
        for level in _levels(payload, group)[1:]:
            orders = [value for key, value in level.items() if key.startswith("observed_order_")]
            assert orders, (group, level.get("panel_count"))
            assert all(1.5 < order < 2.4 for order in orders), (group, orders)


def test_chief_regularisation_flips_the_heave_damping(payload):
    values = payload["studies"]["hulme_chief|" + json.dumps(
        {"ka_values": [2.58], "subdivision": 3}, sort_keys=True
    )]
    assert values["ka2.58_unregularized_b33_dimless"] < 0.0, "the non-passive branch is the point"
    assert values["ka2.58_chief_regularized_b33_dimless"] > 0.0
    assert (
        values["ka2.58_unregularized_condition_number"]
        / values["ka2.58_chief_regularized_condition_number"]
        > 10.0
    )
    assert int(values["ka2.58_chief_regularized_chief_point_count"]) >= 1


# --- the report side -------------------------------------------------------


def test_report_is_generated_from_the_audit_json(payload):
    """The committed report must be exactly what the builder emits (no hand edits)."""
    builder = _load_script("build_report")
    generated = builder.build(payload)
    committed = REPORT.read_text(encoding="utf-8")
    if generated != committed:
        pytest.fail(
            "dev_report_m3181_three_d_free_surface_registry.md is stale; "
            "re-run PYTHONPATH=src python3 docs/m3181/build_report.py"
        )
    assert builder.BEGIN in committed and builder.END in committed


def test_report_states_the_threshold_verdict_for_every_quantity(payload):
    """The six-element report contract demands an explicit "<3%" answer per item."""
    builder = _load_script("build_report")
    rows = builder._reference_rows(payload)
    text = REPORT.read_text(encoding="utf-8")
    for row in rows:
        marker = f"| {row['table']} | {row['label']} | {row['panels']} |"
        assert marker in text, row["label"]
    failing = [row for row in rows if row["error"] >= THRESHOLD]
    passing = [row for row in rows if row["error"] < THRESHOLD]
    # the honest-failure list is data driven: it must name exactly the failures
    assert f"{len(passing)} / {len(rows)}" in text
    for row in failing:
        assert f"{row['table']} 的 {row['label']}" in text, row["label"]
    assert "**否**" in text
    assert "不成立" in text, "the report must not claim a blanket '<3%' pass"


def test_report_quotes_no_deviation_it_did_not_measure(payload):
    """Every per-cell deviation printed in the report exists in the JSON."""
    text = REPORT.read_text(encoding="utf-8")
    for row in payload["registry"]["checks"]:
        line = next(
            (ln for ln in text.splitlines() if ln.startswith(f"| `{row['key']}` |")), None
        )
        assert line is not None, row["key"]
        assert f"{row['relative_deviation']:+.2e}" in line, row["key"]
        assert row["verdict"] in line


# --- the figure side -------------------------------------------------------


def _png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()[:33]
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name} is not a PNG"
    assert data[12:16] == b"IHDR"
    width, height = struct.unpack(">II", data[16:24])
    return width, height


@pytest.mark.parametrize("stem", FIGURE_STEMS)
def test_figures_exist_and_are_readable(stem):
    full = FIGURES / f"{stem}.png"
    small = FIGURES / f"{stem}_small.png"
    for path in (full, small):
        assert path.exists(), path
        width, height = _png_size(path)
        assert width >= 600 and height >= 300, (path.name, width, height)
        assert path.stat().st_size > 8_000, path.name
    assert full.stat().st_size > small.stat().st_size, "the preview must be the smaller file"


def test_figures_are_drawn_from_the_audit_json():
    """The plotting script must read the JSON and never invent values."""
    source = (M3181 / "make_figures.py").read_text(encoding="utf-8")
    assert "three_d_free_surface_registry.json" in source
    assert "GROUPS = (" in source
    assert "--input" in source
    for group in GROUPS:  # every group reaches at least one figure
        assert group in source, group