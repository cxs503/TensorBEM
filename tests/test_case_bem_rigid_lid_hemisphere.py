"""Gates for the pool case ``CASE-BEM-RIGID-LID-HEMISPHERE``.

What these tests defend
----------------------
The case makes three separable claims and each one gets its own gate here, so a
regression cannot hide behind the aggregate verdict:

1. **the reference**: the rigid-lid (``ka -> 0``) heave added mass of a floating
   hemisphere is ``0.8309513 * rho*nabla`` -- produced by two independent
   formulations of the even-parity double body that must agree, with a
   truncation ladder that must be converged, and with the *odd*-parity pressure
   release endpoint exactly ``0.5 rho*nabla``;
2. **the solver**: at ``ka = 1e-3`` on 64 constant panels the 3-D radiation
   solver must reproduce the committed row bit for bit (a fresh solve, not a
   re-read of ``results.json``), and no mesh/``ka`` in the endpoint band may
   approach the refuted translating-image reference ``1.0 rho*nabla``;
3. **the artifact**: the committed ``results.json``, ``report.md`` and
   ``case.json`` must be the rendering of the committed payload -- the verdict
   is recomputed from the payload in the test, and the report is checked to
   carry the payload's own numbers instead of hand-typed ones.

The case's NEGATIVE control is part of the contract: the translating-image
reading ``A33 = rho*nabla`` is refuted here by measurement (every band value is
below ``0.95``), which is what makes the ~14 % "error" of report #15809 a
reference defect rather than a solver defect
(``docs/rigid_lid_zero_frequency_anchor/ERRATUM.md``).
"""

from __future__ import annotations

import json
import math
import pathlib
import subprocess
import sys

import pytest

from tensorbem.benchmarks import case_bem_rigid_lid_hemisphere as case

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "benchmarks" / "verified" / "case_bem_rigid_lid_hemisphere"
REPORT = ROOT / "docs" / "case_bem_rigid_lid_hemisphere.md"


@pytest.fixture(scope="module")
def committed() -> dict:
    return json.loads((ARTIFACT / "results.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. the reference
# ---------------------------------------------------------------------------
def test_two_routes_of_the_double_body_agree() -> None:
    reference = case.rigid_lid_reference(degree_max=60)
    energy = reference["routes"]["energy"]["dimless"]
    force = reference["routes"]["force"]["dimless"]
    assert 0.83 < energy < 0.832, energy
    assert abs(force - energy) / energy < 1.0e-4
    # the same double body also has to reproduce the two exact sub-limits
    assert reference["pressure_release_dimless"] == pytest.approx(0.5, abs=1.0e-12)
    assert (
        reference["routes"]["energy"]["moment_closed_form_residual"] < 1.0e-13
    ), "exact Gauss moments must match J_0 = 1/2, J_1 = 1/3, J_2 = 1/8"


def test_truncation_ladder_is_converged() -> None:
    reference = case.rigid_lid_reference(degree_max=60)
    values = [item["value_dimless"] for item in reference["truncation_ladder"]]
    assert values == sorted(values), "even-mode partial sums must increase monotonically"
    assert values[-1] - values[-2] < 1.0e-7
    assert values[-1] - values[0] < 1.0e-5


def test_hulme_extrapolation_corroborates_the_endpoint() -> None:
    hulme = case.hulme_zero_frequency_extrapolation()
    reference = case.rigid_lid_reference(degree_max=60)["value_dimless"]
    assert abs(hulme["limit_dimless"] / reference - 1.0) < 1.0e-2
    # the multipole series must also be converged in its own term count
    assert hulme["truncation_spread_at_smallest_ka"] < 1.0e-4


# ---------------------------------------------------------------------------
# 2. the solver
# ---------------------------------------------------------------------------
def test_solver_row_reproduces_the_committed_number() -> None:
    reference = case.rigid_lid_reference(degree_max=60)
    rows = case.solver_rows(2, (1.0e-3,), reference)
    committed = json.loads((ARTIFACT / "results.json").read_text(encoding="utf-8"))
    expected = next(
        row
        for level in committed["levels"]
        for row in level["rows"]
        if row["panels"] == rows[0]["panels"] and row["ka"] == rows[0]["ka"]
    )
    assert rows[0]["added_mass_dimless"] == pytest.approx(
        expected["added_mass_dimless"], rel=1.0e-9
    )
    assert abs(rows[0]["relative_error"]) < case.TOLERANCE


def test_no_band_value_approaches_the_translating_image() -> None:
    reference = case.rigid_lid_reference(degree_max=60)
    rows = case.solver_rows(2, case.ENDPOINT_BAND_KA, reference)
    values = [row["added_mass_dimless"] for row in rows]
    assert max(values) < 0.95, values
    # ... and the refuted reference is what the pool gloss invites: 1.0
    assert min(row["relative_error_vs_naive_image"] for row in rows) < -0.10


# ---------------------------------------------------------------------------
# 3. the artifacts
# ---------------------------------------------------------------------------
def test_committed_verdict_is_reproduced_from_the_payload(committed: dict) -> None:
    recomputed = case.verdict(committed)
    assert recomputed == committed["verdict"]
    assert recomputed["passed"] is True, recomputed["failed_checks"]
    # every G1 line must be the row error it names, not a remembered constant
    finest = max(committed["levels"], key=lambda item: item["level"]["panels"])
    for row in finest["rows"]:
        name = "G1_band_abs_error@ka=%.0e" % row["ka"]
        check = next(item for item in recomputed["checks"] if item["check"] == name)
        assert check["value"] == pytest.approx(abs(row["relative_error"]), rel=1.0e-12)


def test_committed_reference_matches_a_fresh_evaluation(committed: dict) -> None:
    fresh = case.rigid_lid_reference(degree_max=60)
    assert committed["reference"]["value_dimless"] == pytest.approx(
        fresh["value_dimless"], rel=1.0e-12
    )
    assert committed["reference"]["value_kg"] == pytest.approx(
        fresh["value_kg"], rel=1.0e-12
    )


def test_report_is_rendered_from_the_payload(committed: dict) -> None:
    report = (ARTIFACT / "report.md").read_text(encoding="utf-8")
    assert report == case.render_markdown(committed) + "\n"
    # the pointer document must exist and quote the repro command, not numbers
    pointer = REPORT.read_text(encoding="utf-8")
    assert "repro_case_bem_rigid_lid_hemisphere.sh" in pointer
    assert "Case card" in pointer


def test_case_card_carries_a_runnable_repro_path(committed: dict) -> None:
    card = json.loads((ARTIFACT / "case.json").read_text(encoding="utf-8"))
    assert card["case_id"] == case.CASE_ID
    assert card["repro_cmd"].startswith("bash scripts/repro_case_bem_rigid_lid_hemisphere.sh")
    script = ROOT / card["repro_cmd"].split()[1]
    assert script.exists(), script
    assert card["verdict_passed"] is True
    assert card["reference"]["value_dimless"] == pytest.approx(
        committed["reference"]["value_dimless"], rel=1.0e-12
    )


def test_quick_ladder_renders_all_artifacts(tmp_path: pathlib.Path) -> None:
    result = case.run_case(mesh_levels=(1,), with_band=False, with_sweeps=False)
    (tmp_path / "results.json").write_text(json.dumps(result), encoding="utf-8")
    (tmp_path / "report.md").write_text(case.render_markdown(result), encoding="utf-8")
    case.write_case_card(result, tmp_path / "case.json")
    figures = case.write_figures(result, tmp_path / "figures")
    assert figures and figures[0].stat().st_size > 1024
    assert "## 3. Reference vs computed" in case.render_markdown(result)
    # a 16-panel mesh cannot pass the geometry gate: the verdict must say so
    assert result["levels"][0]["level"]["panels"] == 16
    assert result["verdict"]["passed"] is False
    assert any(
        check["check"].startswith("G0_") for check in result["verdict"]["failed_checks"]
    )


def test_module_help_exits_zero() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "tensorbem.benchmarks.case_bem_rigid_lid_hemisphere", "--help"],
        capture_output=True,
        text=True,
        env={
            **__import__("os").environ,
            "PYTHONPATH": str(ROOT / "src"),
        },
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--check" in completed.stdout