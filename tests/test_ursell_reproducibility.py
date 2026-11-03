"""Reproducibility lock for the Ursell submerged-cylinder benchmark.

The 2026-10-24 expansion audit flagged an honest FAIL: the documented 128-panel
added-mass error (7.7e-3) did not reproduce (8.51e-3). The follow-up audit in
``docs/ursell_provenance_audit.md`` traced the discrepancy to the documentation,
not the solver: the reference is truncation-saturated, the quadrature is
node-insensitive, the panel solution converges at first order to the multipole
reference, and a Richardson extrapolation of 256/512 panels lands within 8e-5 of
it. These tests pin the reproduced numbers and the corrected documentation, so
that any future change of the kernel, the mesh or the quadrature turns red here
instead of silently drifting from the published tables.

Only the cheap layers are tested (64 and 128 panels); the 256/512-panel evidence
lives in the audit document.
"""

import math
import re
from pathlib import Path

import pytest

from tensorbem.analytic import ursell_cylinder_added_mass, ursell_cylinder_coefficients
from tensorbem.benchmarks.ursell_cylinder import submerged_circle_mesh
from tensorbem.free_surface_solver import (
    far_field_damping,
    radiation_reference,
    solve_radiation_indirect,
)

RADIUS, SUBMERGENCE, KA = 1.0, 2.0, 0.6
DENSITY, GRAVITY = 1025.0, 9.80665
OMEGA = math.sqrt(KA * GRAVITY / RADIUS)
WAVENUMBER = KA / RADIUS
EXACT_ADDED_MASS, EXACT_DAMPING = ursell_cylinder_added_mass(
    RADIUS, SUBMERGENCE, OMEGA, DENSITY, gravity=GRAVITY
)

# Reproduced values, docs/ursell_provenance_audit.md §3.3 / §3.4.
LOCKED = {
    64: {"added_mass": 2587.762743068625, "damping": 2787.2367800364336,
         "near_over_far": 1.0105357109796724},
    128: {"added_mass": 2566.8343342915696, "damping": 2754.470100002123,
          "near_over_far": 1.0053413740188557,
          "mfs_added_mass": 2544.6864273460715,
          "mfs_damping": 2715.7555545010223},
}

_CACHE: dict[int, dict] = {}


def solution(panels: int, panel_nodes: int = 24) -> dict:
    """Heave radiation solution at the given panel count (memoised per test run)."""
    key = (panels, panel_nodes)
    if key not in _CACHE:
        mesh = submerged_circle_mesh(panels, RADIUS, SUBMERGENCE)
        result = solve_radiation_indirect(
            mesh, WAVENUMBER, "heave", density=DENSITY, panel_nodes=panel_nodes
        )
        far = far_field_damping(
            mesh, result.potential, result.normal_velocity, WAVENUMBER, DENSITY,
            gravity=GRAVITY,
        )
        _CACHE[key] = {
            "added_mass": float(result.added_mass[1]),
            "damping": float(result.damping[1]),
            "near_over_far": float(result.damping[1]) / float(far),
            "mesh": mesh,
        }
    return _CACHE[key]


# --------------------------------------------------------------------------
# Reference solution: the truncation is saturated, and an oversized request is
# rejected instead of overflowing somewhere inside math.exp.
# --------------------------------------------------------------------------
def test_reference_truncation_is_saturated():
    coarse = ursell_cylinder_coefficients(KA, RADIUS / SUBMERGENCE, terms=20)
    fine = ursell_cylinder_coefficients(KA, RADIUS / SUBMERGENCE, terms=100)
    assert math.isclose(coarse[0], fine[0], rel_tol=1.0e-14)
    assert math.isclose(coarse[1], fine[1], rel_tol=1.0e-14)


@pytest.mark.parametrize("terms", [101, 120, 400])
def test_oversized_truncation_is_rejected(terms):
    with pytest.raises(ValueError, match="must not exceed"):
        ursell_cylinder_coefficients(KA, RADIUS / SUBMERGENCE, terms=terms)


# --------------------------------------------------------------------------
# Quadrature: even and odd node counts agree, which is what the historical
# odd-node self-panel fix was for. This rules the quadrature out as a source of
# any historical number drift.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("panel_nodes", [5, 12, 32])
def test_quadrature_node_count_does_not_change_the_answer(panel_nodes):
    reference = solution(64)
    variant = solution(64, panel_nodes)
    assert math.isclose(variant["added_mass"], reference["added_mass"], rel_tol=1.0e-13)
    assert math.isclose(variant["damping"], reference["damping"], rel_tol=1.0e-13)


# --------------------------------------------------------------------------
# Reproduced numbers of the panel solution and of the MFS second solution.
# --------------------------------------------------------------------------
def test_panel_solution_is_locked_at_64_panels():
    row = solution(64)
    locked = LOCKED[64]
    assert math.isclose(row["added_mass"], locked["added_mass"], rel_tol=1.0e-12)
    assert math.isclose(row["damping"], locked["damping"], rel_tol=1.0e-12)
    assert math.isclose(row["near_over_far"], locked["near_over_far"], rel_tol=1.0e-10)
    # First-order scheme: well inside the 3 % acceptance band, far from 15 %.
    assert abs(row["added_mass"] / EXACT_ADDED_MASS - 1.0) < 2.0e-2
    assert abs(row["damping"] / EXACT_DAMPING - 1.0) < 3.0e-2


def test_panel_solution_is_locked_at_128_panels():
    row = solution(128)
    locked = LOCKED[128]
    assert math.isclose(row["added_mass"], locked["added_mass"], rel_tol=1.0e-12)
    assert math.isclose(row["damping"], locked["damping"], rel_tol=1.0e-12)
    assert math.isclose(row["near_over_far"], locked["near_over_far"], rel_tol=1.0e-10)


def test_method_of_fundamental_solutions_matches_the_locked_values():
    reference = radiation_reference(
        submerged_circle_mesh(128, RADIUS, SUBMERGENCE),
        WAVENUMBER,
        "heave",
        density=DENSITY,
        source_count=24,
    )
    locked = LOCKED[128]
    assert math.isclose(
        float(reference.added_mass[1]), locked["mfs_added_mass"], rel_tol=1.0e-10
    )
    assert math.isclose(
        float(reference.damping[1]), locked["mfs_damping"], rel_tol=1.0e-10
    )


# --------------------------------------------------------------------------
# Documentation lock: the published tables must quote the reproduced numbers.
# --------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]


def _section(text: str, start: str, stop: str) -> str:
    begin = text.index(start)
    return text[begin:text.index(stop, begin)]


def _cell_value(cell: str) -> float:
    """Parse a table cell holding either a plain number or ``a\times10^{b}``."""
    match = re.search(r"(-?[0-9.]+)\s*\\times10\^\{(-?\d+)\}", cell)
    if match:
        return float(match.group(1)) * 10.0 ** int(match.group(2))
    match = re.search(r"-?[0-9.]+", cell)
    return float(match.group(0)) if match else float("nan")


def _table_row(section: str, label: str) -> list[float]:
    match = re.search(rf"^\|\s*{label}\s*\|(.+)$", section, flags=re.MULTILINE)
    assert match is not None, f"no table row labelled {label!r} in the section"
    return [_cell_value(cell) for cell in match.group(1).split("|")]


def test_documented_panel_table_matches_the_code():
    section = _section(
        (_ROOT / "docs" / "marine_bem_benchmarks.md").read_text(encoding="utf-8"),
        "### 4.6",
        "### 4.7",
    )
    row = solution(64)
    documented = _table_row(section, "64")
    # | 64 | A | eps_A | B | eps_B | near/far |
    assert math.isclose(documented[0], row["added_mass"], rel_tol=1.0e-4)
    assert math.isclose(documented[2], row["damping"], rel_tol=1.0e-4)
    assert math.isclose(documented[4], row["near_over_far"], rel_tol=1.0e-3)
    assert math.isclose(
        documented[1], abs(row["added_mass"] / EXACT_ADDED_MASS - 1.0), rel_tol=2.0e-2
    )
    assert math.isclose(
        documented[3], abs(row["damping"] / EXACT_DAMPING - 1.0), rel_tol=2.0e-2
    )


def test_documented_mfs_row_is_labelled_with_the_right_panel_count():
    section = _section(
        (_ROOT / "docs" / "marine_bem_benchmarks.md").read_text(encoding="utf-8"),
        "### 4.6",
        "### 4.7",
    )
    match = re.search(r"^\|\s*基本解方法（MFS[^|]*）\s*\|(.+)$", section, flags=re.MULTILINE)
    assert match is not None
    assert "128" in match.group(0), "the MFS row must quote the panel count it used"
    reference = radiation_reference(
        submerged_circle_mesh(128, RADIUS, SUBMERGENCE), WAVENUMBER, "heave",
        density=DENSITY, source_count=24,
    )
    numbers = _table_row(section, "基本解方法（MFS，128 面元、24 源）")
    assert math.isclose(numbers[0], float(reference.added_mass[1]), rel_tol=1.0e-4)
    assert math.isclose(numbers[2], float(reference.damping[1]), rel_tol=1.0e-4)


def test_readme_quotes_the_reproduced_error():
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    assert "8.5\\times10^{-3}" in readme
    assert "7.7\\times10^{-3}" not in readme