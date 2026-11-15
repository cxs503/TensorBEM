"""Oblique-heading verification of the floating-hemisphere case (small meshes).

These tests are the fast, always-on companion of
``benchmarks/verified/casepool_bem_hulme_heading``: they run on 64/256-panel
meshes instead of the 1024-panel case mesh, so the invariant they assert is
checked on every CI run at a cost of a few seconds.

The invariant is exact in the continuum and rests only on the axial symmetry of
a surface-piercing hemisphere: with the incident potential

    phi_I(beta) = -(i g A / omega) exp(nu z + i nu (x cos beta + y sin beta))

being the rotation of the ``beta = 0`` potential, the complex load six-vector
must satisfy ``V(beta) = R_beta V(0)``, hence ``Fz`` == ``Fz(0)`` and
``Mz`` == 0 for every heading.

One test is deliberately *negative*: it applies the rotation in the wrong
direction and asserts that the identity breaks. A test that only ever passes
cannot tell a correct convention from a trivially satisfied one.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "verified" / "casepool_bem_hulme_heading"))

import run_case as heading  # noqa: E402


def _loads_np(row: dict) -> np.ndarray:
    return np.array([complex(v["re"], v["im"]) for v in row["loads"]])


def _network(ka: float = 1.0, subdivisions: int = 2, headings=(0.0, 45.0, 90.0, 180.0)):
    return heading.heading_cell({
        "kind": "heading", "ka": ka, "subdivisions": subdivisions,
        "headings_deg": list(headings),
    })


@pytest.fixture(scope="module")
def small_mesh_cell() -> dict:
    return _network()


def test_rotation_invariance_of_the_load_six_vector(small_mesh_cell):
    """V(beta) == R_beta V(0) on a 64-panel mesh, to the anisotropy level."""
    base = _loads_np(small_mesh_cell["headings"][0])
    for row in small_mesh_cell["headings"]:
        beta = math.radians(row["heading_deg"])
        got = _loads_np(row)
        predicted = heading.rotate_loads(base, beta)
        deviation = np.max(np.abs(got - predicted)) / np.max(np.abs(predicted))
        assert deviation < 5.0e-3, (
            f"heading {row['heading_deg']}: rotation invariance broken, "
            f"deviation {deviation:.3e}")


def test_heave_force_is_heading_invariant(small_mesh_cell):
    base = _loads_np(small_mesh_cell["headings"][0])
    for row in small_mesh_cell["headings"]:
        got = _loads_np(row)
        assert abs(got[2] - base[2]) / abs(base[2]) < 5.0e-3


def test_yaw_exciting_moment_is_null(small_mesh_cell):
    """The yaw exciting moment of an axially symmetric body is exactly zero."""
    for row in small_mesh_cell["headings"]:
        got = _loads_np(row)
        assert abs(got[5]) < 1.0e-4, (
            f"heading {row['heading_deg']}: |Mz|/(rho g A a^3) = {abs(got[5]):.3e}")


def test_rotated_mesh_algebraic_identity():
    """Solving on the rotated mesh at heading 0 reproduces the direct solve."""
    cell = heading.rotated_cell({
        "kind": "rotated", "ka": 1.0, "subdivisions": 2,
        "headings_deg": [45.0, 90.0],
    })
    for row in cell["headings"]:
        assert row["max_deviation"] < 1.0e-2, (
            f"heading {row['heading_deg']}: identity deviation "
            f"{row['max_deviation']:.3e}")


def test_identity_is_discriminating_against_a_wrong_rotation_direction():
    """Applying R_{-beta} instead of R_{+beta} must break the identity.

    Without this, the identity test could be vacuous: it has to be sensitive to
    the sign convention it claims to police.
    """
    cell = heading.rotated_cell({
        "kind": "rotated", "ka": 1.0, "subdivisions": 2,
        "headings_deg": [45.0],
    })
    row = cell["headings"][0]
    beta = math.radians(row["heading_deg"])
    rebuilt = np.array([complex(v["re"], v["im"])
                        for v in row["rotated_mesh_loads"]])
    wrong = heading.rotate_loads(rebuilt, -beta)
    right = heading.rotate_loads(rebuilt, beta)
    direct = np.array([complex(v["re"], v["im"]) for v in row["direct_loads"]])
    scale = np.max(np.abs(rebuilt))
    assert np.max(np.abs(direct - right)) / scale < 1.0e-2
    assert np.max(np.abs(direct - wrong)) / scale > 0.2


def test_haskind_reciprocity_holds_at_an_oblique_heading(small_mesh_cell):
    for row in small_mesh_cell["headings"]:
        got = _loads_np(row)[:3]
        kochin = np.array([complex(v["re"], v["im"]) for v in row["kochin_force"]])
        boundary = np.array([complex(v["re"], v["im"])
                             for v in row["boundary_haskind_force"]])
        scale = np.max(np.abs(got))
        assert np.max(np.abs(got - kochin)) / scale < 5.0e-2
        assert np.max(np.abs(got - boundary)) / scale < 5.0e-2


def test_radiation_block_is_axially_symmetric(small_mesh_cell):
    """A22 == A11 and B22 == B11 exactly for a body of revolution."""
    rad = small_mesh_cell["radiation"]
    assert abs(rad["A22_dimless"] / rad["A11_dimless"] - 1.0) < 5.0e-3
    assert abs(rad["B22_dimless"] / rad["B11_dimless"] - 1.0) < 5.0e-3
    assert abs(rad["A12_dimless"]) < 1.0e-2
    assert abs(rad["B12_dimless"]) < 1.0e-2


def test_case_mesh_has_rotational_symmetry_at_multiples_of_90_degrees():
    """The 1024-panel case mesh is unchanged (up to relabelling) by a 90 deg turn.

    This is what makes 90/180/270-degree headings *exactly* invariant on the
    fixed case mesh while 45 degrees is not: the anisotropy seen there is a
    property of the mesh, not of the heading implementation.
    """
    cell = _network(ka=1.0, subdivisions=3, headings=(0.0, 90.0, 180.0))
    base = _loads_np(cell["headings"][0])
    for row in cell["headings"][1:]:
        got = _loads_np(row)
        beta = math.radians(row["heading_deg"])
        predicted = heading.rotate_loads(base, beta)
        assert np.max(np.abs(got - predicted)) / np.max(np.abs(predicted)) < 1.0e-5


def test_task_list_is_deterministic_and_covers_the_case_mesh():
    tasks = heading.build_tasks()
    assert heading.build_tasks() == tasks
    kinds = [t["kind"] for t in tasks]
    assert kinds.count("heading") == len(heading.CASE_KA) + 1
    assert kinds.count("rotated") == 1
    case_tasks = [t for t in tasks
                  if t["kind"] == "heading" and t["subdivisions"] == heading.CASE_SUBDIVISIONS]
    assert sorted(t["ka"] for t in case_tasks) == sorted(heading.CASE_KA)
    assert all(t["headings_deg"] == list(heading.HEADINGS_DEG) for t in case_tasks)