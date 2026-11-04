"""Regression tests for the 6-DOF radiation-matrix algebra (m3630).

The benchmark in :mod:`tensorbem.benchmarks.six_dof_matrix_algebra` states four
exact predictions about the ``(6, 6)`` added-mass and damping matrices that
:func:`tensorbem.free_surface_solver3d.solve_radiation_3d` returns:

* reciprocity ``A = A^T``, ``B = B^T`` (a theorem of the continuous problem);
* reflection selection rules: entries whose two modes have opposite parity under
  ``x -> -x`` or ``y -> -y`` vanish identically;
* the spin-axis identity ``A_66 = B_66 = 0`` for a body of revolution;
* reference-centre covariance ``A' = M^{-T} A M^{-1}``, exact for any mesh.

These tests pin each one to its measured convergence order instead of a magic
constant, add a negative control for the covariance check (so a silently
transposed matrix cannot pass), and gate the numbers printed in
``docs/six_dof_matrix_algebra.md`` against a recomputation.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from tensorbem.benchmarks import six_dof_matrix_algebra as module

DOC = Path(__file__).resolve().parents[1] / "docs" / "six_dof_matrix_algebra.md"
PINNED = Path(__file__).resolve().parent / "data" / "six_dof_matrix_algebra_metrics.json"


@pytest.fixture(scope="module")
def study() -> dict:
    """The published study: three meshes, six modes, one centre-shift pair."""
    return module.run((1, 2, 3), 1.0, covariance_subdivisions=2)


# --------------------------------------------------------------------------
# structure of the prediction, before any physics is run
# --------------------------------------------------------------------------
def test_selection_mask_is_the_parity_product_rule() -> None:
    mask = module.selection_mask()
    assert mask.shape == (6, 6)
    assert not np.any(np.diag(mask)), "the diagonal is never forbidden"
    assert np.array_equal(mask, mask.T), "parity of an unordered pair is symmetric"
    # 20 of the 30 off-diagonal entries are killed by one of the two reflections
    assert int(mask.sum()) == 26
    assert module.forbidden_pairs() == [(i, j) for i in range(6) for j in range(6) if mask[i, j]]
    # the survivors are exactly the two same-parity 2x2 blocks
    assert module.same_class_couplings() == ((0, 4), (1, 3))


def test_parity_table_is_realised_by_the_assembled_normals() -> None:
    """The mesh and its generalized normals must obey the table the mask assumes."""
    from tensorbem.geometry3d import hemisphere_hull_mesh

    for subdivisions in (1, 2):
        mesh = hemisphere_hull_mesh(subdivisions)
        parity = module.reflection_parity_residuals(mesh)
        for axis in ("x", "y"):
            assert parity["mesh"][axis] < 1.0e-12, (
                f"hull is not mirror symmetric about {axis}=0"
            )
            for name, residual in parity["modes"][axis].items():
                assert residual < 1.0e-12, f"parity of {name} under S_{axis} is wrong"


# --------------------------------------------------------------------------
# the four predictions
# --------------------------------------------------------------------------
def test_added_mass_reciprocity_converges(study: dict) -> None:
    levels = study["levels"]
    values = [row["added_mass_block"]["reciprocity_rel"] for row in levels]
    assert values == sorted(values, reverse=True), values
    assert values[0] < 1.0e-3
    assert values[-1] < 1.0e-6, values
    order = study["observed_order"]["added_mass_block.reciprocity_rel"][1]
    assert order is not None and order > 1.0


def test_damping_reciprocity_converges(study: dict) -> None:
    levels = study["levels"]
    values = [row["damping_block"]["reciprocity_rel"] for row in levels]
    assert values == sorted(values, reverse=True), values
    assert values[0] < 1.0e-3
    assert values[-1] < 3.0e-5, values


def test_reflection_forbidden_couplings_vanish(study: dict) -> None:
    """The 26 reflection-forbidden entries must be zero; measured ones fall like ``h**3``."""
    levels = study["levels"]
    for key in ("added_mass_block", "damping_block"):
        values = [row[key]["forbidden_rel"] for row in levels]
        assert max(values) < 1.0e-4, (key, values)
        assert values[-1] < 3.0e-6, (key, values)
        assert values == sorted(values, reverse=True), (key, values)
        order = study["observed_order"][f"{key}.forbidden_rel"][-1]
        assert order is not None and order > 2.0, (key, order)
    # the worst offender is named, so a regression reports which pair broke
    for row in levels:
        assert row["forbidden_worst_added_mass"]["pair"] in study["forbidden_pairs"]
        assert row["forbidden_worst_damping"]["pair"] in study["forbidden_pairs"]


def test_spin_axis_added_inertia_and_damping_vanish(study: dict) -> None:
    """Yaw about the axis of a body of revolution: ``A66 = B66 = 0`` exactly."""
    levels = study["levels"]
    added = [row["spin_axis_added_mass_rel"] for row in levels]
    damping = [row["spin_axis_damping_rel"] for row in levels]
    assert added == sorted(added, reverse=True), added
    assert damping == sorted(damping, reverse=True), damping
    assert added[-1] < 1.0e-4, added
    assert damping[-1] < 1.0e-8, damping
    assert added[0] / added[-1] > 20.0, "the residual must shrink with refinement"


def test_axisymmetry_of_the_hemisphere(study: dict) -> None:
    """``A11 = A22`` to the level the mesh is actually rotation invariant.

    The octahedron-based hull is mirror symmetric about the two axis planes but
    not exactly invariant under a 90-degree rotation about ``z``, so the residual
    is a small discretisation effect (1.7e-6 of the diagonal at 16 panels), not
    roundoff.
    """
    for row in study["levels"]:
        assert row["axisymmetry_rel"] < 1.0e-5, row["subdivisions"]


def test_centre_shift_covariance_is_exact(study: dict) -> None:
    """``A' = M^{-T} A M^{-1}`` must hold to roundoff, not to discretisation error."""
    covariance = study["covariance"]
    assert covariance["added_mass_residual_rel"] < 1.0e-12, covariance
    assert covariance["damping_residual_rel"] < 1.0e-12, covariance
    # a shifted reference point really does fill in the couplings (the check is
    # not passing because both matrices are diagonal)
    origin = np.asarray(covariance["added_mass_diagonal_origin"])
    shifted = np.asarray(covariance["added_mass_diagonal_shifted"])
    assert np.max(np.abs(shifted[:3] - origin[:3])) == 0.0
    assert np.max(np.abs(shifted[3:] - origin[3:])) > 1.0e-2 * np.max(origin)


def test_centre_shift_covariance_negative_controls(study: dict) -> None:
    """A transposed matrix, or the wrong shift sign, must fail the same check."""
    mesh = module.hemisphere_hull_mesh(1)
    base = module.solve_radiation_3d(mesh, 1.0)
    offset = (0.0, -0.4, 0.1)
    moved = module.solve_radiation_3d(mesh, 1.0, center=np.asarray(offset))
    good = module.covariance_residual(base.added_mass, moved.added_mass, offset)
    assert good < 1.0e-12
    # negative control 1: wrong sign on the lever arm
    flipped = module.covariance_residual(
        base.added_mass, moved.added_mass, np.negative(np.asarray(offset))
    )
    # negative control 2: index order scrambled
    transposed = module.covariance_residual(
        np.asarray(base.added_mass).T, moved.added_mass, offset
    )
    assert flipped > 1.0e3 * good, (flipped, good)
    assert transposed > 1.0e3 * good, (transposed, good)


def test_permitted_couplings_decay_but_are_not_asserted_to_vanish(study: dict) -> None:
    """Surge-pitch / sway-roll survive the reflections; only their decay is pinned.

    Reciprocity makes them symmetric, so they cannot be an artefact of an
    antisymmetric relation; the measured order (nearly 2) shows they are
    discretisation error, but the data cannot prove an exact zero, so the
    benchmark reports them instead of asserting one.
    """
    for name, series in study["same_class_orders"].items():
        magnitudes = [
            abs(row["same_class"][name]["symmetric_part_rel"]) for row in study["levels"]
        ]
        assert magnitudes == sorted(magnitudes, reverse=True), (name, magnitudes)
        order = series[-1]
        assert order is not None and order > 1.5, (name, order)
        for row in study["levels"]:
            entry = row["same_class"][name]
            # the two orders agree: reciprocity holds *inside* the block
            assert abs(entry["antisymmetric_part_rel"]) < 0.2 * abs(
                entry["symmetric_part_rel"]
            ), entry


# --------------------------------------------------------------------------
# pinned metrics and the document gate
# --------------------------------------------------------------------------
def _pinned_metrics(study: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in study["levels"]:
        tag = f"subdivisions_{row['subdivisions']}"
        out[f"{tag}.panels"] = float(row["panels"])
        out[f"{tag}.panel_size"] = row["panel_size"]
        out[f"{tag}.added_mass_reciprocity_rel"] = row["added_mass_block"]["reciprocity_rel"]
        out[f"{tag}.damping_reciprocity_rel"] = row["damping_block"]["reciprocity_rel"]
        out[f"{tag}.added_mass_forbidden_rel"] = row["added_mass_block"]["forbidden_rel"]
        out[f"{tag}.damping_forbidden_rel"] = row["damping_block"]["forbidden_rel"]
        out[f"{tag}.spin_axis_added_mass_rel"] = row["spin_axis_added_mass_rel"]
        out[f"{tag}.spin_axis_damping_rel"] = row["spin_axis_damping_rel"]
        out[f"{tag}.surge_pitch_symmetric_part_rel"] = row["same_class"]["surge-pitch"][
            "symmetric_part_rel"
        ]
    out["covariance.added_mass_residual_rel"] = study["covariance"][
        "added_mass_residual_rel"
    ]
    out["covariance.damping_residual_rel"] = study["covariance"]["damping_residual_rel"]
    return out


def test_metrics_are_pinned(study: dict) -> None:
    measured = _pinned_metrics(study)
    pinned = json.loads(PINNED.read_text(encoding="utf-8"))
    missing = sorted(set(pinned) - set(measured))
    assert not missing, f"pinned keys no longer produced: {missing}"
    for key, expected in pinned.items():
        assert measured[key] == pytest.approx(expected, rel=1.0e-3, abs=1.0e-15), key


def _parse_first_table(text: str) -> list[list[str]]:
    """Return the data rows of the measurement table (the one headed ``subdiv``).

    The note also carries a parity table and an order table; anchoring on the
    ``subdiv`` header keeps the gate pointed at the measured numbers.
    """
    rows: list[list[str]] = []
    header_seen = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            if header_seen and rows:
                break
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not header_seen:
            if cells[0] != "subdiv":
                continue
            header_seen = True
            continue
        if set("".join(cells)) <= set("-: "):
            continue
        rows.append(cells)
    return rows


def test_document_table_matches_the_recomputation(study: dict) -> None:
    """The published table is compared cell by cell with the run above."""
    assert DOC.exists(), f"missing document {DOC}"
    rows = _parse_first_table(DOC.read_text(encoding="utf-8"))
    levels = study["levels"]
    assert len(rows) == len(levels), f"document table has {len(rows)} data rows"
    for cells, row in zip(rows, levels):
        numbers = [float(value) for value in cells[1:]]
        assert int(numbers[0]) == row["panels"]
        expected = [
            row["panel_size"],
            row["condition_number"],
            row["relative_residual"],
            row["added_mass_block"]["reciprocity_rel"],
            row["damping_block"]["reciprocity_rel"],
            row["added_mass_block"]["forbidden_rel"],
            row["damping_block"]["forbidden_rel"],
            row["spin_axis_added_mass_rel"],
            row["spin_axis_damping_rel"],
        ]
        for label, got, want in zip(cells[2:], numbers[1:], expected):
            assert got == pytest.approx(want, rel=5.0e-2, abs=1.0e-15), (
                f"{label} in the document disagrees with the code: {got} vs {want}"
            )
    # the document must also quote the exact (roundoff) covariance residuals
    covariance = study["covariance"]
    text = DOC.read_text(encoding="utf-8")
    for value in (
        covariance["added_mass_residual_rel"],
        covariance["damping_residual_rel"],
    ):
        assert f"{value:.1e}" in text, f"covariance residual {value:.1e} not quoted"