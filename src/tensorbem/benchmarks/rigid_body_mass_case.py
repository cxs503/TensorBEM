"""CASE-BEM-RIGID-BODY-MASS: rigid-body mass matrix against the closed block form.

Reference (zero hand-transcription)
-----------------------------------
The target is the 6-DOF rigid-body inertia matrix in the block form of Fossen,
*Handbook of Marine Craft Hydrodynamics and Motion Control* (Wiley, 2011),
Sec. 3.3 "Rigid-Body Dynamics", eq. (3.44) -- the form used by every Cummins /
frequency-domain equation of motion in TensorBEM::

    M_RB = [[ m I3 ,        -m S(r_g)                        ],
            [ m S(r_g) ,     I_b + m S(r_g)^T S(r_g)          ]]

with ``S(a) b = a x b`` and ``I_b`` the inertia tensor about the centre of
gravity.  Every reference number in this module is *evaluated* from that closed
form in :mod:`tensorbem.rigid_body_mass` (``solid_ellipsoid_mass_properties`` +
``rigid_body_mass_matrix``); nothing is copied from a table.

The case body is a triaxial "ship-form" ellipsoid, ``L = 120 m``, ``B = 24 m``,
``T = 16 m`` (semi-axes 60 / 12 / 8 m), density 1025 kg/m^3, with its centre of
gravity offset from the reference point (midship, centreline, waterline) by
``r_g = (5, 0, -3) m`` -- so all four blocks are nonzero and the CG offset, the
``S(r_g)`` skew blocks and the parallel-axis transfer are all exercised.

The *measured* side is deliberately independent of the closed form: the body is
discretised with TensorBEM's own BEM panelisation (``ellipsoid_mesh``, icosphere
topology) and its volume, centroid and inertia tensor are integrated over the
closed triangulation by signed-tetrahedron quadrature
(:func:`tensorbem.rigid_body_mass.mesh_mass_properties`).  The 6x6 matrix is then
assembled by :func:`tensorbem.rigid_body_mass.rigid_body_mass_matrix`.  The
discretisation error of that quadrature is what the 3 % criterion measures, and
it converges quadratically (see the ``levels`` table).

What is measured
----------------
1. ``levels``      volume / centroid / inertia / 6x6 entrywise error vs panel
                   refinement (0..4 icosphere subdivisions = 20..5120 panels).
2. ``assembly``    the assembler fed with the *analytic* mass properties: error
                   must be roundoff, i.e. the block form is reproduced exactly.
3. ``negative_control``  the same matrix with both off-diagonal block signs
                   flipped: the kinetic-energy identity must break at O(1),
                   proving the main criterion is sensitive, not tautological.
4. ``sensitivity`` (a) CG-offset magnitude sweep: the coupling block norm must
                   grow linearly with ``|r_g|`` (fitted exponent ~1) and the
                   rotational block must follow the parallel-axis theorem;
                   (b) reference-point covariance: the same physical body
                   referred to a different point must satisfy
                   ``M_2 = A^T M_1 A`` to roundoff and keep the energy identity.
5. ``platform_assembly``  :func:`tensorbem.hydrostatic.compute_mass_matrix`
                   (the assembly used by ``tensorbem.rao`` and
                   ``tensorbem.ice_rao``) against the same closed form, so the
                   benchmark is bound to the integrated solver path.

Run::

    PYTHONPATH=src python -m tensorbem.benchmarks.rigid_body_mass_case \
        --out docs/cases/case_bem_rigid_body_mass/run.json

Exit status is 0 when every accepted level and every structural check passes,
1 otherwise (``verdict.all_pass``).
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorbem.geometry3d import ellipsoid_mesh
from tensorbem.hydrostatic import compute_mass_matrix
from tensorbem.rigid_body_mass import (
    MODES,
    block_form_residual,
    energy_residual,
    mesh_mass_properties,
    rigid_body_mass_matrix,
    skew_matrix,
    solid_ellipsoid_mass_properties,
)

#: Reference point of the body-fixed frame: midship, centreline, waterline.
REFERENCE_POINT = (0.0, 0.0, 0.0)
#: Semi-axes of the case body (L=120 m, B=24 m, T=16 m).
SEMI_AXES = (60.0, 12.0, 8.0)
DEFAULT_DENSITY = 1025.0
DEFAULT_CG_OFFSET = (5.0, 0.0, -3.0)
DEFAULT_SUBDIVISIONS: tuple[int, ...] = (0, 1, 2, 3, 4)
#: Two-sided criterion from the case goal: |ref - got| / |ref| < 3 %.
TOLERANCE = 0.03
#: A level is accepted (counts towards the verdict) from this subdivision on;
#: the coarser levels are reported to show the (failing) convergence floor.
ACCEPTED_FROM = 3

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = _REPO_ROOT / "docs" / "cases" / "case_bem_rigid_body_mass" / "run.json"

REFERENCE_CITATION = (
    "Fossen (2011) Handbook of Marine Craft Hydrodynamics and Motion Control, "
    "Sec. 3.3 'Rigid-Body Dynamics', eq. (3.44): "
    "M_RB = [[m I3, -m S(r_g)], [m S(r_g), I_b + m S(r_g)^T S(r_g)]]; "
    "solid-ellipsoid second moments I_cg = m(b^2+c^2, a^2+c^2, a^2+b^2)/5 "
    "(Lamb 1932 Ch. VI). All numbers machine-evaluated from these closed forms "
    "(solid_ellipsoid_mass_properties / rigid_body_mass_matrix); no table copied."
)


def _translated_mesh(semi_axes: Sequence[float], subdivisions: int, offset: Sequence[float]):
    """Ellipsoid panel mesh whose centroid sits at ``offset`` from the origin."""
    mesh = ellipsoid_mesh(int(subdivisions), np.asarray(semi_axes, dtype=float))
    return mesh.vertices + np.asarray(offset, dtype=float), mesh.faces


def _matrix_entry_rows(got: np.ndarray, ref: np.ndarray) -> list[dict[str, Any]]:
    """Flatten the 6x6 comparison into per-entry rows with the 3 % verdict."""
    rows: list[dict[str, Any]] = []
    for i in range(6):
        for j in range(6):
            reference = float(ref[i, j])
            measured = float(got[i, j])
            scale = abs(reference)
            error = abs(measured - reference) / scale if scale > 0.0 else abs(measured - reference)
            rows.append(
                {
                    "mode_i": MODES[i],
                    "mode_j": MODES[j],
                    "ref": reference,
                    "got": measured,
                    "rel_error": float(error),
                    "within_tolerance": bool(error < TOLERANCE),
                }
            )
    return rows


def _worst(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Worst entry of a flattened comparison, with the offending mode pair."""
    worst = max(rows, key=lambda row: row["rel_error"])
    return {
        "mode_i": worst["mode_i"],
        "mode_j": worst["mode_j"],
        "ref": worst["ref"],
        "got": worst["got"],
        "rel_error": worst["rel_error"],
    }


def _reference_properties(
    semi_axes: Sequence[float], density: float, cg_offset: Sequence[float]
) -> dict[str, Any]:
    """Closed-form mass properties and 6x6 block matrix (the ``ref`` side)."""
    base = solid_ellipsoid_mass_properties(semi_axes, density)
    inertia_cg = np.asarray(base["inertia_cg"], dtype=float)
    matrix = rigid_body_mass_matrix(
        base["mass"],
        cg_offset,
        inertia_cg,
        inertia_about="cg",
        reference_point=REFERENCE_POINT,
    )
    return {
        "volume": base["volume"],
        "mass": base["mass"],
        "centroid": list(cg_offset),
        "inertia_cg": inertia_cg.tolist(),
        "radii_of_gyration": base["radii_of_gyration"],
        "matrix": matrix.tolist(),
        "citation": REFERENCE_CITATION,
    }


def _level(
    subdivisions: int,
    semi_axes: Sequence[float],
    density: float,
    cg_offset: Sequence[float],
    reference: dict[str, Any],
) -> dict[str, Any]:
    """One mesh-resolution level of the case."""
    start = time.perf_counter()
    vertices, faces = _translated_mesh(semi_axes, subdivisions, cg_offset)
    props = mesh_mass_properties(
        vertices, faces, density, reference_point=REFERENCE_POINT
    )
    mass = float(props["mass"])
    centroid = np.asarray(props["centroid"], dtype=float)
    inertia_cg = np.asarray(props["inertia_cg"], dtype=float)
    matrix = rigid_body_mass_matrix(
        mass, centroid, inertia_cg, inertia_about="cg", reference_point=REFERENCE_POINT
    )
    wall = time.perf_counter() - start

    ref_matrix = np.asarray(reference["matrix"], dtype=float)
    ref_mass = float(reference["mass"])
    ref_inertia = np.asarray(reference["inertia_cg"], dtype=float)
    ref_centroid = np.asarray(reference["centroid"], dtype=float)

    rows = _matrix_entry_rows(matrix, ref_matrix)
    diagonal_scales = np.abs(np.diag(ref_matrix))
    inertia_error = float(
        np.max(np.abs(np.diag(inertia_cg) - np.diag(ref_inertia)) / np.diag(ref_inertia))
    )
    form = block_form_residual(
        matrix, mass, inertia_cg, centroid, reference_point=REFERENCE_POINT
    )
    return {
        "subdivisions": int(subdivisions),
        "panels": int(props["face_count"]),
        "wall_time_s": float(wall),
        "volume": float(props["volume"]),
        "volume_rel_error": float(
            abs(props["volume"] - float(reference["volume"])) / float(reference["volume"])
        ),
        "mass": mass,
        "mass_rel_error": float(abs(mass - ref_mass) / ref_mass),
        "centroid": centroid.tolist(),
        "centroid_abs_error": float(np.max(np.abs(centroid - ref_centroid))),
        "inertia_cg": inertia_cg.tolist(),
        "inertia_cg_diagonal": inertia_cg.diagonal().tolist(),
        "inertia_rel_error": inertia_error,
        "radii_of_gyration": props["radii_of_gyration"],
        "matrix": matrix.tolist(),
        "matrix_scale": float(np.max(diagonal_scales)),
        "max_entry_rel_error": float(max(row["rel_error"] for row in rows)),
        "worst_entry": _worst(rows),
        "entries": rows,
        "selfconsistency_block_form_error": float(form["max_relative_error"]),
        "energy_identity_error": float(
            energy_residual(matrix, mass, inertia_cg, centroid, reference_point=REFERENCE_POINT)
        ),
        "within_tolerance": bool(
            max(row["rel_error"] for row in rows) < TOLERANCE
        ),
    }


def _assembly_check(
    semi_axes: Sequence[float], density: float, cg_offset: Sequence[float], reference: dict[str, Any]
) -> dict[str, Any]:
    """Feed the *analytic* mass properties to the assembler (pure form check)."""
    mass = float(reference["mass"])
    inertia_cg = np.asarray(reference["inertia_cg"], dtype=float)
    centroid = np.asarray(reference["centroid"], dtype=float)
    matrix = rigid_body_mass_matrix(
        mass, centroid, inertia_cg, inertia_about="cg", reference_point=REFERENCE_POINT
    )
    form = block_form_residual(
        matrix, mass, inertia_cg, centroid, reference_point=REFERENCE_POINT
    )
    ref_matrix = np.asarray(reference["matrix"], dtype=float)
    # Hand-written copy of the block form: an independent re-derivation that does
    # not call the assembler, so a sign slip inside the assembler cannot hide.
    s = skew_matrix(centroid)
    hand = np.zeros((6, 6), dtype=float)
    hand[:3, :3] = mass * np.eye(3)
    hand[:3, 3:] = -mass * s
    hand[3:, :3] = mass * s
    hand[3:, 3:] = inertia_cg + mass * (s.T @ s)
    return {
        "requested_subdivisions": None,
        "max_entry_rel_error_vs_reference": float(
            np.max(np.abs(matrix - ref_matrix)) / np.max(np.abs(ref_matrix))
        ),
        "hand_written_form_vs_assembler": float(
            np.max(np.abs(matrix - hand)) / np.max(np.abs(hand))
        ),
        "block_form_residual": float(form["max_relative_error"]),
        "block_relative_error": form["block_relative_error"],
        "symmetry_error": float(form["symmetry_error"]),
        "energy_identity_error": float(
            energy_residual(matrix, mass, inertia_cg, centroid, reference_point=REFERENCE_POINT)
        ),
        "positive_definite": bool(np.all(np.linalg.eigvalsh(matrix) > 0.0)),
        "smallest_eigenvalue": float(np.min(np.linalg.eigvalsh(matrix))),
        "mode_order": list(MODES),
        "diagonal": np.diag(matrix).tolist(),
    }


def _negative_control(reference: dict[str, Any]) -> dict[str, Any]:
    """Two controlled perturbations of the reference matrix.

    (a) both off-diagonal blocks negated -- a pure sign slip;
    (b) the *exact* legacy TensorBEM assembly
        ``[[m I3, -m S^T], [-m S, I_O + m S S^T]]`` that
        ``hydrostatic.compute_mass_matrix`` produced before mission m17324,
        evaluated on the same body.

    Both stay symmetric, so any test that only checks symmetry passes them;
    the kinetic-energy identity must break at O(1) in both cases, which is what
    makes the main criterion sensitive rather than tautological.
    """
    mass = float(reference["mass"])
    inertia_cg = np.asarray(reference["inertia_cg"], dtype=float)
    centroid = np.asarray(reference["centroid"], dtype=float)
    ref_matrix = np.asarray(reference["matrix"], dtype=float)

    flipped = ref_matrix.copy()
    flipped[:3, 3:] = -flipped[:3, 3:]
    flipped[3:, :3] = -flipped[3:, :3]

    s = skew_matrix(centroid)
    inertia_origin = inertia_cg + mass * (
        float(centroid @ centroid) * np.eye(3) - np.outer(centroid, centroid)
    )
    legacy = np.zeros((6, 6), dtype=float)
    legacy[:3, :3] = mass * np.eye(3)
    legacy[:3, 3:] = -mass * s.T
    legacy[3:, :3] = -mass * s
    legacy[3:, 3:] = inertia_origin + mass * (s @ s.T)

    def _describe(matrix: np.ndarray) -> dict[str, Any]:
        blocks = {
            "translation_translation": (slice(0, 3), slice(0, 3)),
            "translation_rotation": (slice(0, 3), slice(3, 6)),
            "rotation_translation": (slice(3, 6), slice(0, 3)),
            "rotation_rotation": (slice(3, 6), slice(3, 6)),
        }
        return {
            "energy_identity_error": float(
                energy_residual(
                    matrix, mass, inertia_cg, centroid, reference_point=REFERENCE_POINT
                )
            ),
            "max_entry_rel_error_vs_reference": float(
                np.max(np.abs(matrix - ref_matrix)) / np.max(np.abs(ref_matrix))
            ),
            "block_relative_error": {
                name: float(
                    np.max(np.abs(matrix[idx] - ref_matrix[idx]))
                    / np.max(np.abs(ref_matrix[idx]))
                )
                for name, idx in blocks.items()
            },
            "still_symmetric": bool(np.allclose(matrix, matrix.T, atol=0.0)),
            "block_form_residual": float(
                block_form_residual(
                    matrix, mass, inertia_cg, centroid, reference_point=REFERENCE_POINT
                )["max_relative_error"]
            ),
        }

    return {
        "sign_flip": {
            "what": "both off-diagonal blocks of the reference matrix negated",
            **_describe(flipped),
        },
        "legacy_assembly": {
            "what": (
                "pre-m17324 hydrostatic.compute_mass_matrix formula "
                "[[m I3, -m S^T], [-m S, I_O + m S S^T]] on this body"
            ),
            **_describe(legacy),
        },
        "detects": {
            "sign_flip": bool(_describe(flipped)["energy_identity_error"] > 0.1),
            "legacy_assembly": bool(_describe(legacy)["energy_identity_error"] > 0.1),
        },
        "note": (
            "Both perturbations leave the matrix symmetric, so symmetry alone "
            "cannot detect them; only the kinetic-energy identity (or an "
            "entrywise comparison with the closed form) can."
        ),
    }


def _offset_sensitivity(
    semi_axes: Sequence[float],
    density: float,
    cg_offset: Sequence[float],
    scales: Sequence[float] = (0.0, 0.5, 1.0, 2.0),
) -> dict[str, Any]:
    """Coupling block norm vs |r_g| and the parallel-axis transfer."""
    base = solid_ellipsoid_mass_properties(semi_axes, density)
    mass = float(base["mass"])
    inertia_cg = np.asarray(base["inertia_cg"], dtype=float)
    direction = np.asarray(cg_offset, dtype=float)
    rows: list[dict[str, Any]] = []
    for scale in scales:
        r_g = direction * float(scale)
        matrix = rigid_body_mass_matrix(
            mass, r_g, inertia_cg, inertia_about="cg", reference_point=REFERENCE_POINT
        )
        s = skew_matrix(r_g)
        parallel_axis = inertia_cg + mass * (s.T @ s)
        rows.append(
            {
                "cg_scale": float(scale),
                "cg_offset": r_g.tolist(),
                "cg_distance": float(np.linalg.norm(r_g)),
                "coupling_block_max_abs": float(np.max(np.abs(matrix[:3, 3:]))),
                "rotational_block": matrix[3:, 3:].tolist(),
                "rotational_block_vs_parallel_axis": float(
                    np.max(np.abs(matrix[3:, 3:] - parallel_axis)) / np.max(np.abs(parallel_axis))
                ),
                "coupling_over_mass_distance": (
                    float(np.max(np.abs(matrix[:3, 3:]))) / (mass * float(np.linalg.norm(r_g)))
                    if np.linalg.norm(r_g) > 0.0
                    else None
                ),
                "energy_identity_error": float(
                    energy_residual(
                        matrix, mass, inertia_cg, r_g, reference_point=REFERENCE_POINT
                    )
                ),
            }
        )
    # Fitted exponent of coupling norm vs distance (should be 1: linear in m|r_g|)
    usable = [row for row in rows if row["cg_distance"] > 0.0]
    exponent = None
    if len(usable) >= 2:
        x = np.log([row["cg_distance"] for row in usable])
        y = np.log([row["coupling_block_max_abs"] for row in usable])
        exponent = float(np.polyfit(x, y, 1)[0])
    return {
        "scales": [float(value) for value in scales],
        "rows": rows,
        "coupling_norm_exponent_vs_distance": exponent,
        "expected_exponent": 1.0,
    }


def _reference_point_covariance(
    semi_axes: Sequence[float], density: float, cg_offset: Sequence[float]
) -> dict[str, Any]:
    """Same body referred to a second point: ``M_2 = A^T M_1 A``.

    ``A`` maps a twist of the original reference point to the twist of the new
    one: ``v' = v + w x (o - o')``, ``w' = w``, so the quadratic form -- the
    kinetic energy -- is invariant.
    """
    base = solid_ellipsoid_mass_properties(semi_axes, density)
    mass = float(base["mass"])
    inertia_cg = np.asarray(base["inertia_cg"], dtype=float)
    original = np.asarray(REFERENCE_POINT, dtype=float)
    shifted = np.array([10.0, 0.0, -2.0])
    m1 = rigid_body_mass_matrix(
        mass, cg_offset, inertia_cg, inertia_about="cg", reference_point=original
    )
    m2 = rigid_body_mass_matrix(
        mass, cg_offset, inertia_cg, inertia_about="cg", reference_point=shifted
    )
    # A = [[I3, -S(o - o')], [0, I3]] in row-wise (v, w) ordering, because
    # v' = v + w x (o - o') = v - S(o - o') w.
    delta = original - shifted
    a = np.zeros((6, 6), dtype=float)
    a[:3, :3] = np.eye(3)
    a[:3, 3:] = -skew_matrix(delta)
    a[3:, 3:] = np.eye(3)
    transformed = a.T @ m1 @ a
    return {
        "reference_points": [original.tolist(), shifted.tolist()],
        "covariance_residual_rel": float(
            np.max(np.abs(transformed - m2)) / np.max(np.abs(m2))
        ),
        "energy_identity_error_original": float(
            energy_residual(m1, mass, inertia_cg, cg_offset, reference_point=original)
        ),
        "energy_identity_error_shifted": float(
            energy_residual(m2, mass, inertia_cg, cg_offset, reference_point=shifted)
        ),
        "rotational_diagonal_original": m1[3:, 3:].diagonal().tolist(),
        "rotational_diagonal_shifted": m2[3:, 3:].diagonal().tolist(),
    }


def _platform_assembly(reference: dict[str, Any]) -> dict[str, Any]:
    """The integrated solver-path assembly (``hydrostatic.compute_mass_matrix``)."""
    mass = float(reference["mass"])
    centroid = np.asarray(reference["centroid"], dtype=float)
    inertia_cg = np.asarray(reference["inertia_cg"], dtype=float)
    inertia_origin = inertia_cg + mass * (
        float(centroid @ centroid) * np.eye(3) - np.outer(centroid, centroid)
    )
    # Explicit flag: the solver call sites (``rao.py`` / ``ice_rao.py``) pass an
    # origin-referred tensor, which stays the default convention; the transfer
    # must not be added twice (it would be, with ``inertia_about="cg"``).
    measured = compute_mass_matrix(
        mass, centroid, inertia=inertia_origin, inertia_about="origin"
    )  # equivalent to compute_mass_matrix(mass, centroid, inertia=inertia_cg)
    ref_matrix = np.asarray(reference["matrix"], dtype=float)
    form = block_form_residual(
        measured,
        mass,
        inertia_cg,
        centroid,
        reference_point=REFERENCE_POINT,
    )
    return {
        "entry": "tensorbem.hydrostatic.compute_mass_matrix",
        "callers": [
            "tensorbem.rao.compute_rao_at_frequency (line 434)",
            "tensorbem.ice_rao.compute_ice_rao (line 413)",
            "tensorbem.benchmarks.hydrostatic_rao_verification.run_mass_matrix_check",
        ],
        "inertia_convention": "inertia_about=origin (explicitly passed; matches rao.py / ice_rao.py call sites)",
        "max_entry_rel_error": float(
            np.max(np.abs(measured - ref_matrix)) / np.max(np.abs(ref_matrix))
        ),
        "block_form_residual": float(form["max_relative_error"]),
        "block_relative_error": form["block_relative_error"],
        "energy_identity_error": float(
            energy_residual(measured, mass, inertia_cg, centroid, reference_point=REFERENCE_POINT)
        ),
        "symmetry_error": float(form["symmetry_error"]),
    }


def run(
    subdivisions: Sequence[int] = DEFAULT_SUBDIVISIONS,
    semi_axes: Sequence[float] = SEMI_AXES,
    density: float = DEFAULT_DENSITY,
    cg_offset: Sequence[float] = DEFAULT_CG_OFFSET,
) -> dict[str, Any]:
    """Run the whole case and return a JSON-serialisable report."""
    reference = _reference_properties(semi_axes, density, cg_offset)
    levels = [
        _level(int(sd), semi_axes, density, cg_offset, reference)
        for sd in subdivisions
    ]
    assembly = _assembly_check(semi_axes, density, cg_offset, reference)
    negative = _negative_control(reference)
    offset_study = _offset_sensitivity(semi_axes, density, cg_offset)
    covariance = _reference_point_covariance(semi_axes, density, cg_offset)
    platform_side = _platform_assembly(reference)

    accepted = [row for row in levels if row["subdivisions"] >= ACCEPTED_FROM]
    if not accepted:
        accepted = levels[-1:]
    worst_level = max(levels, key=lambda row: row["max_entry_rel_error"])
    checks = {
        "accepted_levels_within_3pct": all(row["within_tolerance"] for row in accepted),
        "assembly_is_roundoff": assembly["block_form_residual"] < 1.0e-12,
        "hand_written_form_matches": assembly["hand_written_form_vs_assembler"] < 1.0e-12,
        "energy_identity_assembly": assembly["energy_identity_error"] < 1.0e-12,
        "energy_identity_platform_assembly": platform_side["energy_identity_error"] < 1.0e-12,
        "platform_block_form": platform_side["block_form_residual"] < 1.0e-12,
        "negative_control_breaks_energy": bool(
            all(negative["detects"].values())
        ),
        "coupling_linear_in_cg_distance": (
            offset_study["coupling_norm_exponent_vs_distance"] is not None
            and abs(offset_study["coupling_norm_exponent_vs_distance"] - 1.0) < 0.02
        ),
        "reference_point_covariance": covariance["covariance_residual_rel"] < 1.0e-12,
        "inertia_converges_monotonically": all(
            levels[i + 1]["inertia_rel_error"] <= levels[i]["inertia_rel_error"]
            for i in range(len(levels) - 1)
        ),
    }
    return {
        "benchmark": "CASE-BEM-RIGID-BODY-MASS",
        "subject": "6x6 rigid-body mass matrix, closed block form",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "parameters": {
            "semi_axes_m": [float(value) for value in semi_axes],
            "length_overall_m": 2.0 * float(semi_axes[0]),
            "breadth_m": 2.0 * float(semi_axes[1]),
            "depth_m": 2.0 * float(semi_axes[2]),
            "density_kg_m3": float(density),
            "cg_offset_m": [float(value) for value in cg_offset],
            "reference_point_m": list(REFERENCE_POINT),
            "mode_order": list(MODES),
            "subdivisions": [int(sd) for sd in subdivisions],
            "panels_per_level": [row["panels"] for row in levels],
            "tolerance": TOLERANCE,
            "accepted_from_subdivision": ACCEPTED_FROM,
        },
        "reference": reference,
        "levels": levels,
        "assembly": assembly,
        "negative_control": negative,
        "sensitivity_cg_offset": offset_study,
        "sensitivity_reference_point": covariance,
        "platform_assembly": platform_side,
        "verdict": {
            "all_pass": bool(all(checks.values())),
            "checks": checks,
            "worst_level": {
                "subdivisions": worst_level["subdivisions"],
                "max_entry_rel_error": worst_level["max_entry_rel_error"],
                "worst_entry": worst_level["worst_entry"],
            },
            "worst_accepted_level": {
                "subdivisions": max(accepted, key=lambda row: row["max_entry_rel_error"])[
                    "subdivisions"
                ],
                "max_entry_rel_error": max(row["max_entry_rel_error"] for row in accepted),
            },
            "accepted_errors": {
                str(row["subdivisions"]): row["max_entry_rel_error"] for row in accepted
            },
            "criterion": "|ref-got|/|ref| < 3% per matrix entry on accepted levels",
        },
        "expected": {
            "mass_kg": float(reference["mass"]),
            "inertia_cg": reference["inertia_cg"],
            "diagonal": [reference["matrix"][i][i] for i in range(6)],
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subdivisions", nargs="+", type=int, default=list(DEFAULT_SUBDIVISIONS))
    parser.add_argument("--density", type=float, default=DEFAULT_DENSITY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    report = run(subdivisions=args.subdivisions, density=args.density)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    verdict = report["verdict"]
    print(
        json.dumps(
            {
                "benchmark": report["benchmark"],
                "worst_level": verdict["worst_accepted_level"],
                "all_pass": verdict["all_pass"],
                "checks": verdict["checks"],
                "written": str(args.out),
            },
            indent=2,
        )
    )
    return 0 if verdict["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())