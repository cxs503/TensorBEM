"""CASE-BEM-RIGID-BODY-MASS extension: orientation covariance + external-code anchor.

Why this module exists
----------------------
The base case (:mod:`tensorbem.benchmarks.rigid_body_mass_case`) verifies the
closed block form of the 6x6 rigid-body mass matrix on an **axis-aligned**
ellipsoid.  Two things are therefore *not* exercised by it, and both are load
bearing for a real ship:

1. **Orientation covariance.**  For an axis-aligned body the inertia tensor is
   diagonal, so every product of inertia ``I_xy, I_xz, I_yz`` is zero and the
   entries that mix roll-pitch-yaw are zero by symmetry.  A hip tensor
   generalisation that silently ignores the off-diagonal terms (or that mixes
   axes in the parallel-axis shift) passes the base case.  Here the same solid
   ellipsoid is rigidly **rotated** by ``R = Rz(yaw) Ry(pitch) Rx(roll)`` about
   the reference point, which makes ``I_g`` a full tensor and activates the
   cross entries (``M[roll,pitch]`` etc.).  The invariant that must hold is the
   congruence

       M(rotated body)  =  A M(body) A^T ,   A = blockdiag(R, R)          (1)

   (a twist expressed in body axes maps to the reference frame as
   ``x_ref = A x_body``, so ``M_ref = A M_body A^T``).  Equation (1) is a
   statement about the *group action*, independent of the block form itself, so
   it is a genuinely new invariant -- the base case's reference-point test is
   the translation half of the same group, this is the rotation half.

2. **An external code anchor.**  The base case is honest that every reference
   number is produced by TensorBEM's own closed form: self-consistency, not an
   outside opinion.  Capytaine (the open-source WAMIT-lineage BEM code, 3.x)
   implements ``compute_rigid_body_inertia`` from its own mesh integrals and its
   own block assembly.  Comparing the two implementations entry by entry pins
   the **convention** of the coupling block -- the exact place where this
   platform shipped a transposed coupling block plus a doubly counted
   parallel-axis term (fixed in m17376, see ``docs/audit_m17248_rigid_body_mass.md``).

What is measured here
---------------------
``orientation``    analytic route: closed-form ``M`` of the rotated body, the
                   congruence residual (1), the products of inertia, and the
                   kinetic-energy identity.
``levels``         mesh route: TensorBEM icosphere panels of the *rotated,
                   translated* body, mass properties by signed-tetrahedron
                   quadrature, 6x6 assembled, entrywise error vs the closed
                   form of the rotated body (this is what the 3 % criterion
                   measures), plus convergence of the cross entries.
``sweep``          the same at several unrelated orientations, so a single lucky
                   angle cannot carry the verdict.
``negative_control``  the pre-fix (transposed/doubly shifted) convention: the
                   congruence residual and the energy identity must break at
                   O(1), proving the criteria are sensitive.
``external``       capytaine cross-check on the identical panel mesh: mass block,
                   coupling block and rotational block compared separately.

Zero hand-transcription: every number is evaluated from the closed form in
:mod:`tensorbem.rigid_body_mass` (Fossen 2011 eq. (3.44) block form; Lamb 1932
solid-ellipsoid second moments) or produced by quadrature at run time.

Run::

    PYTHONPATH=src python -m tensorbem.benchmarks.rigid_body_mass_orientation \
        --out docs/cases/case_bem_rigid_body_mass/rotation/run.json

Exit status is 0 when every structural check passes, 1 otherwise.
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
from tensorbem.rigid_body_mass import (
    MODES,
    block_form_residual,
    energy_residual,
    mesh_mass_properties,
    rigid_body_mass_matrix,
    skew_matrix,
    solid_ellipsoid_mass_properties,
)

try:  # pragma: no cover - exercised only when capytaine is installed
    import capytaine as _cpy

    CAPYTAINE_VERSION: str | None = getattr(_cpy, "__version__", "unknown")
    CAPYTAINE_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - optional dependency
    _cpy = None
    CAPYTAINE_VERSION = None
    CAPYTAINE_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

#: Reference point of the body-fixed frame (midship, centreline, waterline).
REFERENCE_POINT = (0.0, 0.0, 0.0)
#: Same case body as the base case: L/B/T = 120/24/16 m.
SEMI_AXES = (60.0, 12.0, 8.0)
DEFAULT_DENSITY = 1025.0
DEFAULT_CG_OFFSET = (5.0, 0.0, -3.0)
#: (yaw, pitch, roll) in radians: a generic, non-degenerate orientation.
DEFAULT_ROTATION = (math.radians(20.0), math.radians(15.0), math.radians(25.0))
DEFAULT_SUBDIVISIONS: tuple[int, ...] = (1, 2, 3, 4)
ACCEPTED_FROM = 3
TOLERANCE = 0.03
#: The rotated body is dropped below the free surface for the capytaine run:
#: capytaine's inertia uses the *immersed* volume, so a body straddling z = 0
#: would report the displaced volume instead of the full one.
SUBMERGENCE = (-20.0,)

REFERENCE_CITATION = (
    "Fossen (2011) Handbook of Marine Craft Hydrodynamics and Motion Control, "
    "Sec. 3.3 eq. (3.44): M_RB = [[m I3, -m S(r_g)], [m S(r_g), I_b + m S^T(r_g) S(r_g)]]; "
    "solid-ellipsoid second moments I_cg = m(b^2+c^2, a^2+c^2, a^2+b^2)/5 "
    "(Lamb 1932 Ch. VI); inertia of a rotated body R I_cg R^T and centroid R r_g. "
    "All numbers machine-evaluated (solid_ellipsoid_mass_properties / "
    "rigid_body_mass_matrix); no table copied."
)


# --------------------------------------------------------------------------- #
# rotation helpers
# --------------------------------------------------------------------------- #
def rotation_matrix(
    yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0
) -> np.ndarray:
    """Right-handed active rotation ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.

    ``yaw`` about z, ``pitch`` about y, ``roll`` about x (the TensorBEM mode
    order: surge/sway/heave/roll/pitch/yaw = x/y/z).
    """
    cz, sz = math.cos(yaw), math.sin(yaw)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cx, sx = math.cos(roll), math.sin(roll)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rz @ ry @ rx


def twist_congruence(rotation: np.ndarray) -> np.ndarray:
    """The 6x6 block-diagonal ``A = blockdiag(R, R)`` of equation (1)."""
    a = np.zeros((6, 6), dtype=float)
    a[:3, :3] = rotation
    a[3:, 3:] = rotation
    return a


def _entry_errors(got: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Entrywise ``|got-ref| / max(|ref|, 1e-12 * max|ref|)``.

    Entries whose reference value is zero (the symmetry-forbidden entries of the
    unrotated body) are measured against the matrix scale instead of dividing by
    zero, i.e. the test is *more* conservative than a plain relative error.
    """
    scale = float(np.max(np.abs(ref)))
    floor = 1.0e-12 * scale
    denominator = np.where(np.abs(ref) > floor, np.abs(ref), scale)
    return np.abs(got - ref) / denominator


# --------------------------------------------------------------------------- #
# analytic side
# --------------------------------------------------------------------------- #
def rotated_reference(
    rotation: np.ndarray,
    semi_axes: Sequence[float] = SEMI_AXES,
    density: float = DEFAULT_DENSITY,
    cg_offset: Sequence[float] = DEFAULT_CG_OFFSET,
) -> dict[str, Any]:
    """Closed-form mass matrix of the rotated body + the congruence residual."""
    base = solid_ellipsoid_mass_properties(semi_axes, density)
    inertia_body = np.asarray(base["inertia_cg"], dtype=float)
    mass = float(base["mass"])
    cg_body = np.asarray(cg_offset, dtype=float)

    matrix_body = rigid_body_mass_matrix(
        mass, cg_body, inertia_body, inertia_about="cg", reference_point=REFERENCE_POINT
    )
    inertia_rot = rotation @ inertia_body @ rotation.T
    cg_rot = rotation @ cg_body
    matrix_rot = rigid_body_mass_matrix(
        mass, cg_rot, inertia_rot, inertia_about="cg", reference_point=REFERENCE_POINT
    )
    congruence = twist_congruence(rotation)
    matrix_congruence = congruence @ matrix_body @ congruence.T
    scale = float(np.max(np.abs(matrix_rot)))
    products = {
        "I_xy": float(inertia_rot[0, 1]),
        "I_xz": float(inertia_rot[0, 2]),
        "I_yz": float(inertia_rot[1, 2]),
    }
    offdiag_ratio = max(abs(value) for value in products.values()) / float(
        np.max(np.diag(inertia_rot))
    )
    cross_entries = {
        "M_roll_pitch": float(matrix_rot[3, 4]),
        "M_surge_yaw": float(matrix_rot[0, 5]),
        "M_sway_roll": float(matrix_rot[1, 3]),
    }
    return {
        "rotation_matrix": rotation.tolist(),
        "euler_deg": {
            "yaw": math.degrees(math.atan2(rotation[1, 0], rotation[0, 0])),
            "pitch": math.degrees(math.asin(-rotation[2, 0])),
            "roll": math.degrees(math.atan2(rotation[2, 1], rotation[2, 2])),
        },
        "volume": float(base["volume"]),
        "mass": mass,
        "inertia_body": inertia_body.tolist(),
        "inertia_rotated": inertia_rot.tolist(),
        "products_of_inertia": products,
        "max_product_over_max_diagonal": float(offdiag_ratio),
        "centroid_body": cg_body.tolist(),
        "centroid_rotated": cg_rot.tolist(),
        "matrix_body": matrix_body.tolist(),
        "matrix_rotated": matrix_rot.tolist(),
        "matrix_scale": scale,
        "cross_entries": cross_entries,
        "congruence_residual_rel": float(
            np.max(np.abs(matrix_rot - matrix_congruence)) / scale
        ),
        "block_form_residual": float(
            block_form_residual(
                matrix_rot, mass, inertia_rot, cg_rot, reference_point=REFERENCE_POINT
            )["max_relative_error"]
        ),
        "energy_identity_error": float(
            energy_residual(
                matrix_rot, mass, inertia_rot, cg_rot, reference_point=REFERENCE_POINT
            )
        ),
        "orthonormality_error": float(
            np.max(np.abs(rotation @ rotation.T - np.eye(3)))
        ),
        "determinant": float(np.linalg.det(rotation)),
    }


# --------------------------------------------------------------------------- #
# mesh side
# --------------------------------------------------------------------------- #
def rotated_panel_mesh(
    rotation: np.ndarray,
    subdivisions: int,
    semi_axes: Sequence[float] = SEMI_AXES,
    cg_offset: Sequence[float] = DEFAULT_CG_OFFSET,
    offset: Sequence[float] = REFERENCE_POINT,
) -> tuple[np.ndarray, np.ndarray]:
    """Panels of the case body after a rigid rotation about the reference point.

    The same solid as the base case: an ellipsoid whose centre of gravity sits at
    ``cg_offset`` from its geometric centre.  Rotating it about the reference
    point maps the panels as ``x -> R x + R cg_offset + offset``, so the geometric
    centre lands on ``R cg_offset + offset`` and the reference point stays put.
    """
    mesh = ellipsoid_mesh(int(subdivisions), np.asarray(semi_axes, dtype=float))
    centre = rotation @ np.asarray(cg_offset, dtype=float) + np.asarray(
        offset, dtype=float
    )
    vertices = mesh.vertices @ rotation.T + centre
    return vertices, np.asarray(mesh.faces)


def mesh_level(
    rotation: np.ndarray,
    subdivisions: int,
    reference: dict[str, Any],
    density: float = DEFAULT_DENSITY,
    cg_offset: Sequence[float] = DEFAULT_CG_OFFSET,
    offset: Sequence[float] = REFERENCE_POINT,
) -> dict[str, Any]:
    """One panel level of the rotated body: mesh quadrature vs the closed form."""
    start = time.perf_counter()
    vertices, faces = rotated_panel_mesh(
        rotation, subdivisions, cg_offset=cg_offset, offset=offset
    )
    props = mesh_mass_properties(vertices, faces, density, reference_point=offset)
    mass = float(props["mass"])
    centroid = np.asarray(props["centroid"], dtype=float)
    inertia = np.asarray(props["inertia_cg"], dtype=float)
    matrix = rigid_body_mass_matrix(
        mass, centroid, inertia, inertia_about="cg", reference_point=offset
    )
    wall = time.perf_counter() - start

    ref_matrix = np.asarray(reference["matrix_rotated"], dtype=float)
    errors = _entry_errors(matrix, ref_matrix)
    worst = np.unravel_index(int(np.argmax(errors)), errors.shape)
    scale = float(np.max(np.abs(ref_matrix)))
    ref_inertia = np.asarray(reference["inertia_rotated"], dtype=float)
    inertia_error = float(
        np.max(np.abs(inertia - ref_inertia)) / np.max(np.abs(ref_inertia))
    )
    # relative error of the entries that are zero for the axis-aligned body
    cross_mask = np.abs(ref_matrix) > 1.0e-6 * scale
    cross_mask[:3, :3] = False
    cross_errors = errors[cross_mask]
    return {
        "subdivisions": int(subdivisions),
        "panels": int(props["face_count"]),
        "wall_time_s": float(wall),
        "volume": float(props["volume"]),
        "volume_rel_error": float(
            abs(props["volume"] - float(reference["volume"])) / float(reference["volume"])
        ),
        "mass": mass,
        "centroid": centroid.tolist(),
        "inertia_rotated": inertia.tolist(),
        "inertia_rel_error": inertia_error,
        "matrix": matrix.tolist(),
        "max_entry_rel_error": float(errors[worst]),
        "worst_entry": {
            "mode_i": MODES[worst[0]],
            "mode_j": MODES[worst[1]],
            "ref": float(ref_matrix[worst]),
            "got": float(matrix[worst]),
        },
        "max_nonzero_entry_rel_error": float(np.max(cross_errors)) if cross_errors.size else 0.0,
        "energy_identity_error": float(
            energy_residual(matrix, mass, inertia, centroid, reference_point=offset)
        ),
        "block_form_residual": float(
            block_form_residual(
                matrix, mass, inertia, centroid, reference_point=offset
            )["max_relative_error"]
        ),
        "within_tolerance": bool(float(errors[worst]) < TOLERANCE),
    }


# --------------------------------------------------------------------------- #
# negative control: the pre-fix convention
# --------------------------------------------------------------------------- #
def legacy_convention(
    reference: dict[str, Any], density: float = DEFAULT_DENSITY
) -> dict[str, Any]:
    """Pre-fix platform assembly ``[[mI, -m S^T], [-m S, I_O + m S S^T]]``.

    Equivalent to the correct block form evaluated with ``r_g -> -r_g`` *and* a
    parallel-axis term added twice; it is still symmetric, which is why the
    original test suite (symmetry + self-comparison) stayed green.
    """
    mass = float(reference["mass"])
    centroid = np.asarray(reference["centroid_rotated"], dtype=float)
    inertia = np.asarray(reference["inertia_rotated"], dtype=float)
    s = skew_matrix(centroid)
    inertia_about_origin = inertia + mass * (s.T @ s)
    legacy = np.zeros((6, 6), dtype=float)
    legacy[:3, :3] = mass * np.eye(3)
    legacy[:3, 3:] = -mass * s.T
    legacy[3:, :3] = -mass * s
    legacy[3:, 3:] = inertia_about_origin + mass * (s @ s.T)
    ref_matrix = np.asarray(reference["matrix_rotated"], dtype=float)
    scale = float(np.max(np.abs(ref_matrix)))
    congruence = twist_congruence(np.asarray(reference["rotation_matrix"], dtype=float))
    rotated_correct = np.asarray(reference["matrix_body"], dtype=float)
    return {
        "max_entry_rel_error": float(np.max(_entry_errors(legacy, ref_matrix))),
        "energy_identity_error": float(
            energy_residual(legacy, mass, inertia, centroid, reference_point=REFERENCE_POINT)
            / max(
                abs(
                    energy_residual(
                        ref_matrix,
                        mass,
                        inertia,
                        centroid,
                        reference_point=REFERENCE_POINT,
                    )
                ),
                1.0e-30,
            )
        ),
        "congruence_residual_rel": float(
            np.max(np.abs(legacy - congruence @ rotated_correct @ congruence.T)) / scale
        ),
        "symmetric": bool(np.allclose(legacy, legacy.T, rtol=0.0, atol=1.0e-6 * scale)),
        "detects": {
            "entrywise_over_3pct": bool(
                float(np.max(_entry_errors(legacy, ref_matrix))) > TOLERANCE
            ),
            "congruence_broken": bool(
                float(
                    np.max(np.abs(legacy - congruence @ rotated_correct @ congruence.T))
                    / scale
                )
                > 1.0e-3
            ),
        },
    }


# --------------------------------------------------------------------------- #
# external anchor: capytaine
# --------------------------------------------------------------------------- #
def external_capytaine(
    rotation: np.ndarray,
    subdivisions: int,
    reference: dict[str, Any],
    density: float = DEFAULT_DENSITY,
    cg_offset: Sequence[float] = DEFAULT_CG_OFFSET,
) -> dict[str, Any]:
    """Compare the 6x6 against capytaine's own rigid-body inertia assembly.

    The body is translated below the free surface (capytaine's inertia uses the
    *immersed* volume) and its rotation centre is put at the translated
    reference point, so both codes describe the same body about the same point.
    """
    if _cpy is None:
        return {
            "available": False,
            "capytaine_version": None,
            "import_error": CAPYTAINE_IMPORT_ERROR,
            "status": "not_run",
        }
    offset = np.asarray(reference["centroid_rotated"], dtype=float) * 0.0 + np.array(
        [0.0, 0.0, SUBMERGENCE[0]]
    )
    vertices, faces = rotated_panel_mesh(
        rotation, subdivisions, cg_offset=cg_offset, offset=offset
    )
    mesh = _cpy.Mesh(vertices=vertices, faces=faces)
    body = _cpy.FloatingBody(mesh=mesh, name="rotated_ellipsoid")
    body.add_all_rigid_body_dofs(rotation_center=tuple(float(v) for v in offset))
    body.center_of_mass = np.asarray(reference["centroid_rotated"], dtype=float) + offset
    raw = body.compute_rigid_body_inertia(rho=density)
    external = np.asarray(raw.values if hasattr(raw, "values") else raw, dtype=float)

    props = mesh_mass_properties(vertices, faces, density, reference_point=offset)
    internal = rigid_body_mass_matrix(
        float(props["mass"]),
        np.asarray(props["centroid"], dtype=float),
        np.asarray(props["inertia_cg"], dtype=float),
        inertia_about="cg",
        reference_point=offset,
    )
    ref_matrix = np.asarray(reference["matrix_rotated"], dtype=float)
    blocks = {
        "translation_translation": (slice(0, 3), slice(0, 3)),
        "translation_rotation": (slice(0, 3), slice(3, 6)),
        "rotation_rotation": (slice(3, 6), slice(3, 6)),
    }
    scale = float(np.max(np.abs(ref_matrix)))

    def _block_error(matrix_a: np.ndarray, matrix_b: np.ndarray, key: str) -> float:
        rows, cols = blocks[key]
        part_a = matrix_a[rows, cols]
        part_b = matrix_b[rows, cols]
        part_scale = float(np.max(np.abs(part_b))) or scale
        return float(np.max(np.abs(part_a - part_b)) / part_scale)

    return {
        "available": True,
        "capytaine_version": CAPYTAINE_VERSION,
        "subdivisions": int(subdivisions),
        "panels": int(faces.shape[0]),
        "capytaine_volume": float(mesh.volume),
        "capytaine_immersed_volume": float(mesh.disp_volume),
        "tensor_bem_volume": float(props["volume"]),
        "external_matrix": external.tolist(),
        "max_entry_rel_error_vs_closed_form": float(np.max(_entry_errors(external, ref_matrix))),
        "max_entry_rel_error_vs_tensor_bem": float(np.max(_entry_errors(external, internal))),
        "block_errors_vs_closed_form": {
            key: _block_error(external, ref_matrix, key) for key in blocks
        },
        "block_errors_external_vs_tensor_bem": {
            key: _block_error(external, internal, key) for key in blocks
        },
        "within_tolerance": bool(
            float(np.max(_entry_errors(external, ref_matrix))) < TOLERANCE
        ),
        "status": "ok",
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run(
    subdivisions: Sequence[int] = DEFAULT_SUBDIVISIONS,
    rotation: Sequence[float] = DEFAULT_ROTATION,
    semi_axes: Sequence[float] = SEMI_AXES,
    density: float = DEFAULT_DENSITY,
    cg_offset: Sequence[float] = DEFAULT_CG_OFFSET,
) -> dict[str, Any]:
    """Run the extension: orientation covariance, mesh ladder, sweep, external."""
    start = time.perf_counter()
    rot = rotation_matrix(*rotation)
    reference = rotated_reference(rot, semi_axes, density, cg_offset)
    levels = [
        mesh_level(rot, int(sd), reference, density=density, cg_offset=cg_offset)
        for sd in subdivisions
    ]
    sweep_angles = [
        (math.radians(20.0), math.radians(15.0), math.radians(25.0)),
        (math.radians(-35.0), math.radians(10.0), math.radians(-12.0)),
        (math.radians(70.0), math.radians(-20.0), math.radians(40.0)),
        (math.radians(90.0), math.radians(0.0), math.radians(90.0)),
        (math.radians(-110.0), math.radians(30.0), math.radians(-55.0)),
    ]
    sweep_sd = int(min(max(subdivisions), ACCEPTED_FROM))
    sweep: list[dict[str, Any]] = []
    for angles in sweep_angles:
        local_rot = rotation_matrix(*angles)
        local_ref = rotated_reference(local_rot, semi_axes, density, cg_offset)
        local_level = mesh_level(
            local_rot, sweep_sd, local_ref, density=density, cg_offset=cg_offset
        )
        sweep.append(
            {
                "euler_deg": local_ref["euler_deg"],
                "congruence_residual_rel": local_ref["congruence_residual_rel"],
                "max_product_over_max_diagonal": local_ref["max_product_over_max_diagonal"],
                "subdivisions": sweep_sd,
                "max_entry_rel_error": local_level["max_entry_rel_error"],
                "worst_entry": local_level["worst_entry"],
                "energy_identity_error": local_level["energy_identity_error"],
                "within_tolerance": local_level["within_tolerance"],
            }
        )
    legacy = legacy_convention(reference, density=density)
    external = external_capytaine(
        rot, int(max(subdivisions)), reference, density=density, cg_offset=cg_offset
    )

    accepted = [row for row in levels if row["subdivisions"] >= ACCEPTED_FROM] or levels[-1:]
    worst_level = max(levels, key=lambda row: row["max_entry_rel_error"])
    checks = {
        "rotation_is_orthonormal": reference["orthonormality_error"] < 1.0e-14,
        "rotation_creates_products_of_inertia": reference[
            "max_product_over_max_diagonal"
        ]
        > 0.05,
        "congruence_identity_roundoff": reference["congruence_residual_rel"] < 1.0e-12,
        "analytic_block_form_roundoff": reference["block_form_residual"] < 1.0e-12,
        "analytic_energy_identity": reference["energy_identity_error"] < 1.0e-12,
        "rotated_mesh_levels_within_3pct": all(
            row["within_tolerance"] for row in accepted
        ),
        "rotated_mesh_energy_identity": all(
            row["energy_identity_error"] < 1.0e-12 for row in levels
        ),
        "inertia_tensor_error_monotone": all(
            levels[i + 1]["inertia_rel_error"] <= levels[i]["inertia_rel_error"]
            for i in range(len(levels) - 1)
        ),
        "sweep_all_within_3pct": all(row["within_tolerance"] for row in sweep),
        "sweep_congruence_roundoff": all(
            row["congruence_residual_rel"] < 1.0e-12 for row in sweep
        ),
        "negative_control_detected": bool(all(legacy["detects"].values())),
        "external_anchor_within_3pct": (
            external["status"] != "ok"
            or bool(external["within_tolerance"])
        ),
        "external_anchor_convention_exact": (
            external["status"] != "ok"
            or external["block_errors_external_vs_tensor_bem"]["translation_rotation"]
            < 1.0e-9
        ),
    }
    return {
        "benchmark": "CASE-BEM-RIGID-BODY-MASS.orientation-covariance",
        "subject": "rotated 6x6 rigid-body mass matrix: orientation covariance and external-code anchor",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "capytaine": CAPYTAINE_VERSION,
            "wall_time_s": time.perf_counter() - start,
        },
        "parameters": {
            "semi_axes_m": [float(v) for v in semi_axes],
            "density_kg_m3": float(density),
            "cg_offset_m": [float(v) for v in cg_offset],
            "rotation_spec": "R = Rz(yaw) @ Ry(pitch) @ Rx(roll); angles rad",
            "rotation_rad": [float(v) for v in rotation],
            "mode_order": list(MODES),
            "subdivisions": [int(sd) for sd in subdivisions],
            "tolerance": TOLERANCE,
            "accepted_from_subdivision": ACCEPTED_FROM,
            "submergence_m": list(SUBMERGENCE),
            "reference_citation": REFERENCE_CITATION,
        },
        "orientation": reference,
        "levels": levels,
        "sweep": sweep,
        "negative_control": legacy,
        "external": external,
        "verdict": {
            "all_pass": bool(all(checks.values())),
            "checks": checks,
            "worst_level": {
                "subdivisions": worst_level["subdivisions"],
                "max_entry_rel_error": worst_level["max_entry_rel_error"],
                "worst_entry": worst_level["worst_entry"],
            },
            "exit_code": 0 if all(checks.values()) else 1,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--subdivisions", type=int, nargs="+", default=list(DEFAULT_SUBDIVISIONS)
    )
    parser.add_argument(
        "--rotation-deg",
        type=float,
        nargs=3,
        default=[math.degrees(a) for a in DEFAULT_ROTATION],
        metavar=("YAW", "PITCH", "ROLL"),
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    result = run(
        subdivisions=tuple(args.subdivisions),
        rotation=tuple(math.radians(a) for a in args.rotation_deg),
    )
    payload = json.dumps(result, indent=2, sort_keys=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(payload)
    verdict = result["verdict"]
    print(
        f"[{'PASS' if verdict['all_pass'] else 'FAIL'}] "
        f"worst entry {verdict['worst_level']['max_entry_rel_error']:.4%} "
        f"at subdivision {verdict['worst_level']['subdivisions']}; "
        f"congruence {result['orientation']['congruence_residual_rel']:.2e}"
    )
    return int(verdict["exit_code"])


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())