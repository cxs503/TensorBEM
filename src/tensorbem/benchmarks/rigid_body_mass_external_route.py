"""Route "T": an *external* referee for the rigid-body mass properties.

Why a fifth route
-----------------
The anchor's evidence chain hangs on one operation: turn a closed surface into
``(m, r_g, I_g)`` and assemble the 6x6 block form.  Routes R (closed form),
M (TensorBEM's signed-tetrahedron quadrature) and P (the momentum definitions)
all feed that assembly from *TensorBEM's own* integration of the volume, and
route X (capytaine) assembles the added mass ``A``, not the rigid mass ``M_RB``.
A defect in TensorBEM's volume integration, or in the mesh generator that feeds
it, would therefore be invisible to the whole chain: the reference would drift
together with the object under test.

This module replaces both layers with third-party code:

* the discretisation -- :func:`trimesh.creation.uv_sphere` (longitude/latitude
  parametrisation with two pole fans) instead of
  :func:`tensorbem.geometry3d.ellipsoid_mesh` (icosphere topology);
* the integration -- trimesh's Mirtich (1996) projection route instead of the
  signed-tetrahedron quadrature of
  :func:`tensorbem.benchmarks.case_rigid_body_mass.tetra_mass_properties`.

Three things are then asked, in increasing order of strength:

1. **layer attribution** -- on the *same* TensorBEM mesh the two integrators
   must agree to round-off, which pins the accepted-level error on the
   surface discretisation (panelisation) and not on the integration;
2. **independent route** -- on its *own* mesh, with its *own* integration, the
   external route must reproduce the closed-form 6x6 block form inside the
   anchor tolerance (``< 3 %``), i.e. the anchor's claim survives a
   discretisation it never saw;
3. **external referee of the guard** -- :func:`trimesh.Trimesh.is_watertight`
   must agree with TensorBEM's :func:`tensorbem.rigid_body_mass.
   mesh_topology_report` on meshes corrupted on purpose (open patch, flipped
   face, duplicated face, inward winding).  The guard was added as a
   remediation in :mod:`docs/case_bem_rigid_body_mass/ADVERSARIAL_REVIEW.md`;
   an external implementation checking it is what keeps that remediation from
   being self-certified.

The same route is finally applied to the *ship-shaped* Wigley solid, where the
anchor's box/ellipsoid bodies have no ship-like coupling blocks.

Dependency
----------
``trimesh`` is an *optional* verification dependency
(``pip install tensorbem[external-verification]``).  When it is missing the
section reports ``status="not_run"`` and contributes ``not_run`` entries to the
anchor verdict -- never a pass (same discipline as the ``--no-rao`` probe).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from tensorbem.benchmarks import case_rigid_body_mass as base_case
from tensorbem.rigid_body_mass import (
    MODES,
    mesh_topology_report,
    rigid_body_mass_matrix,
)

__all__ = [
    "DEFAULT_SUBDIVISIONS",
    "DEFAULT_UV_COUNTS",
    "EXTERNAL_ACCEPTED_COUNTS",
    "ROUNDOFF_LIMIT",
    "TOLERANCE",
    "available",
    "external_mass_properties",
    "integrator_cross_check",
    "independent_mesh_ladder",
    "run",
    "self_calibration",
    "ship_shape_check",
    "topology_referee",
]

TOLERANCE = 0.03
#: Two independent integrators on the *same* triangles must reproduce the same
#: mass properties; anything above round-off would mean one of them is a bias.
ROUNDOFF_LIMIT = 1.0e-9
#: TensorBEM ellipsoid mesh levels used for the layer-attribution probe.
DEFAULT_SUBDIVISIONS = (1, 2, 3, 4)
#: ``(count_latitude, 2*count_latitude)`` of the external UV sphere.  Panels per
#: level are ``2*n*(2*n-1)+2``: 992 / 4032 / 16256 / 65408 -- a different panel
#: count *and* a different distribution than the icosphere's ``20*4**s``.
DEFAULT_UV_COUNTS = (16, 32, 64, 128)
#: Predeclared acceptance levels of the external route (>= 32 -> >= 4032 panels).
EXTERNAL_ACCEPTED_COUNTS = (32, 64, 128)

CHECKS = (
    "external_integrator_roundoff",
    "external_mesh_route_within_tolerance",
    "external_ship_shape_within_tolerance",
    "external_topology_guard_agrees",
    "external_guard_is_load_bearing",
    "external_route_rejects_legacy_coupling",
)


# ---------------------------------------------------------------------------
# the external integrator
# ---------------------------------------------------------------------------
def _trimesh() -> Any:
    """Import ``trimesh`` lazily, with an actionable error."""
    try:  # pragma: no cover - the missing-dependency path is covered by a test
        import trimesh  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "route T needs the optional verification dependency trimesh "
            "(pip install 'tensorbem[external-verification]'); the anchor must "
            "then be reported as INCOMPLETE, not as PASS"
        ) from exc
    return trimesh


def available() -> bool:
    """Whether the external referee is installed in this environment."""
    try:
        _trimesh()
    except RuntimeError:
        return False
    return True


def external_mass_properties(
    vertices: np.ndarray,
    faces: np.ndarray,
    density: float,
    *,
    require_watertight: bool = True,
) -> dict[str, Any]:
    """Volume, centroid and inertia of a closed triangle mesh by *trimesh*.

    trimesh integrates the second moments with Mirtich's (1996) projection
    algorithm, which shares no line of code and no quadrature rule with
    TensorBEM's signed-tetrahedron route.

    Parameters
    ----------
    vertices, faces:
        ``(V, 3)`` float array and ``(F, 3)`` integer array.  Vertex order and
        face winding are used as given (``process=False``), so an inverted
        surface stays inverted instead of being silently repaired.
    density:
        Uniform density (kg/m^3).  trimesh's own integration is unit-density;
        the returned tensors are scaled here.
    require_watertight:
        When true (default) a non-watertight surface raises ``ValueError`` --
        the same domain restriction TensorBEM's guard enforces, applied by an
        external implementation.

    Returns
    -------
    dict
        ``volume``, ``mass``, ``center_of_gravity``, ``inertia_about_cg``,
        ``inertia_about_origin``, ``panels``, ``watertight``,
        ``winding_positive``.
    """
    trimesh = _trimesh()
    verts = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=int)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("vertices must have shape (V, 3)")
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")
    if not np.all(np.isfinite(verts)):
        raise ValueError("vertices must be finite")
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError("density must be finite and positive")

    mesh = trimesh.Trimesh(vertices=verts, faces=tris, process=False)
    watertight = bool(mesh.is_watertight)
    if require_watertight and not watertight:
        raise ValueError(
            "surface is not watertight; mass-property integration is undefined "
            "(external referee: trimesh.Trimesh.is_watertight)"
        )
    props = mesh.mass_properties  # unit density
    volume = float(props.volume)
    # trimesh reports the centroid of the unit-density solid, i.e. the
    # geometric centroid, and the inertia tensor about it.
    centroid = np.asarray(props.center_mass, dtype=float)
    inertia_cg = float(density) * np.asarray(props.inertia, dtype=float)
    mass = float(density) * volume
    d = centroid
    inertia_origin = inertia_cg + mass * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return {
        "volume": volume,
        "mass": mass,
        "center_of_gravity": centroid.tolist(),
        "inertia_about_cg": inertia_cg.tolist(),
        "inertia_about_origin": inertia_origin.tolist(),
        "panels": int(tris.shape[0]),
        "watertight": watertight,
        "winding_positive": bool(volume > 0.0),
    }


def _matrix(mass: float, cg: Sequence[float], inertia: np.ndarray) -> np.ndarray:
    """Assemble ``M_RB`` with the shipped assembly (errors belong to the inputs)."""
    return rigid_body_mass_matrix(
        float(mass), np.asarray(cg, dtype=float).reshape(3), np.asarray(inertia, dtype=float)
    )


def _max_entry_error(got: np.ndarray, ref: np.ndarray) -> dict[str, Any]:
    """Entry-relative error, identical metric to the anchor's acceptance test."""
    err = np.asarray(base_case.relative_error_matrix(ref, got), dtype=float)
    flat = int(np.argmax(err))
    i, j = divmod(flat, err.shape[1])
    return {
        "max_rel_error": float(err.reshape(-1)[flat]),
        "worst_entry": f"{MODES[i]}-{MODES[j]}",
    }


# ---------------------------------------------------------------------------
# 0. the referee must first calibrate itself
# ---------------------------------------------------------------------------
def self_calibration(density: float = 1025.0, side: float = 3.0) -> dict[str, Any]:
    """Cube of side ``side``: trimesh vs the textbook ``I_cg = m L^2/6``.

    Before the external route judges TensorBEM it has to be able to reproduce a
    number nobody had to integrate: a cube mesh built by trimesh itself, whose
    mass properties are ``m = rho L^3``, ``r_g = centre`` and
    ``I_cg = (m L^2 / 6) I3``.
    """
    trimesh = _trimesh()
    mesh = trimesh.creation.box(extents=(side, side, side))
    verts = np.asarray(mesh.vertices, dtype=float) + np.array([1.5, -2.5, 0.75])
    got = external_mass_properties(verts, np.asarray(mesh.faces, dtype=int), density)
    mass = density * side**3
    inertia = np.eye(3) * (mass * side * side / 6.0)
    return {
        "side_m": float(side),
        "density_kg_m3": float(density),
        "panels": got["panels"],
        "watertight": got["watertight"],
        "volume_rel_error": abs(got["volume"] - side**3) / side**3,
        "mass_rel_error": abs(got["mass"] - mass) / mass,
        "centroid_abs_error_m": float(
            np.max(np.abs(np.asarray(got["center_of_gravity"]) - np.array([1.5, -2.5, 0.75])))
        ),
        "inertia_rel_error": float(
            np.max(np.abs(np.asarray(got["inertia_about_cg"]) - inertia)) / np.max(np.abs(inertia))
        ),
    }


# ---------------------------------------------------------------------------
# 1. layer attribution: same triangles, two integrators
# ---------------------------------------------------------------------------
def integrator_cross_check(
    subdivisions: Sequence[int] = DEFAULT_SUBDIVISIONS,
) -> dict[str, Any]:
    """trimesh vs TensorBEM on the *same* ellipsoid mesh.

    Isolates the integration layer: if the two integrators agree to round-off
    on identical triangles, the error budget at the accepted level belongs to
    the surface discretisation (panelisation) and not to the quadrature.
    """
    from tensorbem.geometry3d import ellipsoid_mesh  # noqa: PLC0415

    axes = np.asarray(base_case.ELLIPSOID_SEMI_AXES, dtype=float)
    centre = np.asarray(base_case.ELLIPSOID_CENTRE, dtype=float)
    density = float(base_case.ELLIPSOID_DENSITY)
    ref_props = base_case.ellipsoid_reference()
    ref_matrix = _matrix(
        float(ref_props["mass"]),
        np.asarray(ref_props["center_of_gravity"], dtype=float),
        np.asarray(ref_props["inertia_about_cg"], dtype=float),
    )

    levels = []
    for sub in subdivisions:
        mesh = ellipsoid_mesh(int(sub), axes)
        verts = np.asarray(mesh.vertices, dtype=float) + centre
        faces = np.asarray(mesh.faces, dtype=int)
        ext = external_mass_properties(verts, faces, density)
        ten = base_case.tetra_mass_properties(verts, faces, density)
        ext_matrix = _matrix(
            ext["mass"],
            np.asarray(ext["center_of_gravity"], dtype=float),
            np.asarray(ext["inertia_about_cg"], dtype=float),
        )
        ten_matrix = _matrix(
            ten["mass"],
            np.asarray(ten["center_of_gravity"], dtype=float),
            np.asarray(ten["inertia_about_cg"], dtype=float),
        )
        agreement = _max_entry_error(ext_matrix, ten_matrix)
        levels.append(
            {
                "subdivisions": int(sub),
                "panels": int(ext["panels"]),
                "external_watertight": bool(ext["watertight"]),
                "volume_rel_difference": abs(ext["volume"] - ten["volume"]) / abs(ten["volume"]),
                "inertia_rel_difference": float(
                    np.max(
                        np.abs(
                            np.asarray(ext["inertia_about_cg"])
                            - np.asarray(ten["inertia_about_cg"])
                        )
                    )
                    / np.max(np.abs(np.asarray(ref_props["inertia_about_cg"])))
                ),
                "matrix_max_entry_difference": agreement["max_rel_error"],
                "worst_entry": agreement["worst_entry"],
                # each integrator's own distance to the closed form
                "external_vs_closed_form": _max_entry_error(ext_matrix, ref_matrix)[
                    "max_rel_error"
                ],
                "tensorbem_vs_closed_form": _max_entry_error(ten_matrix, ref_matrix)[
                    "max_rel_error"
                ],
            }
        )
    worst = max(row["matrix_max_entry_difference"] for row in levels)
    worst_bias = max(
        abs(row["external_vs_closed_form"] - row["tensorbem_vs_closed_form"]) for row in levels
    )
    return {
        "method": "trimesh (Mirtich projection) vs tensorbem signed-tetrahedron, same vertices+faces",
        "roundoff_limit": ROUNDOFF_LIMIT,
        "levels": levels,
        "worst_matrix_difference": float(worst),
        "worst_error_budget_difference": float(worst_bias),
        "checks": {
            "external_integrator_roundoff": bool(worst < ROUNDOFF_LIMIT),
        },
    }


# ---------------------------------------------------------------------------
# 2. independent route: external mesh + external integrator vs the closed form
# ---------------------------------------------------------------------------
def _fit_slope(panels: Sequence[float], errors: Sequence[float]) -> float | None:
    """Log-log slope of ``error`` against ``panels`` (observed order, if positive)."""
    x = np.log(np.asarray(panels, dtype=float))
    y = np.log(np.maximum(np.asarray(errors, dtype=float), 1.0e-300))
    if x.size < 2 or not np.all(np.isfinite(y)) or np.ptp(x) == 0.0:
        return None
    return float(np.polyfit(x - x.mean(), y, 1)[0])


def independent_mesh_ladder(
    counts: Sequence[int] = DEFAULT_UV_COUNTS,
    accepted_counts: Sequence[int] = EXTERNAL_ACCEPTED_COUNTS,
) -> dict[str, Any]:
    """UV-sphere ellipsoid, trimesh integration, vs the closed-form block form.

    Neither layer is TensorBEM's: the sphere is parametrised in
    longitude/latitude (two pole fans, ``2n(2n-1)+2`` panels) and the second
    moments come from trimesh.  The closed form is the same reference the
    anchor uses, so this is a *new discretisation of the same claim*.
    """
    trimesh = _trimesh()
    axes = np.asarray(base_case.ELLIPSOID_SEMI_AXES, dtype=float)
    centre = np.asarray(base_case.ELLIPSOID_CENTRE, dtype=float)
    density = float(base_case.ELLIPSOID_DENSITY)
    ref_props = base_case.ellipsoid_reference()
    ref_mass = float(ref_props["mass"])
    ref_cg = np.asarray(ref_props["center_of_gravity"], dtype=float)
    ref_inertia = np.asarray(ref_props["inertia_about_cg"], dtype=float)
    ref_matrix = _matrix(ref_mass, ref_cg, ref_inertia)

    levels = []
    for count in counts:
        sphere = trimesh.creation.uv_sphere(radius=1.0, count=(int(count), 2 * int(count)))
        verts = np.asarray(sphere.vertices, dtype=float) * axes + centre
        faces = np.asarray(sphere.faces, dtype=int)
        props = external_mass_properties(verts, faces, density)
        matrix = _matrix(
            props["mass"],
            np.asarray(props["center_of_gravity"], dtype=float),
            np.asarray(props["inertia_about_cg"], dtype=float),
        )
        error = _max_entry_error(matrix, ref_matrix)
        control = _max_entry_error(
            base_case.legacy_coupling_matrix(
                props["mass"],
                np.asarray(props["center_of_gravity"], dtype=float),
                np.asarray(props["inertia_about_cg"], dtype=float),
            ),
            ref_matrix,
        )
        levels.append(
            {
                "uv_count": int(count),
                "panels": int(props["panels"]),
                "watertight": bool(props["watertight"]),
                "accepted": bool(int(count) in set(int(c) for c in accepted_counts)),
                "volume_rel_error": abs(props["volume"] - ref_props["volume"])
                / ref_props["volume"],
                "cg_abs_error_m": float(
                    np.max(np.abs(np.asarray(props["center_of_gravity"]) - ref_cg))
                ),
                "inertia_rel_error": float(
                    np.max(np.abs(np.asarray(props["inertia_about_cg"]) - ref_inertia))
                    / np.max(np.abs(ref_inertia))
                ),
                "max_entry_rel_error": error["max_rel_error"],
                "worst_entry": error["worst_entry"],
                "legacy_coupling_rel_error": control["max_rel_error"],
                "within_tolerance": bool(error["max_rel_error"] < TOLERANCE),
            }
        )
    accepted = [row for row in levels if row["accepted"]]
    if not accepted:  # pragma: no cover - guarded by the defaults
        raise ValueError("accepted_counts must select at least one level")
    worst_accepted = max(row["max_entry_rel_error"] for row in accepted)
    worst_control = max(row["legacy_coupling_rel_error"] for row in levels)
    return {
        "method": "trimesh.creation.uv_sphere scaled by the semi-axes + trimesh mass_properties; "
        "assembly = shipped rigid_body_mass_matrix (inputs are external)",
        "tolerance": TOLERANCE,
        "accepted_counts": [int(c) for c in accepted_counts],
        "levels": levels,
        "worst_accepted_entry_rel_error": float(worst_accepted),
        "worst_accepted_level": min(
            (row for row in accepted if row["max_entry_rel_error"] == worst_accepted),
            key=lambda row: row["uv_count"],
        )["uv_count"],
        "coarsest_within_tolerance_count": min(
            (row["uv_count"] for row in levels if row["within_tolerance"]), default=None
        ),
        "declared_acceptance_note": (
            "the acceptance set is predeclared at uv >= %d; the ladder also reports the "
            "coarsest level that already meets the tolerance, so the margin is visible "
            "rather than implied" % min(int(c) for c in accepted_counts)
        ),
        "observed_order": _fit_slope(
            [row["panels"] for row in levels], [row["max_entry_rel_error"] for row in levels]
        ),
        "legacy_coupling_worst_rel_error": float(worst_control),
        "checks": {
            "external_mesh_route_within_tolerance": bool(worst_accepted < TOLERANCE),
            "external_route_rejects_legacy_coupling": bool(worst_control > TOLERANCE),
        },
    }


# ---------------------------------------------------------------------------
# 3. the ship-shaped body the anchor's box/ellipsoid cases do not cover
# ---------------------------------------------------------------------------
def ship_shape_check(n_sections: int = 48, n_levels: int = 24) -> dict[str, Any]:
    """Wigley solid: trimesh integration vs the hull's closed-form integrals.

    The Wigley hull is the anchor's ship-shaped extension; its closed form
    (:func:`...case_rigid_body_mass_wigley.closed_form_mass_properties`) is a
    sum of hand-integrated polynomial moments, i.e. exactly the kind of
    reference an independent integrator can falsify.
    """
    from tensorbem.benchmarks import case_rigid_body_mass_wigley as wigley  # noqa: PLC0415

    hull = wigley.HULL
    ref = wigley.closed_form_mass_properties(hull)
    ref_matrix = _matrix(
        float(ref["mass"]),
        np.asarray(hull.cg, dtype=float),
        np.asarray(ref["inertia_at_design_cg"], dtype=float),
    )

    vertices, faces, meta = wigley.solid_mesh(hull, n_sections=n_sections, n_levels=n_levels)
    props = external_mass_properties(vertices, faces, hull.density)
    inertia_at_cg = wigley.transfer_inertia(
        np.asarray(props["inertia_about_cg"], dtype=float),
        float(props["mass"]),
        np.asarray(props["center_of_gravity"], dtype=float),
        np.asarray(hull.cg, dtype=float),
    )
    matrix = _matrix(props["mass"], np.asarray(hull.cg, dtype=float), inertia_at_cg)
    error = _max_entry_error(matrix, ref_matrix)

    # the bilge/waterline-dominated entries are the interesting ones: report the
    # block-wise worst error as well as the global one.
    blocks = {
        "translation-translation": error["max_rel_error"]
        if error["worst_entry"].split("-")[0] in MODES[:3]
        and error["worst_entry"].split("-")[1] in MODES[:3]
        else float(
            np.max(
                base_case.relative_error_matrix(ref_matrix, matrix)[:3, :3]
            )
        ),
        "translation-rotation": float(
            np.max(base_case.relative_error_matrix(ref_matrix, matrix)[:3, 3:])
        ),
        "rotation-rotation": float(
            np.max(base_case.relative_error_matrix(ref_matrix, matrix)[3:, 3:])
        ),
    }
    return {
        "hull": {
            "length_m": hull.length,
            "beam_m": hull.beam,
            "draft_m": hull.draft,
            "block_coefficient": hull.block_coefficient,
            "density_kg_m3": hull.density,
            "cg_m": [float(v) for v in hull.cg],
        },
        "mesh": {
            "n_sections": int(n_sections),
            "n_levels": int(n_levels),
            "panels": int(meta.get("panels", props["panels"])),
            "wetted_panels": int(meta.get("wetted_panels", 0)),
            "lid_panels": int(meta.get("lid_panels", 0)),
            "external_watertight": bool(props["watertight"]),
        },
        "reference": {
            "volume_m3": float(ref["volume"]),
            "mass_kg": float(ref["mass"]),
            "centroid_m": [float(v) for v in ref["centroid"]],
            "inertia_at_design_cg": np.asarray(ref["inertia_at_design_cg"]).tolist(),
        },
        "external": {
            "volume_m3": float(props["volume"]),
            "mass_kg": float(props["mass"]),
            "centroid_m": [float(v) for v in props["center_of_gravity"]],
        },
        "volume_rel_error": abs(props["volume"] - ref["volume"]) / ref["volume"],
        "mass_rel_error": abs(props["mass"] - ref["mass"]) / ref["mass"],
        "centroid_abs_error_m": float(
            np.max(
                np.abs(
                    np.asarray(props["center_of_gravity"]) - np.asarray(ref["centroid"], dtype=float)
                )
            )
        ),
        "inertia_rel_error": float(
            np.max(
                np.abs(inertia_at_cg - np.asarray(ref["inertia_at_design_cg"], dtype=float))
            )
            / np.max(np.abs(np.asarray(ref["inertia_at_design_cg"], dtype=float)))
        ),
        "matrix_max_entry_rel_error": error["max_rel_error"],
        "worst_entry": error["worst_entry"],
        "block_max_entry_rel_error": blocks,
        "tolerance": TOLERANCE,
        "checks": {
            "external_ship_shape_within_tolerance": bool(error["max_rel_error"] < TOLERANCE),
        },
    }


# ---------------------------------------------------------------------------
# 4. external referee for the topology guard
# ---------------------------------------------------------------------------
def topology_referee(subdivisions: int = 2) -> dict[str, Any]:
    """Corrupted meshes: TensorBEM's guard vs trimesh's ``is_watertight``.

    The guard was the remediation of the anchor's adversarial review.  A
    remediation certified only by its own tests is self-referential, so an
    external implementation is asked for its verdict on the same triangles.

    The comparison is deliberately *not* an equality test.  What is required is
    the safety direction -- the guard may never accept a surface that the
    external referee rejects -- and, separately, the empirical statement that
    the external predicate is **not** a substitute: trimesh decides watertightness
    on edge incidence, so a single flipped triangle keeps it "watertight" while
    the signed volume it integrates is wrong.  Every corruption is therefore
    also pushed through the external integrator as a *naive external workflow*
    would do (``require_watertight=False``) and scored against the closed form:
    the ones whose 6x6 error stays inside the 3 % tolerance are exactly the ones
    a tolerance test can never see.
    """
    from tensorbem.geometry3d import ellipsoid_mesh  # noqa: PLC0415

    axes = np.asarray(base_case.ELLIPSOID_SEMI_AXES, dtype=float)
    centre = np.asarray(base_case.ELLIPSOID_CENTRE, dtype=float)
    density = float(base_case.ELLIPSOID_DENSITY)
    ref_props = base_case.ellipsoid_reference()
    ref_matrix = _matrix(
        float(ref_props["mass"]),
        np.asarray(ref_props["center_of_gravity"], dtype=float),
        np.asarray(ref_props["inertia_about_cg"], dtype=float),
    )

    base = ellipsoid_mesh(int(subdivisions), axes)
    verts = np.asarray(base.vertices, dtype=float) + centre
    faces = np.asarray(base.faces, dtype=int)
    n_faces = int(faces.shape[0])
    variants: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "closed (control)": (verts, faces),
        "open patch (every 7th face removed)": (
            verts,
            np.delete(faces, np.arange(0, n_faces, 7), axis=0),
        ),
        "single flipped face": (verts, np.vstack([faces[:-1], faces[-1][::-1]])),
        "duplicated face": (verts, np.vstack([faces, faces[:1]])),
        "inward winding (all faces)": (verts, faces[:, ::-1]),
    }

    rows = []
    for name, (v, f) in variants.items():
        report = mesh_topology_report(v, f)
        guard_closed = bool(report["is_closed"])
        guard_positive = bool(base_case.tetra_mass_properties(v, f, density)["volume"] > 0.0)
        ext = external_mass_properties(v, f, density, require_watertight=False)
        external_closed = bool(ext["watertight"])
        external_positive = bool(ext["winding_positive"])
        # what a naive external workflow would certify with the corrupted mesh.
        # A negative enclosed volume makes the assembly undefined (mass <= 0), so
        # the naive workflow fails loudly there -- noted, not scored.
        if float(ext["mass"]) > 0.0:
            naive_matrix = _matrix(
                ext["mass"],
                np.asarray(ext["center_of_gravity"], dtype=float),
                np.asarray(ext["inertia_about_cg"], dtype=float),
            )
            agreement = _max_entry_error(naive_matrix, ref_matrix)
            naive_error = agreement["max_rel_error"]
            naive_worst = agreement["worst_entry"]
            naive_undefined = False
        else:
            naive_error = float("inf")
            naive_worst = None
            naive_undefined = True
        rows.append(
            {
                "variant": name,
                "guard_is_closed": guard_closed,
                "guard_volume_positive": guard_positive,
                "guard_open_edges": int(report["open_edges"]),
                "guard_orientation_mismatches": int(report["orientation_mismatches"]),
                "guard_nonmanifold_edges": int(report["nonmanifold_edges"]),
                "external_watertight": external_closed,
                "external_winding_positive": external_positive,
                "guard_accepts": bool(guard_closed and guard_positive),
                "external_accepts": bool(external_closed and external_positive),
                "guard_laxer_than_external": bool(
                    guard_closed and guard_positive and not (external_closed and external_positive)
                ),
                "external_laxer_than_guard": bool(
                    external_closed and external_positive and not (guard_closed and guard_positive)
                ),
                # a 3 % tolerance test against the closed form, on the corrupted mesh
                "volume_rel_error_vs_closed_form": abs(ext["volume"] - ref_props["volume"])
                / ref_props["volume"],
                "naive_workflow_undefined": bool(naive_undefined),
                "naive_worst_entry": naive_worst,
                "matrix_max_entry_rel_error": naive_error,
                "maskable_by_tolerance": bool(naive_error < TOLERANCE),
            }
        )

    corruptions = rows[1:]
    # --- welding policy probe: fully split vertices on the *closed* surface
    split_v = verts[faces.reshape(-1)]
    split_f = np.arange(split_v.shape[0], dtype=int).reshape(-1, 3)
    split_report = mesh_topology_report(split_v, split_f)
    trimesh = _trimesh()
    merged = trimesh.Trimesh(vertices=split_v, faces=split_f, process=False)
    merged.merge_vertices()
    weld_sensitivity = {
        "variant": "fully split vertices (same closed surface, unwelded indices)",
        "guard_is_closed": bool(split_report["is_closed"]),
        "guard_volume_m3": float(base_case.tetra_mass_properties(split_v, split_f, density)["volume"]),
        "external_watertight_as_given": bool(
            external_mass_properties(split_v, split_f, density, require_watertight=False)[
                "watertight"
            ]
        ),
        "external_watertight_after_merge": bool(merged.is_watertight),
        "policies_agree_after_alignment": bool(split_report["is_closed"] and merged.is_watertight),
    }

    guard_not_laxer = not any(row["guard_laxer_than_external"] for row in rows)
    guard_rejects_all = all(not row["guard_accepts"] for row in corruptions)
    external_misses = [row["variant"] for row in corruptions if row["external_accepts"]]
    missed_and_maskable = [
        row["variant"]
        for row in corruptions
        if row["external_accepts"] and row["maskable_by_tolerance"]
    ]
    note = (
        "The external predicate is *weaker*, not merely different: it decides "
        "watertightness on edge incidence, so a single flipped triangle keeps it "
        "'watertight' while the enclosed volume it integrates is wrong. In this "
        "configuration the corrupted 6x6 is nevertheless far outside the 3 % "
        "tolerance (the entry-relative metric's floor on structurally-zero "
        "entries amplifies it), so the tolerance test did see the defect here -- "
        "but the external predicate did not, which is why the guard cannot be "
        "outsourced to trimesh."
    )
    return {
        "method": "tensorbem.rigid_body_mass.mesh_topology_report vs trimesh.Trimesh.is_watertight "
        "(+ winding sign) on the same triangles; every corruption is also integrated "
        "through the external route as a naive external workflow would",
        "subdivisions": int(subdivisions),
        "variants": rows,
        "weld_sensitivity": weld_sensitivity,
        "guard_never_laxer": bool(guard_not_laxer),
        "guard_rejects_every_corruption": bool(guard_rejects_all),
        "corruptions_the_external_predicate_misses": external_misses,
        "missed_and_inside_the_tolerance": missed_and_maskable,
        "note": note,
        "checks": {
            "external_topology_guard_agrees": bool(
                guard_not_laxer and guard_rejects_all
            ),
            "external_guard_is_load_bearing": bool(external_misses and guard_rejects_all),
        },
    }


# ---------------------------------------------------------------------------
# section assembly
# ---------------------------------------------------------------------------
def not_run(reason: str) -> dict[str, Any]:
    """Payload for the unrun section -- a skipped check is never a pass."""
    return {
        "status": "not_run",
        "reason": reason,
        "checks": {},
        "required_checks": list(CHECKS),
    }


def run(
    subdivisions: Sequence[int] = DEFAULT_SUBDIVISIONS,
    counts: Sequence[int] = DEFAULT_UV_COUNTS,
    accepted_counts: Sequence[int] = EXTERNAL_ACCEPTED_COUNTS,
) -> dict[str, Any]:
    """Run route T end to end and return the machine-checkable section."""
    if not available():
        return not_run(
            "trimesh is not installed: route T was skipped, not passed "
            "(pip install 'tensorbem[external-verification]')"
        )

    calibration = self_calibration()
    attribution = integrator_cross_check(subdivisions)
    ladder = independent_mesh_ladder(counts, accepted_counts)
    ship = ship_shape_check()
    topology = topology_referee()

    checks: dict[str, bool] = {}
    for part in (attribution, ladder, ship, topology):
        checks.update(part["checks"])
    missing = [name for name in CHECKS if name not in checks]

    return {
        "status": "run" if not missing else "not_run",
        "reason": "route T executed" if not missing else f"checks missing: {missing}",
        "route": "external referee: trimesh (Mirtich 1996) integration, uv_sphere discretisation",
        "independence": {
            "mesh_generator": "trimesh.creation.uv_sphere (not tensorbem.geometry3d.ellipsoid_mesh)",
            "integration": "trimesh mass_properties (not tensorbem signed-tetrahedron quadrature)",
            "shared_with_the_anchor": [
                "the closed-form reference (the object under test)",
                "the 6x6 assembly rigid_body_mass_matrix (route P pins it separately)",
                "the entry-relative error metric (the anchor's acceptance metric)",
            ],
        },
        "tolerance": TOLERANCE,
        "roundoff_limit": ROUNDOFF_LIMIT,
        "self_calibration": calibration,
        "layer_attribution": attribution,
        "independent_mesh_route": ladder,
        "ship_shape": ship,
        "topology_referee": topology,
        "checks": checks,
        "required_checks": list(CHECKS),
        "residual_limitations": [
            "trimesh and TensorBEM both integrate a *triangulated* surface, so a systematic "
            "geometric error common to all triangle meshes (a mis-scaled semi-axis, say) is "
            "outside every route in this anchor; Cb/L/B geometry verification is the job of "
            "the geometry anchors",
            "the external route assembles M_RB with the shipped assembly, so it certifies the "
            "mass-property inputs, not the block convention (route P owns that)",
            "trimesh's is_watertight is a topological predicate only; it does not check "
            "self-intersection",
        ],
    }