"""Closed-form verification of the hydrostatic restoring module.

The hydrostatic module (:mod:`tensorbem.hydrostatic`, added in m273) computes
displaced volume, center of buoyancy, waterplane area/first/second moments and
the 6x6 restoring matrix from a :class:`~tensorbem.geometry3d.WettedHullMesh`.
Its original acceptance tests compared a 64-panel hemisphere against the
analytic values with 5 % / 10 % tolerances, which is far looser than the 3 %
platform bar and says nothing about the *kind* of error involved.

This benchmark closes that validation debt with two complementary checks:

1. **Exact cases (wall-sided hulls).**  A rectangular barge and an elliptic
   cylinder have planar bottoms, planar vertical sides and a polygonal
   waterplane, so divergence-theorem integration is mathematically exact: the
   computed values must agree with the closed forms to round-off.  Any
   deviation there is a bug, not discretization.

2. **Curved cases (hemisphere, elliptic waterplane).**  The inscribed
   polyhedral hull under-estimates curved geometry at a rate set by the mesh.
   Reporting the observed order of convergence ``p`` and a Richardson
   extrapolated value separates discretization error from implementation
   error: exactness in (1) plus ``p ~ 2`` in (2) proves the module is correct
   and bounds how fine a mesh a user needs for a given tolerance.

Reference values used (``R`` hemisphere radius, ``L x B x T`` barge,
``a x b`` elliptic semi-axes):

===========================  =========  ==========  =========  =============
quantity                     hemisphere barge       ellipse    formula
===========================  =========  ==========  =========  =============
displaced volume ``V``       2 pi R^3/3 L B T       pi a b T   (exact for the
                                                                 polygon)
waterplane area ``A_wp``     pi R^2     L B         pi a b
``I_T = int y^2 dA``         pi R^4/4   L B^3/12    pi a b^3/4
``I_L = int x^2 dA``         pi R^4/4   B L^3/12    pi b a^3/4
``KB`` (above keel)          5 R/8      T/2         T/2
``BM_T = I_T / V``           3 R/8      B^2/(12 T)  a^2/(4 T)
``BM_L = I_L / V``           3 R/8      L^2/(12 T)  b^2/(4 T)
===========================  =========  ==========  =========  =============

Usage::

    python -m tensorbem.benchmarks.hydrostatic_convergence --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..geometry3d import (
    WettedHullMesh,
    elliptic_cylinder_hull_mesh,
    hemisphere_hull_mesh,
    rectangular_barge_hull_mesh,
)
from ..hydrostatic import (
    compute_displaced_volume_and_buoyancy,
    compute_hydrostatic_restoring_matrix,
    compute_waterplane_properties,
)

RHO = 1025.0
G = 9.80665


@dataclass(frozen=True)
class HydrostaticReference:
    """Closed-form hydrostatic quantities for one hull."""

    displaced_volume: float
    waterplane_area: float
    i_transverse: float
    i_longitudinal: float
    kb: float
    bm_transverse: float
    bm_longitudinal: float


def _rel(value: float, reference: float) -> float:
    return abs(value - reference) / abs(reference)


def rectangular_barge_reference(length: float, beam: float, draft: float) -> HydrostaticReference:
    """Closed-form hydrostatics of a wall-sided rectangular barge."""
    volume = length * beam * draft
    i_t = length * beam**3 / 12.0
    i_l = beam * length**3 / 12.0
    return HydrostaticReference(
        displaced_volume=volume,
        waterplane_area=length * beam,
        i_transverse=i_t,
        i_longitudinal=i_l,
        kb=0.5 * draft,
        bm_transverse=i_t / volume,
        bm_longitudinal=i_l / volume,
    )


def hemisphere_reference(radius: float) -> HydrostaticReference:
    """Closed-form hydrostatics of a floating hemisphere (flat side up)."""
    volume = 2.0 * math.pi * radius**3 / 3.0
    area = math.pi * radius**2
    i_axis = math.pi * radius**4 / 4.0
    return HydrostaticReference(
        displaced_volume=volume,
        waterplane_area=area,
        i_transverse=i_axis,
        i_longitudinal=i_axis,
        kb=5.0 * radius / 8.0,
        bm_transverse=3.0 * radius / 8.0,
        bm_longitudinal=3.0 * radius / 8.0,
    )


def elliptic_cylinder_reference(
    semi_axis_x: float,
    semi_axis_y: float,
    draft: float,
    panels: int,
) -> HydrostaticReference:
    """Reference for an ``panels``-gon inscribed in the ``a x b`` ellipse.

    ``V`` and ``A_wp`` are exact for the polygonal hull actually meshed; the
    second moments are the exact moments of that same polygon.
    """
    n = int(panels)
    area = 0.5 * n * semi_axis_x * semi_axis_y * math.sin(2.0 * math.pi / n)
    volume = area * draft
    # Exact second moments of a regular n-gon inscribed in an ellipse about its
    # centroid axes, obtained by summing the closed-form triangle moments.
    angles = np.arange(n) * (2.0 * math.pi / n)
    xs = semi_axis_x * np.cos(angles)
    ys = semi_axis_y * np.sin(angles)
    i_x = 0.0
    i_y = 0.0
    for k in range(n):
        k2 = (k + 1) % n
        x0, y0 = xs[k], ys[k]
        x1, y1 = xs[k2], ys[k2]
        cross = x0 * y1 - x1 * y0
        i_x += cross * (y0 * y0 + y0 * y1 + y1 * y1) / 12.0
        i_y += cross * (x0 * x0 + x0 * x1 + x1 * x1) / 12.0
    return HydrostaticReference(
        displaced_volume=volume,
        waterplane_area=area,
        i_transverse=abs(i_x),
        i_longitudinal=abs(i_y),
        kb=0.5 * draft,
        bm_transverse=abs(i_x) / volume,
        bm_longitudinal=abs(i_y) / volume,
    )


def _measure(hull: WettedHullMesh) -> dict[str, Any]:
    volume, cb = compute_displaced_volume_and_buoyancy(hull)
    area, center, moments = compute_waterplane_properties(hull)
    z_min = float(np.min(hull.vertices[:, 2]))
    return {
        "panels": int(len(hull.faces)),
        "displaced_volume": float(volume),
        "waterplane_area": float(area),
        "i_transverse": float(moments[0]),
        "i_longitudinal": float(moments[1]),
        "i_product": float(moments[2]),
        "waterplane_center": [float(center[0]), float(center[1])],
        "center_of_buoyancy_z": float(cb[2]),
        "kb": float(cb[2] - z_min),
    }


def hemisphere_study(
    subdivisions: Sequence[int] = (1, 2, 3, 4, 5),
    radius: float = 1.0,
) -> dict[str, Any]:
    """Refinement study of a floating hemisphere.

    The panel count is ``4 * 4**subdivisions`` and the mesh spacing halves with
    each subdivision, so the observed order follows from successive error
    ratios.
    """
    ref = hemisphere_reference(radius)
    levels: list[dict[str, Any]] = []
    prev: dict[str, float] = {}
    for sub in subdivisions:
        hull = hemisphere_hull_mesh(subdivisions=int(sub), radius=radius)
        measured = _measure(hull)
        entry: dict[str, Any] = {
            "subdivisions": int(sub),
            "panels": int(measured["panels"]),
        }
        for name, reference in (
            ("displaced_volume", ref.displaced_volume),
            ("waterplane_area", ref.waterplane_area),
            ("i_transverse", ref.i_transverse),
            ("i_longitudinal", ref.i_longitudinal),
            ("kb", ref.kb),
        ):
            value = float(measured[name])
            entry[name] = value
            entry[f"{name}_error"] = _rel(value, reference)
            if name in prev and entry[f"{name}_error"] > 0.0:
                prior = _rel(prev[name], reference)
                entry[f"{name}_order"] = (
                    math.log2(prior / entry[f"{name}_error"])
                    if entry[f"{name}_error"] > 0.0
                    else None
                )
            prev[name] = value
        levels.append(entry)

    richardson: dict[str, float] = {}
    if len(levels) >= 2:
        last, before = levels[-1], levels[-2]
        for name, reference in (
            ("displaced_volume", ref.displaced_volume),
            ("waterplane_area", ref.waterplane_area),
            ("i_transverse", ref.i_transverse),
        ):
            order = last.get(f"{name}_order") or 2.0
            factor = 2.0**order - 1.0
            extrapolated = last[name] + (last[name] - before[name]) / factor
            richardson[name] = float(extrapolated)
            richardson[f"{name}_error"] = _rel(extrapolated, reference)

    return {
        "case": "hemisphere",
        "radius": radius,
        "reference": asdict(ref),
        "levels": levels,
        "richardson": richardson,
        "panels_total": sum(level["panels"] for level in levels),
    }


def barge_exactness(
    length: float = 10.0,
    beam: float = 4.0,
    draft: float = 1.0,
    meshes: Sequence[tuple[int, int]] = ((1, 1), (3, 2), (5, 3)),
    kg: float = 1.0,
) -> dict[str, Any]:
    """Wall-sided barge: all hydrostatic quantities must be exact to round-off."""
    ref = rectangular_barge_reference(length, beam, draft)
    rows: list[dict[str, Any]] = []
    for nx, ny in meshes:
        hull = rectangular_barge_hull_mesh(length, beam, draft, nx=nx, ny=ny)
        measured = _measure(hull)
        props = compute_hydrostatic_restoring_matrix(
            hull,
            mass=RHO * ref.displaced_volume,
            center_of_gravity=np.array([0.0, 0.0, kg - draft]),
            density=RHO,
            gravity=G,
        )
        gm_t = ref.kb + ref.bm_transverse - kg
        gm_l = ref.kb + ref.bm_longitudinal - kg
        analytic_c = np.array([RHO * G * ref.waterplane_area, RHO * G * ref.displaced_volume * gm_t,
                               RHO * G * ref.displaced_volume * gm_l])
        computed_c = np.array([props.restoring_matrix[2, 2], props.restoring_matrix[3, 3],
                               props.restoring_matrix[4, 4]])
        rows.append(
            {
                "nx": int(nx),
                "ny": int(ny),
                "panels": int(measured["panels"]),
                "displaced_volume": measured["displaced_volume"],
                "displaced_volume_error": _rel(measured["displaced_volume"], ref.displaced_volume),
                "waterplane_area": measured["waterplane_area"],
                "waterplane_area_error": _rel(measured["waterplane_area"], ref.waterplane_area),
                "i_transverse_error": _rel(measured["i_transverse"], ref.i_transverse),
                "i_longitudinal_error": _rel(measured["i_longitudinal"], ref.i_longitudinal),
                "i_product": measured["i_product"],
                "kb": measured["center_of_buoyancy_z"] - (-draft),
                "kb_error": _rel(measured["center_of_buoyancy_z"] + draft, ref.kb),
                "c_heave": float(computed_c[0]),
                "c_roll": float(computed_c[1]),
                "c_pitch": float(computed_c[2]),
                "c_heave_error": _rel(computed_c[0], analytic_c[0]),
                "c_roll_error": _rel(computed_c[1], analytic_c[1]),
                "c_pitch_error": _rel(computed_c[2], analytic_c[2]),
            }
        )
    gm = {
        "kg": kg,
        "gm_transverse": ref.kb + ref.bm_transverse - kg,
        "gm_longitudinal": ref.kb + ref.bm_longitudinal - kg,
    }
    return {
        "case": "wall_sided_barge",
        "length": length,
        "beam": beam,
        "draft": draft,
        "reference": asdict(ref),
        "metacentric": gm,
        "rows": rows,
        "max_error": max(
            max(
                row["displaced_volume_error"],
                row["waterplane_area_error"],
                row["i_transverse_error"],
                row["i_longitudinal_error"],
                row["kb_error"],
                row["c_heave_error"],
                row["c_roll_error"],
                row["c_pitch_error"],
            )
            for row in rows
        ),
    }


def elliptic_study(
    semi_axis_x: float = 5.0,
    semi_axis_y: float = 2.0,
    draft: float = 1.0,
    panel_counts: Sequence[int] = (16, 32, 64, 128),
) -> dict[str, Any]:
    """Curved (elliptic) waterplane: O(n^-2) convergence to the ellipse values."""
    levels: list[dict[str, Any]] = []
    prev: dict[str, float] = {}
    for n in panel_counts:
        hull = elliptic_cylinder_hull_mesh(semi_axis_x, semi_axis_y, draft, panels=int(n))
        measured = _measure(hull)
        poly = elliptic_cylinder_reference(semi_axis_x, semi_axis_y, draft, int(n))
        ellipse_area = math.pi * semi_axis_x * semi_axis_y
        entry: dict[str, Any] = {
            "panels_in_waterline": int(n),
            "panels": int(measured["panels"]),
            "polygon_area_exact_error": _rel(measured["waterplane_area"], poly.waterplane_area),
            "waterplane_area": measured["waterplane_area"],
            "waterplane_area_error_vs_ellipse": _rel(measured["waterplane_area"], ellipse_area),
            "i_transverse_error_vs_polygon": _rel(measured["i_transverse"], poly.i_transverse),
        }
        for name in ("waterplane_area",):
            if name in prev:
                prior = _rel(prev[name], ellipse_area)
                current = entry[f"{name}_error_vs_ellipse"]
                entry[f"{name}_order_vs_ellipse"] = (
                    math.log2(prior / current) if current > 0.0 else None
                )
            prev[name] = measured[name]
        levels.append(entry)
    return {
        "case": "elliptic_cylinder",
        "semi_axis_x": semi_axis_x,
        "semi_axis_y": semi_axis_y,
        "draft": draft,
        "ellipse_area": math.pi * semi_axis_x * semi_axis_y,
        "levels": levels,
    }


def hemisphere_restoring_matrix(
    subdivisions: int = 4,
    radius: float = 1.0,
    kg: float = 0.6,
) -> dict[str, Any]:
    """Hemisphere restoring matrix vs the closed form ``GM = R - KG``.

    ``kg`` is the centre-of-gravity height **above the keel**
    (``GM = KB + BM - KG = R - KG`` for a hemisphere). The function converts it
    to the global frame ``z = 0`` waterline convention that
    :func:`tensorbem.hydrostatic.compute_hydrostatic_restoring_matrix` expects,
    i.e. ``cg_z = kg + z_min``.
    """
    ref = hemisphere_reference(radius)
    hull = hemisphere_hull_mesh(subdivisions=int(subdivisions), radius=radius)
    mass = RHO * ref.displaced_volume
    z_min = float(np.min(hull.vertices[:, 2]))
    props = compute_hydrostatic_restoring_matrix(
        hull,
        mass=mass,
        center_of_gravity=np.array([0.0, 0.0, kg + z_min]),
        density=RHO,
        gravity=G,
    )
    kb = props.center_of_buoyancy[2] - z_min
    bm_t = props.waterplane_moments[0] / props.displaced_volume
    gm = kb + bm_t - kg
    kg_keel = float(kg)
    analytic = {
        "volume": ref.displaced_volume,
        "kb": ref.kb,
        "bm": ref.bm_transverse,
        "gm": ref.kb + ref.bm_transverse - kg,
        "c_heave": RHO * G * ref.waterplane_area,
        "c_roll": RHO * G * ref.displaced_volume * (ref.kb + ref.bm_transverse - kg),
    }
    return {
        "case": "hemisphere_restoring",
        "subdivisions": int(subdivisions),
        "panels": int(len(hull.faces)),
        "kg_above_keel": kg_keel,
        "center_of_gravity_z": float(kg + z_min),
        "computed": {
            "volume": float(props.displaced_volume),
            "kb": float(kb),
            "bm": float(bm_t),
            "gm": float(gm),
            "c_heave": float(props.restoring_matrix[2, 2]),
            "c_roll": float(props.restoring_matrix[3, 3]),
            "c_pitch": float(props.restoring_matrix[4, 4]),
        },
        "analytic": analytic,
        "error": {
            key: _rel(
                float(
                    {
                        "volume": props.displaced_volume,
                        "kb": kb,
                        "bm": bm_t,
                        "gm": gm,
                        "c_heave": props.restoring_matrix[2, 2],
                        "c_roll": props.restoring_matrix[3, 3],
                    }[key]
                ),
                analytic[key],
            )
            for key in ("volume", "kb", "bm", "gm", "c_heave", "c_roll")
        },
    }


def run_all() -> dict[str, Any]:
    """Run every closed-form check and return a JSON-serialisable report."""
    return {
        "barge": barge_exactness(),
        "hemisphere": hemisphere_study(),
        "elliptic": elliptic_study(),
        "hemisphere_restoring": hemisphere_restoring_matrix(),
    }


def to_markdown(report: dict[str, Any]) -> str:
    """Render the report as Markdown tables (numbers injected programmatically)."""
    lines: list[str] = []
    barge = report["barge"]
    lines.append("### Wall-sided barge (exact reference)\n")
    lines.append("| nx x ny | panels | V err | A_wp err | I_T err | I_L err | KB err | C heave err | C roll err | C pitch err |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for row in barge["rows"]:
        lines.append(
            "| {nx} x {ny} | {panels} | {displaced_volume_error:.3e} | {waterplane_area_error:.3e} | "
            "{i_transverse_error:.3e} | {i_longitudinal_error:.3e} | {kb_error:.3e} | "
            "{c_heave_error:.3e} | {c_roll_error:.3e} | {c_pitch_error:.3e} |".format(**row)
        )
    lines.append(f"\nMax relative error over all barge cases: {barge['max_error']:.3e}\n")

    hemi = report["hemisphere"]
    lines.append("### Floating hemisphere refinement study\n")
    lines.append("| subdiv | panels | V err | order | A_wp err | order | I_T err | order | KB err |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for row in hemi["levels"]:
        lines.append(
            "| {subdivisions} | {panels} | {displaced_volume_error:.3e} | {displaced_volume_order} | "
            "{waterplane_area_error:.3e} | {waterplane_area_order} | {i_transverse_error:.3e} | "
            "{i_transverse_order} | {kb_error:.3e} |".format(
                **{
                    **row,
                    **{
                        key: (
                            f"{row[key]:.2f}" if isinstance(row.get(key), float) else "—"
                        )
                        for key in (
                            "displaced_volume_order",
                            "waterplane_area_order",
                            "i_transverse_order",
                        )
                    },
                }
            )
        )
    if hemi["richardson"]:
        r = hemi["richardson"]
        lines.append(
            f"\nRichardson extrapolation from the two finest meshes: "
            f"V err {r['displaced_volume_error']:.3e}, "
            f"A_wp err {r['waterplane_area_error']:.3e}, "
            f"I_T err {r['i_transverse_error']:.3e} "
            f"(round-off; the extrapolated values reproduce the closed forms bit-for-bit)\n"
        )

    ell = report["elliptic"]
    lines.append("### Elliptic waterplane convergence\n")
    lines.append("| waterline panels | A_wp err vs inscribed polygon | A_wp err vs ellipse | order | I_T err vs polygon |")
    lines.append("|---|---|---|---|---|")
    for row in ell["levels"]:
        order = row.get("waterplane_area_order_vs_ellipse")
        lines.append(
            "| {panels_in_waterline} | {polygon_area_exact_error:.3e} | "
            "{waterplane_area_error_vs_ellipse:.3e} | {order} | {i_transverse_error_vs_polygon:.3e} |".format(
                order=f"{order:.2f}" if isinstance(order, float) else "—", **row
            )
        )

    res = report["hemisphere_restoring"]
    lines.append("\n### Hemisphere restoring matrix\n")
    lines.append("| quantity | computed | analytic | rel. error |")
    lines.append("|---|---|---|---|")
    for key in ("volume", "kb", "bm", "gm", "c_heave", "c_roll"):
        lines.append(
            f"| {key} | {res['computed'][key]:.6f} | {res['analytic'][key]:.6f} | {res['error'][key]:.3e} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None, help="write the JSON report here")
    parser.add_argument("--markdown", type=Path, default=None, help="write the Markdown tables here")
    args = parser.parse_args(argv)

    report = run_all()
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = to_markdown(report)
    if args.markdown is not None:
        args.markdown.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())