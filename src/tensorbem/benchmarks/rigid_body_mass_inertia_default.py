"""Audit: what the rigid-body inertia default does when no tensor is supplied.

The rigid-body mass matrix is the anchor's subject (``CASE-BEM-RIGID-BODY-MASS``).
Its block form is verified elsewhere; this benchmark audits the *default* that
:func:`tensorbem.hydrostatic.compute_mass_matrix` applied when ``inertia`` was
omitted -- ``I_cg = m*(0.4*|r_g|)^2 * I3`` -- against two independent things:

1. the invariance law every rigid-body mass matrix must obey (a body property
   cannot depend on where the coordinate origin sits), and
2. the closed-form inertia of the shape the hull actually is.

It then measures the geometry-anchored replacement
(:func:`tensorbem.hydrostatic.mass_matrix_from_hull_geometry` ->
:func:`tensorbem.rigid_body_mass.equivalent_inertia_from_hydrostatics`).

Run as ``python -m tensorbem.benchmarks.rigid_body_mass_inertia_default``; writes
``results.json`` and ``figures.png`` next to ``--outdir``.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ..geometry3d import (
    WettedHullMesh,
    rectangular_barge_hull_mesh,
    spheroid_hull_mesh,
)
from ..hydrostatic import (
    compute_hydrostatic_restoring_matrix,
    compute_mass_matrix,
    mass_matrix_from_hull_geometry,
)
from ..rigid_body_mass import (
    equivalent_half_ellipsoid,
    equivalent_inertia_from_hydrostatics,
)

DENSITY = 1025.0
MODES = ("surge", "sway", "heave", "roll", "pitch", "yaw")


def _rotational(tensor: np.ndarray) -> np.ndarray:
    return np.array([tensor[0, 0], tensor[1, 1], tensor[2, 2]])


def half_ellipsoid_inertia_cg(semi_axes, mass: float) -> np.ndarray:
    """Closed-form CG inertia of the lower half of a solid ellipsoid.

    About the waterplane-section centre the half body has the solid-ellipsoid
    second moments ``m(b^2+c^2)/5`` etc. (Lamb 1932 Ch. VI; Fossen 2011 App. B);
    its centroid is ``3c/8`` below that section, so the CG-referred roll/pitch
    entries lose ``m*(3c/8)^2`` and yaw is unchanged.
    """
    a, b, c = (float(v) for v in semi_axes)
    lever = 3.0 * c / 8.0
    return np.array(
        [
            mass * (b * b + c * c) / 5.0 - mass * lever ** 2,
            mass * (a * a + c * c) / 5.0 - mass * lever ** 2,
            mass * (a * a + b * b) / 5.0,
        ]
    )


def box_inertia_cg(length: float, beam: float, draft: float, mass: float) -> np.ndarray:
    """Closed-form CG inertia of a rectangular block (roll, pitch, yaw)."""
    b, t, l = float(beam), float(draft), float(length)
    return mass * np.array([(b * b + t * t) / 12.0, (l * l + t * t) / 12.0, (l * l + b * b) / 12.0])


def legacy_default_inertia(mass: float, center_of_gravity) -> np.ndarray:
    """The pre-audit fallback: ``I_cg = m*(0.4*|r_g|)^2 I3``, origin-dependent."""
    r = float(np.linalg.norm(np.asarray(center_of_gravity, dtype=float)))
    k = 0.4 * r if r > 1e-10 else 0.4
    return mass * k * k * np.eye(3)


def _rel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=float) / np.asarray(b, dtype=float) - 1.0


def origin_dependence_ladder(mass: float = 1.0e6) -> list[dict]:
    """Same body, origin moved: the legacy fallback changes, physics does not."""
    rows = []
    for offset in (0.0, 5.0, 45.0, 200.0, 1000.0):
        cg = np.array([offset, 0.0, -3.0])
        legacy = legacy_default_inertia(mass, cg)
        rows.append(
            {
                "cg_x": offset,
                "k_legacy": 0.4 * float(np.linalg.norm(cg)),
                "legacy_roll_inertia": float(legacy[0, 0]),
                "rel_to_first": float(legacy[0, 0] / legacy_default_inertia(mass, np.array([0.0, 0.0, -3.0]))[0, 0] - 1.0),
            }
        )
    ratio = max(r["legacy_roll_inertia"] for r in rows) / min(
        r["legacy_roll_inertia"] for r in rows
    )
    return rows, ratio


def spheroid_ladder(
    semi_axes: tuple[float, float, float] = (60.0, 12.0, 8.0),
    subdivisions=(2, 3, 4),
    mass: float = 1.0e6,
) -> dict:
    """Derived vs closed-form inertia on the hull family the cases use."""
    truth = half_ellipsoid_inertia_cg(semi_axes, mass)
    rows = []
    for sub in subdivisions:
        hull = spheroid_hull_mesh(semi_axes, subdivisions=int(sub))
        props = compute_hydrostatic_restoring_matrix(hull, mass, (0.0, 0.0, -0.5))
        derived = equivalent_inertia_from_hydrostatics(
            props.displaced_volume,
            props.waterplane_area,
            props.waterplane_moments,
            props.waterplane_center,
            mass=mass,
        )
        axes = np.asarray(derived["semi_axes"], dtype=float)
        rows.append(
            {
                "subdivisions": int(sub),
                "panel_count": int(hull.panel_count),
                "semi_axes": axes.tolist(),
                "semi_axes_error": _rel(axes, np.asarray(semi_axes, dtype=float)).tolist(),
                "inertia_cg": np.diag(derived["inertia_cg"]).tolist(),
                "error_vs_closed_form": _rel(np.diag(derived["inertia_cg"]), truth).tolist(),
                "waterplane_area_ratio": float(derived["waterplane_area_ratio"]),
            }
        )
    legacy = np.diag(legacy_default_inertia(mass, (0.0, 0.0, -0.5)))
    return {
        "semi_axes": list(semi_axes),
        "mass": mass,
        "closed_form_inertia_cg": truth.tolist(),
        "ladder": rows,
        "legacy_default_error": _rel(legacy, truth).tolist(),
        "worst_derived_error_at_finest": float(
            np.max(np.abs(rows[-1]["error_vs_closed_form"]))
        ),
        "worst_legacy_error": float(np.max(np.abs(_rel(legacy, truth)))),
    }


def barge_case(
    length: float = 20.0,
    beam: float = 8.0,
    draft: float = 3.0,
    nx: int = 10,
    ny: int = 8,
) -> dict:
    """Declared-approximation domain: a box-like hull has no elliptical waterplane."""
    volume = length * beam * draft
    mass = DENSITY * volume
    hull = rectangular_barge_hull_mesh(length, beam, draft, nx, ny)
    props = compute_hydrostatic_restoring_matrix(hull, mass, (0.0, 0.0, -draft / 2.0))
    derived = equivalent_inertia_from_hydrostatics(
        props.displaced_volume,
        props.waterplane_area,
        props.waterplane_moments,
        props.waterplane_center,
        mass=mass,
    )
    truth = box_inertia_cg(length, beam, draft, mass)
    axes = np.asarray(derived["semi_axes"], dtype=float)
    return {
        "length": length,
        "beam": beam,
        "draft": draft,
        "mass": mass,
        "equivalent_semi_axes": axes.tolist(),
        "true_half_extents": [length / 2.0, beam / 2.0, draft],
        "waterplane_area_ratio": float(derived["waterplane_area_ratio"]),
        "inertia_cg": np.diag(derived["inertia_cg"]).tolist(),
        "closed_form_inertia_cg": truth.tolist(),
        "error_vs_closed_form": _rel(np.diag(derived["inertia_cg"]), truth).tolist(),
        "legacy_default_error": _rel(
            np.diag(legacy_default_inertia(mass, (0.0, 0.0, -draft / 2.0))), truth
        ).tolist(),
    }


def natural_frequency_impact(
    inertia_ratio_by_mode: np.ndarray, added_mass=None
) -> dict:
    """``omega_n = sqrt(C/(M + A))``: the M error maps to a frequency factor.

    With the same restoring and added mass, ``omega_n`` scales as
    ``(M+A)^(-1/2)``, so an inertial entry that is ``f`` times too large gives a
    natural frequency that is ``sqrt(f)`` too low.  No BEM solve is needed for
    that factor, and it is reported per rotational mode.
    """
    ratio = np.asarray(inertia_ratio_by_mode, dtype=float)
    if added_mass is None:
        factor = 1.0 / np.sqrt(1.0 + ratio)
    else:
        a = np.asarray(added_mass, dtype=float)
        factor = np.sqrt(a / (a * (1.0 + ratio)))
    return {
        "inertia_ratio": ratio.tolist(),
        "mode": list(MODES[3:]),
        "natural_frequency_factor": factor.tolist(),
    }


def run(subdivisions=(2, 3, 4)) -> dict:
    """Assemble the audit payload (all numbers computed, none transcribed)."""
    legacy_rows, legacy_ratio = origin_dependence_ladder()
    anchor = spheroid_ladder(subdivisions=subdivisions)
    prolate = spheroid_ladder(semi_axes=(10.0, 2.0, 1.0), subdivisions=subdivisions, mass=1.0e3)
    sphere = spheroid_ladder(semi_axes=(1.0, 1.0, 1.0), subdivisions=subdivisions, mass=1.0e3)
    barge = barge_case()

    # MC-1: does the geometry-anchored default agree with the assembly the
    # verification cases already use (an explicit closed-form tensor referred to
    # the same CG)?  The CG is taken at the equivalent solid's own centroid so
    # that the two routes describe the same mass distribution.
    mass = 1.0e6
    semi = (60.0, 12.0, 8.0)
    hull = spheroid_hull_mesh(semi, subdivisions=4)
    cg = (0.0, 0.0, -3.0 * semi[2] / 8.0)
    props = compute_hydrostatic_restoring_matrix(hull, mass, cg)
    explicit = compute_mass_matrix(
        mass, cg, np.diag(half_ellipsoid_inertia_cg(semi, mass)), inertia_about="cg"
    )
    derived_matrix = mass_matrix_from_hull_geometry(props, mass, cg)
    entry_error = np.abs(derived_matrix - explicit) / np.maximum(
        np.abs(explicit), 1e-6 * np.max(np.abs(explicit))
    )
    # MC-2: change-of-origin law.  Translate hull, CG and reference point by the
    # same vector: the CG-referred inertia must not move.  (The shift is kept in
    # the z = 0 plane because a wetted hull is defined by z <= 0.)
    shift = np.array([1000.0, -500.0, 0.0])
    hull_shift = WettedHullMesh(hull.vertices + shift, hull.faces)
    cg_shift = tuple(np.add(cg, shift))
    props_shift = compute_hydrostatic_restoring_matrix(hull_shift, mass, cg_shift)
    derived_shift = equivalent_inertia_from_hydrostatics(
        props_shift.displaced_volume,
        props_shift.waterplane_area,
        props_shift.waterplane_moments,
        props_shift.waterplane_center,
        mass=mass,
        center_of_gravity=cg_shift,
        reference_point=tuple(shift),
    )
    derived_base = equivalent_inertia_from_hydrostatics(
        props.displaced_volume,
        props.waterplane_area,
        props.waterplane_moments,
        props.waterplane_center,
        mass=mass,
        center_of_gravity=cg,
    )
    inv_cg = float(
        np.max(
            np.abs(
                np.diag(derived_shift["inertia_cg"])
                - np.diag(derived_base["inertia_cg"])
            )
        )
        / np.max(np.abs(half_ellipsoid_inertia_cg(semi, mass)))
    )

    truth = half_ellipsoid_inertia_cg(semi, mass)
    legacy = np.diag(legacy_default_inertia(mass, cg))
    geometry = np.diag(
        equivalent_inertia_from_hydrostatics(
            props.displaced_volume,
            props.waterplane_area,
            props.waterplane_moments,
            props.waterplane_center,
            mass=mass,
        )["inertia_cg"]
    )

    return {
        "case": "CASE-BEM-RIGID-BODY-MASS / default-inertia audit",
        "modes": list(MODES),
        "origin_dependence": {"ladder": legacy_rows, "max_over_min": legacy_ratio},
        "anchor_spheroid": anchor,
        "prolate_spheroid": prolate,
        "hemisphere": sphere,
        "barge": barge,
        "assembly_agreement": {
            "description": "mass_matrix_from_hull_geometry vs explicit closed-form tensor, anchor ellipsoid, subdivisions=4",
            "max_entry_relative_error": float(np.max(entry_error)),
        },
        "change_of_origin": {
            "description": "hull+CG+reference point translated by (1000,-500,250) m; CG inertia must not move",
            "max_relative_shift_of_inertia_cg": inv_cg,
        },
        "rotational_impact": {
            "closed_form_inertia_cg": truth.tolist(),
            "geometry_derived_inertia_cg": geometry.tolist(),
            "legacy_default_inertia_cg": legacy.tolist(),
            "legacy_natural_frequency_factor": natural_frequency_impact(
                _rel(legacy, truth)
            )["natural_frequency_factor"],
            "geometry_natural_frequency_factor": natural_frequency_impact(
                _rel(geometry, truth)
            )["natural_frequency_factor"],
        },
        "verdict": {
            "legacy_default_is_origin_dependent": bool(legacy_ratio > 10.0),
            "legacy_default_rejected": bool(np.max(np.abs(_rel(legacy, truth))) > 0.03),
            "geometry_derived_within_3pct_finest": bool(anchor["worst_derived_error_at_finest"] < 0.03),
            "geometry_derived_agrees_with_explicit_assembly": bool(np.max(entry_error) < 0.03),
            "change_of_origin_law_holds": bool(inv_cg < 1e-2),
            "barge_is_declared_approximation": bool(
                np.max(np.abs(barge["error_vs_closed_form"])) > 0.10
            ),
        },
    }


def make_figures(payload: dict, outdir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    ax = axes[0]
    ladder = payload["origin_dependence"]["ladder"]
    xs = [row["cg_x"] for row in ladder]
    ys = [row["legacy_roll_inertia"] for row in ladder]
    ax.semilogy(xs, ys, "o-", color="#b2182b")
    ax.set_xlabel("|origin $\\to$ CG| offset $r_g$ in x (m), same body")
    ax.set_ylabel("legacy default $I_{xx}$ (kg m$^2$)")
    ax.set_title(
        "legacy default $k=0.4|r_g|$ is origin-dependent\n(max/min = "
        f"{payload['origin_dependence']['max_over_min']:.0f})"
    )
    ax.grid(alpha=0.3)

    ax = axes[1]
    for key, label, color in (
        ("anchor_spheroid", "spheroid 60:12:8", "#2166ac"),
        ("prolate_spheroid", "spheroid 10:2:1", "#1b7837"),
        ("hemisphere", "hemisphere", "#762a83"),
    ):
        rows = payload[key]["ladder"]
        panels = [row["panel_count"] for row in rows]
        err = [100.0 * max(abs(v) for v in row["error_vs_closed_form"]) for row in rows]
        ax.loglog(panels, err, "o-", label=label, color=color)
    ax.axhline(3.0, color="k", ls="--", lw=1.0, label="3 % criterion")
    ax.axhline(
        100.0 * payload["anchor_spheroid"]["worst_legacy_error"],
        color="#b2182b",
        ls=":",
        lw=1.0,
        label="legacy default (anchor shape)",
    )
    ax.set_xlabel("panels")
    ax.set_ylabel("worst relative error vs closed form (%)")
    ax.set_title("geometry-derived inertia converges")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = axes[2]
    rot = payload["rotational_impact"]
    modes = payload["modes"][3:]
    x = np.arange(len(modes))
    leg = np.array(rot["legacy_natural_frequency_factor"])
    geo = np.array(rot["geometry_natural_frequency_factor"])
    ax.bar(x - 0.2, leg, 0.4, label="legacy default", color="#b2182b")
    ax.bar(x + 0.2, geo, 0.4, label="geometry-derived", color="#2166ac")
    ax.axhline(1.0, color="k", lw=1.0)
    ax.set_xticks(x, modes)
    ax.set_ylabel("$\\omega_n / \\omega_{n,true}$")
    ax.set_title("natural-frequency factor through $M+A$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    path = outdir / "inertia_default_audit.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        default=str(Path(__file__).resolve().parents[3] / "docs" / "case_bem_rigid_body_mass"),
    )
    args = parser.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    payload = run()
    figures = make_figures(payload, outdir)
    payload["figures"] = [str(p) for p in figures]
    (outdir / "default_inertia_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["verdict"], indent=2))
    print("wrote", outdir / "default_inertia_audit.json")
    return 0 if all(payload["verdict"].values()) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())