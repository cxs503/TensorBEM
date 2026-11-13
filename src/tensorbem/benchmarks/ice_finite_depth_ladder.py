"""Depth/thickness ladder for ice-covered radiation in finite water depth.

Generates the published artefacts of ``docs/ice_finite_depth_radiation.md``:

* ``ladder.json`` -- every number of the document, machine readable;
* ``ladder.md``   -- the tables of the document, verbatim;
* ``ice_finite_depth_ladder.png`` -- the figure.

Run from the repository root::

    PYTHONPATH=src python -m tensorbem.benchmarks.ice_finite_depth_ladder \
        --output docs/ice_finite_depth_radiation
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import time
from typing import Any

import numpy as np

from ..finite_depth_solver3d import solve_radiation_finite_depth_3d
from ..geometry3d import submerged_sphere_mesh
from ..ice import IceCover
from ..ice_finite_depth import (
    ice_finite_depth_wavenumber,
    radiation_report,
    solve_radiation_finite_depth_ice,
)
from ..ice_rao import DEEP_WATER_MARGIN, solve_radiation_3d_ice

OMEGA = 1.0
THICKNESS = 0.9
DEPTHS = (10.0, 15.0, 20.0, 30.0, 50.0, 100.0)
THICKNESSES = (0.2, 0.5, 0.9, 1.5, 2.5)
FIXED_DEPTH_FOR_THICKNESS = 30.0


def _mesh(subdivisions: int = 0):
    return submerged_sphere_mesh(subdivisions, radius=1.0, submergence=1.5)


def _relative(a: np.ndarray, b: np.ndarray) -> float:
    """Frobenius relative difference; zero-safe for the near-zero couplings."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(b))
    if denom == 0.0:
        return float("nan")
    return float(np.linalg.norm(a - b) / denom)


def _row(omega: float, ice: IceCover, mesh) -> dict[str, Any]:
    started = time.perf_counter()
    result = solve_radiation_finite_depth_ice(mesh, omega, ice)
    elapsed = time.perf_counter() - started
    report = radiation_report(result)
    row: dict[str, Any] = {
        "water_depth": float(ice.water_depth),
        "thickness": float(ice.thickness),
        "mu": float(ice.mu),
        "depth_parameter": report["depth_parameter"],
        "wavenumber_ice": report["wavenumber_ice"],
        "wavenumber_open": report["wavenumber_open"],
        "robin_coefficient": report["robin_coefficient"],
        "implied_omega": report["implied_omega"],
        "added_mass_heave": report["added_mass_heave"],
        "added_mass_surge": report["added_mass_surge"],
        "damping_heave": report["damping_heave"],
        "damping_surge": report["damping_surge"],
        "condition_number": report["condition_number"],
        "relative_residual": report["relative_residual"],
        "seconds": elapsed,
        "deep_water_reference": None,
        "deep_water_margin": DEEP_WATER_MARGIN,
    }
    if report["depth_parameter"] >= DEEP_WATER_MARGIN:
        reference = solve_radiation_3d_ice(mesh, omega, ice)
        row["deep_water_reference"] = {
            "added_mass_heave": float(np.asarray(reference.added_mass)[2, 2]),
            "damping_heave": float(np.asarray(reference.damping)[2, 2]),
            "added_mass_relative_difference": _relative(
                result.added_mass, reference.added_mass
            ),
            "damping_relative_difference": _relative(
                result.damping, reference.damping
            ),
        }
    return row


def _dispersion_check(omega: float, ice: IceCover) -> dict[str, float]:
    """Residual of the finite-depth relation at the returned wavenumber."""
    k = ice_finite_depth_wavenumber(omega, ice)
    tanh = math.tanh(k * ice.water_depth)
    residual = k * tanh - omega * omega / (ice.gravity - ice.mu * omega * omega)
    return {"wavenumber": k, "residual": abs(residual)}


def _no_ice_limit(omega: float) -> dict[str, float]:
    """``mu -> 0`` must reproduce the open-water finite-depth solver."""
    ice = IceCover(thickness=1.0e-9, water_depth=20.0)
    mesh = _mesh(0)
    result = solve_radiation_finite_depth_ice(mesh, omega, ice)
    nu = omega * omega / ice.gravity
    open_water = solve_radiation_finite_depth_3d(
        mesh, nu, ice.water_depth, gravity=ice.gravity
    )
    return {
        "mu": float(ice.mu),
        "added_mass_relative_difference": _relative(
            result.added_mass, open_water.added_mass
        ),
        "damping_relative_difference": _relative(
            result.damping, open_water.damping
        ),
        "robin_coefficient_relative_difference": abs(
            result.robin_coefficient - nu
        )
        / nu,
    }


def _mesh_check(omega: float) -> list[dict[str, Any]]:
    """Two meshes at one frequency: is the 20-panel answer trustworthy?"""
    rows = []
    for subdivisions in (0, 1):
        mesh = _mesh(subdivisions)
        ice = IceCover(thickness=THICKNESS, water_depth=20.0)
        result = solve_radiation_finite_depth_ice(mesh, omega, ice)
        rows.append(
            {
                "subdivisions": subdivisions,
                "panels": int(mesh.panel_count),
                "added_mass_heave": float(np.asarray(result.added_mass)[2, 2]),
                "damping_heave": float(np.asarray(result.damping)[2, 2]),
            }
        )
    base = rows[0]
    fine = rows[1]
    for key in ("added_mass_heave", "damping_heave"):
        fine[f"{key}_change_vs_coarse"] = (
            (fine[key] - base[key]) / abs(base[key]) if base[key] else float("nan")
        )
    return rows


def _figure(payload: dict[str, Any], path: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depth_rows = payload["depth_ladder"]
    kH = [row["depth_parameter"] for row in depth_rows]
    b_fd = [row["damping_heave"] for row in depth_rows]
    b_dw = [
        row["deep_water_reference"]["damping_heave"]
        if row["deep_water_reference"]
        else np.nan
        for row in depth_rows
    ]
    diff = [
        row["deep_water_reference"]["damping_relative_difference"]
        if row["deep_water_reference"]
        else np.nan
        for row in depth_rows
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    ax = axes[0]
    ax.plot(kH, b_fd, "o-", label="finite-depth ice module")
    ax.plot(kH, b_dw, "s--", label=r"deep-water ice (ice_rao), $k_iH\geq3$")
    ax.axvline(DEEP_WATER_MARGIN, color="0.5", lw=1.0, ls=":")
    ax.annotate(
        "deep-water guard",
        xy=(DEEP_WATER_MARGIN, ax.get_ylim()[0]),
        xytext=(DEEP_WATER_MARGIN + 0.3, ax.get_ylim()[0]),
        fontsize=8,
        color="0.35",
    )
    ax.set_xlabel(r"$k_i H$ (-)")
    ax.set_ylabel("heave damping (kg/s)")
    ax.set_title("ice-covered heave damping vs depth")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.loglog(kH, diff, "o-")
    ax.axhline(1e-2, color="0.5", lw=1.0, ls=":", label="1% band")
    ax.set_xlabel(r"$k_i H$ (-)")
    ax.set_ylabel("relative difference in damping")
    ax.set_title("finite-depth vs deep-water ice kernel")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("m18004: mass-loading ice cover in finite water depth", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("### Depth ladder (omega = %.3g rad/s, ice thickness %.3g m)" % (OMEGA, THICKNESS))
    lines.append("")
    lines.append(
        "| H (m) | k_i H | K (1/m) | A_33 (kg) | B_33 (kg/s) | vs deep-water ice: dA | dB |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for row in payload["depth_ladder"]:
        ref = row["deep_water_reference"]
        if ref:
            da = f"{ref['added_mass_relative_difference']:.3e}"
            db = f"{ref['damping_relative_difference']:.3e}"
        else:
            da = db = "guard (k_iH<3)"
        lines.append(
            "| {H:.0f} | {kH:.2f} | {K:.5f} | {A:.2f} | {B:.3f} | {da} | {db} |".format(
                H=row["water_depth"],
                kH=row["depth_parameter"],
                K=row["robin_coefficient"],
                A=row["added_mass_heave"],
                B=row["damping_heave"],
                da=da,
                db=db,
            )
        )
    lines.append("")
    lines.append(
        "### Thickness ladder (omega = %.3g rad/s, H = %.0f m)"
        % (OMEGA, FIXED_DEPTH_FOR_THICKNESS)
    )
    lines.append("")
    lines.append("| h_i (m) | mu (m) | k_i H | A_33 (kg) | B_33 (kg/s) |")
    lines.append("|---|---|---|---|---|")
    for row in payload["thickness_ladder"]:
        lines.append(
            "| {h:.2f} | {mu:.3f} | {kH:.2f} | {A:.2f} | {B:.3f} |".format(
                h=row["thickness"],
                mu=row["mu"],
                kH=row["depth_parameter"],
                A=row["added_mass_heave"],
                B=row["damping_heave"],
            )
        )
    lines.append("")
    limit = payload["no_ice_limit"]
    lines.append(
        "### Analytic limits\n\n"
        "| check | quantity | value |\n|---|---|---|\n"
        "| dispersion `k tanh(kH) = K` | residual (1/m) | {res:.3e} |\n"
        "| mu -> 0 vs open-water solver | added mass rel. diff | {a:.3e} |\n"
        "| mu -> 0 vs open-water solver | damping rel. diff | {b:.3e} |\n"
        "| mu -> 0 vs `omega^2/g` | `K` rel. diff | {c:.3e} |".format(
            res=payload["dispersion"]["residual"],
            a=limit["added_mass_relative_difference"],
            b=limit["damping_relative_difference"],
            c=limit["robin_coefficient_relative_difference"],
        )
    )
    lines.append("")
    mesh = payload["mesh_check"]
    lines.append("### Mesh sensitivity (H = 20 m, one frequency)\n")
    lines.append("| subdivisions | panels | A_33 (kg) | B_33 (kg/s) | A_33 rel. change | B_33 rel. change |")
    lines.append("|---|---|---|---|---|---|")
    for row in mesh:
        da = row.get("added_mass_heave_change_vs_coarse")
        db = row.get("damping_heave_change_vs_coarse")
        lines.append(
            "| {s} | {p} | {a:.2f} | {b:.3f} | {da} | {db} |".format(
                s=row["subdivisions"],
                p=row["panels"],
                a=row["added_mass_heave"],
                b=row["damping_heave"],
                da="-" if da is None else f"{da:+.3e}",
                db="-" if db is None else f"{db:+.3e}",
            )
        )
    lines.append("")
    return "\n".join(lines)


def run(output: pathlib.Path) -> dict[str, Any]:
    output = pathlib.Path(output)
    output.mkdir(parents=True, exist_ok=True)
    mesh = _mesh(0)

    depth_rows = [
        _row(OMEGA, IceCover(thickness=THICKNESS, water_depth=H), mesh) for H in DEPTHS
    ]
    thickness_rows = [
        _row(OMEGA, IceCover(thickness=h, water_depth=FIXED_DEPTH_FOR_THICKNESS), mesh)
        for h in THICKNESSES
    ]
    payload: dict[str, Any] = {
        "omega": OMEGA,
        "mesh_panels": int(mesh.panel_count),
        "deep_water_margin": DEEP_WATER_MARGIN,
        "depth_ladder": depth_rows,
        "thickness_ladder": thickness_rows,
        "dispersion": _dispersion_check(OMEGA, IceCover(thickness=THICKNESS, water_depth=30.0)),
        "no_ice_limit": _no_ice_limit(OMEGA),
        "mesh_check": _mesh_check(OMEGA),
    }

    (output / "ladder.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    (output / "ladder.md").write_text(_markdown(payload))
    _figure(payload, output / "ice_finite_depth_ladder.png")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="docs/ice_finite_depth_radiation",
        type=pathlib.Path,
        help="directory for ladder.json / ladder.md / the figure",
    )
    args = parser.parse_args(argv)
    payload = run(args.output)
    print(_markdown(payload))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
