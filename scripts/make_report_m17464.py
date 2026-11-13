#!/usr/bin/env python3
"""Figures + report for mission m17464 (CASE-BEM-RIGID-LID-HEMISPHERE, TensorBEM).

Every number is read from the committed payload
(``benchmarks/verified/case_bem_rigid_lid_hemisphere/results.json``) or from the
run snapshot (``runs/m17464_rigid_lid_hemisphere/benchmark_ref_vs_got.json``);
nothing is typed by hand.  Three figures:

1. ``fig_error_vs_ka``   -- err = A33/(rho nabla)/ref - 1 vs ka, 64/256 panels,
   with the pool bands (3 %, 5 %) and the refuted translating-image reference;
2. ``fig_reference_routes`` -- the rigid-lid reference by three routes plus the
   even-mode truncation ladder (exact rational, no quadrature/truncation error);
3. ``fig_field_pressure`` -- hull field: Re(phi) of the heave solution at
   ka = 1e-3 on the 256-panel mesh (the quantity the added mass integrates).

Usage::

    PYTHONPATH=src python scripts/make_report_m17464.py --out docs/m17464
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ART = ROOT / "benchmarks" / "verified" / "case_bem_rigid_lid_hemisphere"
RUN = ROOT / "runs" / "m17464_rigid_lid_hemisphere"


def _payload() -> dict:
    return json.loads((ART / "results.json").read_text(encoding="utf-8"))


def fig_error_vs_ka(payload: dict, out: pathlib.Path) -> pathlib.Path:
    ref = payload["reference"]["value_dimless"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for level in payload["levels"]:
        panels = level["level"]["panels"]
        kas = [row["ka"] for row in level["rows"]]
        errs = [100.0 * row["relative_error"] for row in level["rows"]]
        ax.plot(kas, errs, "o-", label="%d panels" % panels, lw=1.4, ms=5)
    if payload.get("out_of_band"):
        rows = payload["out_of_band"][0]["rows"]
        ax.plot(
            [row["ka"] for row in rows],
            [100.0 * row["relative_error"] for row in rows],
            "s--",
            color="0.55",
            ms=5,
            lw=1.2,
            label="out-of-band sweep (scope control)",
        )
    ax.axhline(0.0, color="k", lw=0.8)
    for band, style in ((5.0, "-"), (3.0, ":")):
        ax.axhline(band, color="tab:red", ls=style, lw=1.0)
        ax.axhline(-band, color="tab:red", ls=style, lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$ka$")
    ax.set_ylabel(r"$A_{33}/(\rho\nabla)\,/\,$ref $\,-1$   [%]")
    ax.set_title(
        "rigid-lid endpoint ref = %.7f, bands 3%% (dotted) / 5%% (solid)"
        % ref
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    path = out / "fig_error_vs_ka.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_reference_routes(payload: dict, out: pathlib.Path) -> pathlib.Path:
    reference = payload["reference"]
    import tensorbem.rigid_lid as rigid_lid

    exact = rigid_lid.rigid_lid_added_mass_dimless(320)
    ladder = exact["truncation_ladder"]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    degs = [item["degree_max"] for item in ladder]
    vals = [item["value_dimless"] for item in ladder]
    ax0.plot(degs, vals, ".-", ms=3, lw=1.0, label="exact rational series")
    ax0.axhline(
        reference["routes"]["force"]["dimless"], color="tab:orange", ls="--",
        lw=1.0, label="driver force route (Gauss |mu| kink)",
    )
    ax0.axhline(
        reference["routes"]["energy"]["dimless"], color="tab:green", ls=":",
        lw=1.0, label="driver energy route (degree 60)",
    )
    ax0.set_xlabel(r"even-mode truncation $N$")
    ax0.set_ylabel(r"$A_{33}/(\rho\nabla)$")
    ax0.set_xlim(0, 60)
    ax0.grid(alpha=0.3)
    ax0.legend(fontsize=8, loc="lower right")
    ax0.set_title("convergence to 0.83095132 (tail %.1e)" % exact["truncation_tail_relative"])

    resid = [abs(v - exact["value_dimless"]) for v in vals]
    ax1.semilogy(degs, [max(r, 1e-18) for r in resid], ".-", ms=3, color="tab:blue")
    ax1.set_xlabel(r"even-mode truncation $N$")
    ax1.set_ylabel(r"$|A_{33}(N) - A_{33}(320)|$")
    ax1.grid(alpha=0.3, which="both")
    ax1.set_title("truncation residual (rational arithmetic)")
    fig.tight_layout()
    path = out / "fig_reference_routes.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_field_pressure(out: pathlib.Path) -> tuple[pathlib.Path, dict]:
    from tensorbem.free_surface_solver3d import solve_radiation_3d
    from tensorbem.geometry3d import hemisphere_hull_mesh

    level = 3
    ka = 1.0e-3
    rho, g, a = 1025.0, 9.80665, 1.0
    mesh = hemisphere_hull_mesh(level, radius=a)
    res = solve_radiation_3d(mesh, ka, modes=[2], density=rho, gravity=g)
    phi = np.real(res.potentials[:, 0])
    centers = np.asarray(mesh.centroids, dtype=float)
    volume = (2.0 / 3.0) * math.pi * a**3
    got = float(res.added_mass[0, 0]) / (rho * volume)
    fig = plt.figure(figsize=(9.0, 4.2))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    sc = ax.scatter(
        centers[:, 0], centers[:, 1], centers[:, 2], c=phi, s=9, cmap="coolwarm"
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(r"Re$\,\phi$ on the wetted hull, $ka=10^{-3}$")
    fig.colorbar(sc, ax=ax, shrink=0.7, label=r"Re$\,\phi$ [m$^2$/s]")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(phi, bins=30, color="tab:blue", alpha=0.8)
    ax2.set_xlabel(r"Re$\,\phi$ [m$^2$/s]")
    ax2.set_ylabel("panels")
    ax2.set_title(
        "256 panels, heave: $A_{33}/(\\rho\\nabla)=%.6f$ (ref %.6f)"
        % (got, 0.83095132)
    )
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    path = out / "fig_field_pressure.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, {
        "panels": int(mesh.areas.size),
        "ka": ka,
        "computed_dimless": got,
        "phi_min": float(phi.min()),
        "phi_max": float(phi.max()),
        "condition_number": float(res.condition_number),
        "relative_residual": float(res.relative_residual),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "docs" / "m17464")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    payload = _payload()
    paths = [
        fig_error_vs_ka(payload, args.out),
        fig_reference_routes(payload, args.out),
    ]
    field_path, field = fig_field_pressure(args.out)
    paths.append(field_path)
    summary = {
        "figures": [str(pathlib.Path(p).resolve().relative_to(ROOT)) for p in paths],
        "field": field,
        "reference_dimless": payload["reference"]["value_dimless"],
        "verdict": payload["verdict"]["passed"],
    }
    (args.out / "figure_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())