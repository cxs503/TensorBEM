"""Kochin/Hulme surge-damping panel-convergence study (m2609).

``tests/test_kochin_validation.py::test_surge_damping_vs_hulme`` is red on
main: at 256 panels (subdivisions=3) the BEM surge radiation damping sits
3.53 % above the Hulme (1982) value, outside the 2.5 % gate.  The m2544 CI
triage explicitly deferred the deciding panel study ("需面板收敛调查，禁止
无证据放宽带宽，留待专项").  This module is that study.

Three candidate causes are separated, each with its own axis:

1. **Panel discretisation.**  Refine the hemisphere mesh (64 -> 256 -> 1024
   panels).  A discretisation error must shrink at the observed order of the
   constant-panel scheme; a normalisation error would stay mesh-independent.
2. **Near-field vs far-field reading.**  The same radiation solution yields
   two damping readings: the surface-pressure (near-field) value and the
   Kochin far-field value.  If only one of them drifts from Hulme, the defect
   localises to that reading, not to the potential itself.
3. **Reference self-consistency.**  The Hulme reference carries *both* a
   near-field damping (boundary integral of the exact potential) and a
   far-field damping (6*pi*ka^3*|c0|^2 from the exact Kochin amplitude).
   Their mutual offset is the reference's own internal error and sets the
   floor any BEM number can reach.

Order convention (m3280).  The ``orders`` field of each row reports the
observed order in the panel size ``h`` -- the convention the documentation
uses (``docs/marine_bem_benchmarks.md`` table 10 and the m2609 paragraph:
"对 h 观测阶 1.62/1.88/1.96").  In three dimensions the panel count scales
as ``N ~ h**-2``, so ``h_lo/h_hi = sqrt(N_hi/N_lo)`` and the per-panel-count
reading is exactly half of it for a 4x panel refinement.  The legacy
per-panel-count value is kept under ``orders_vs_panels`` so both
conventions stay reproducible from one artefact; before m3280 the field
named ``orders`` silently carried the per-panel-count value, which is why
the module printed 0.81/0.94 where the documentation quotes 1.62/1.88.

Usage::

    PYTHONPATH=src python3 -m tensorbem.benchmarks.kochin_hulme_convergence \
        --subdivisions 2 3 4 \
        --json artifacts/kochin_hulme_convergence/study.json \
        --figures artifacts/kochin_hulme_convergence
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from tensorbem.analytic import hulme_hemisphere_radiation
from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    far_field_damping_matrix_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import hemisphere_hull_mesh

_DENSITY = 1025.0
_GRAVITY = 9.80665
_MODES = (0, 2)  # surge, heave


def _hulme_reference(ka: float, radius: float) -> dict:
    """Dimensionalised Hulme (1982) reference, near- and far-field."""
    hulme = hulme_hemisphere_radiation(ka, radius=radius, density=_DENSITY)
    vol = hulme.displaced_volume
    rho = _DENSITY
    omega = hulme.omega
    return {
        "ka": ka,
        "radius": radius,
        "omega": omega,
        "displaced_volume": vol,
        "b11_near": rho * omega * vol * hulme.surge_damping_dimless,
        "b11_far": rho * omega * vol * hulme.surge_far_field_damping_dimless,
        "b33_near": rho * omega * vol * hulme.heave_damping_dimless,
        "b33_far": rho * omega * vol * hulme.heave_far_field_damping_dimless,
        "b11_dimless": hulme.surge_damping_dimless,
        "b11_ff_dimless": hulme.surge_far_field_damping_dimless,
        "b33_dimless": hulme.heave_damping_dimless,
        "b33_ff_dimless": hulme.heave_far_field_damping_dimless,
        "a11_dimless": hulme.surge_added_mass_dimless,
        "a33_dimless": hulme.heave_added_mass_dimless,
        "n_terms": 48,
        "quad_order": 128,
    }


def _rel(a: float, b: float) -> float:
    return (a - b) / b


def run_study(
    subdivisions: list[int],
    ka: float = 1.0,
    radius: float = 1.0,
    n_azimuth: int = 72,
) -> dict:
    ref = _hulme_reference(ka, radius)
    rows = []
    for s in subdivisions:
        t0 = time.time()
        mesh = hemisphere_hull_mesh(s, radius=radius)
        ops = assemble_free_surface_operators_3d(mesh, ka, chief_points="auto")
        rad = solve_radiation_3d(mesh, ka, modes=list(_MODES), operators=ops)
        b_near = np.array([rad.damping[0, 0], rad.damping[1, 1]])
        b_far = np.diag(far_field_damping_matrix_3d(mesh, rad, n_azimuth=n_azimuth))
        a_near = np.array([rad.added_mass[0, 0], rad.added_mass[1, 1]])
        rows.append(
            {
                "subdivisions": s,
                "panels": int(mesh.panel_count),
                "wall_s": round(time.time() - t0, 2),
                "condition_number": float(rad.condition_number),
                "relative_residual": float(rad.relative_residual),
                "b11_near": float(b_near[0]),
                "b33_near": float(b_near[1]),
                "b11_far": float(b_far[0]),
                "b33_far": float(b_far[1]),
                "a11": float(a_near[0]),
                "a33": float(a_near[1]),
                "err_b11_near": _rel(float(b_near[0]), ref["b11_near"]),
                "err_b33_near": _rel(float(b_near[1]), ref["b33_near"]),
                "err_b11_far": _rel(float(b_far[0]), ref["b11_far"]),
                "err_b33_far": _rel(float(b_far[1]), ref["b33_far"]),
                "ratio_b11": float(b_far[0] / b_near[0]),
                "ratio_b33": float(b_far[1] / b_near[1]),
            }
        )
    # reference internal consistency (its own near/far offset)
    ref["self_offset_b11"] = _rel(ref["b11_far"], ref["b11_near"])
    ref["self_offset_b33"] = _rel(ref["b33_far"], ref["b33_near"])
    # Convergence orders between successive meshes.  ``orders`` is the order
    # in h (documentation convention); ``orders_vs_panels`` is the legacy
    # order in the panel count N.  With N ~ h**-2 the two differ by exactly
    # a factor of 2, which the regression test pins.
    for i in range(1, len(rows)):
        lo, hi = rows[i - 1], rows[i]
        for key in ("err_b11_near", "err_b33_near", "err_b11_far", "err_b33_far"):
            if abs(lo[key]) > 1e-12 and abs(hi[key]) > 1e-12:
                log_ratio = math.log2(abs(lo[key] / hi[key]))
                hi.setdefault("orders", {})[key] = float(
                    log_ratio / math.log2(math.sqrt(hi["panels"] / lo["panels"]))
                )
                hi.setdefault("orders_vs_panels", {})[key] = float(
                    log_ratio / math.log2(hi["panels"] / lo["panels"])
                )
    return {"reference": ref, "rows": rows}


def _print_table(study: dict) -> None:
    ref = study["reference"]
    print(
        f"Hulme ref: b11_near={ref['b11_near']:.4f} b11_far={ref['b11_far']:.4f} "
        f"(self {ref['self_offset_b11']:+.4%}) | "
        f"b33_near={ref['b33_near']:.4f} b33_far={ref['b33_far']:.4f} "
        f"(self {ref['self_offset_b33']:+.4%})"
    )
    hdr = (
        f"{'s':>3} {'N':>5} {'B11n':>10} {'B33n':>10} {'B11f':>10} {'B33f':>10} "
        f"{'eB11n':>9} {'eB33n':>9} {'eB11f':>9} {'eB33f':>9} {'rB11':>8} {'rB33':>8} {'t_s':>7}"
    )
    print(hdr)
    for r in study["rows"]:
        print(
            f"{r['subdivisions']:>3} {r['panels']:>5} {r['b11_near']:>10.4f} "
            f"{r['b33_near']:>10.4f} {r['b11_far']:>10.4f} {r['b33_far']:>10.4f} "
            f"{r['err_b11_near']:>9.4%} {r['err_b33_near']:>9.4%} "
            f"{r['err_b11_far']:>9.4%} {r['err_b33_far']:>9.4%} "
            f"{r['ratio_b11']:>8.5f} {r['ratio_b33']:>8.5f} {r['wall_s']:>7.1f}"
        )
        if "orders" in r:
            o = r["orders"]
            print(
                f"    orders (in h): "
                + " ".join(f"{k}={v:.2f}" for k, v in o.items())
            )
            op = r.get("orders_vs_panels")
            if op:
                print(
                    f"    orders (in N): "
                    + " ".join(f"{k}={v:.2f}" for k, v in op.items())
                )


def _write_figures(study: dict, outdir: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    outdir.mkdir(parents=True, exist_ok=True)
    rows = study["rows"]
    n = [r["panels"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for key, lab in (
        ("err_b11_near", "B11 near vs Hulme"),
        ("err_b11_far", "B11 far vs Hulme"),
        ("err_b33_near", "B33 near vs Hulme"),
        ("err_b33_far", "B33 far vs Hulme"),
    ):
        ax.plot(n, [abs(r[key]) for r in rows], marker="o", label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("panels")
    ax.set_ylabel("|rel. error|")
    ax.set_title("Hemisphere ka=1: damping error vs Hulme (1982)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(n, [r["ratio_b11"] for r in rows], marker="o", label="B11 far/near")
    ax.plot(n, [r["ratio_b33"] for r in rows], marker="s", label="B33 far/near")
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("panels")
    ax.set_ylabel("far / near")
    ax.set_title("Energy identity (far/near) per mode")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "kochin_hulme_convergence.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return [str(path)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subdivisions", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--ka", type=float, default=1.0)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--figures", type=Path, default=None)
    args = parser.parse_args(argv)

    study = run_study(args.subdivisions, ka=args.ka, radius=args.radius)
    _print_table(study)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(study, indent=2))
        print(f"wrote {args.json}")
    if args.figures is not None:
        for p in _write_figures(study, args.figures):
            print(f"wrote {p}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
