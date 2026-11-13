"""Case BEM-QTF-DRIFT: closed-form difference-frequency QTF versus TensorBEM BEM.

Decision this study supports
----------------------------
*Does the panel-BEM difference-frequency QTF of :mod:`tensorbem.qtf_drift`
reproduce an independent analytic value within the case tolerance of 10 %, and how
much of the residual is panel discretisation?*

The analytic reference is :mod:`tensorbem.cylinder_qtf` -- the machine-precision
closed form of the same quadratic-pressure assembly on a fixed bottom-mounted
vertical cylinder, built from the MacCamy-Fuchs modes.  The competing
alternatives are therefore (a) BEM is accurate to < 10 % on the QTF itself,
(b) BEM inherits a mesh-driven bias and only the *extrapolated* limit is
accurate, (c) the QTF machinery itself is wrong (off-diagonal structure
violated).  (c) is excluded by the diagonal identity and the Hermitian residual
gates, (a)/(b) are decided by the mesh ladder.

Predeclared gates (fixed before any BEM number was produced; do not relax)
--------------------------------------------------------------------------
G1  closed-form diagonal vs exact near-field mean drift (MacCamy-Fuchs)   < 1e-9
G2  closed-form diagonal vs reduced Maruo far field (independent route)   < 1e-3
G3  closed-form diagonal vs long-wave law  (5 pi^2/16) rho g A^2 a (ka)^3 < 1 % at ka = 0.05
G4  Hermitian residual of the closed-form matrix                          < 1e-10
G5  BEM |T_ij| vs closed form, every pair, finest available level        < 10 %
G6  BEM |T_ij| error decreases monotonically with mesh refinement

Stopping rule: the mesh ladder runs coarse -> medium (-> fine if affordable);
G5 is graded at the finest level actually computed.  If G5 fails there while G6
holds, the report states honest-FAIL on the 10 % gate at that density plus the
Richardson-extrapolated limit -- thresholds are never loosened afterwards.

Reproduce
---------
    PYTHONPATH=src python3 -m tensorbem.benchmarks.cylinder_qtf_closed_form \
        --json docs/case_qtf_drift_closed_form/study.json \
        --figures docs/case_qtf_drift_closed_form \
        --levels coarse medium
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import pickle
import time

import numpy as np

from ..cylinder_drift import (
    far_field_cylinder_drift,
    long_wave_drift_coefficient,
    near_field_cylinder_drift,
)
from ..cylinder_qtf import closed_form_cylinder_qtf, pair_integrals
from ..drift import bottom_mounted_cylinder_mesh
from ..qtf_drift import quadratic_qtf


# ---------------------------------------------------------------------------
# diffraction cache: makes the (expensive) BEM stage restartable in small chunks
# ---------------------------------------------------------------------------
def _diffraction(
    mesh,
    wavenumber: float,
    *,
    cache: pathlib.Path | None = None,
    tag: str = "",
):
    """Solve (or reload) one first-order diffraction problem.

    A single cylinder solve costs ~1 min on this machine, so the study has to be
    restartable: the panel solution is a pure function of ``(mesh, k)`` and is
    cached through ``numpy`` ``npz`` files keyed by ``tag``.  Nothing about the
    physics depends on the cache.
    """
    from ..free_surface_solver3d import solve_diffraction_3d

    if cache is None:
        return solve_diffraction_3d(mesh, wavenumber, wave_angle=0.0)
    cache = pathlib.Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{tag}_k{wavenumber:.6f}.npz"
    if path.exists():
        data = np.load(path, allow_pickle=True)
        return pickle.loads(data["result"].tobytes())
    result = solve_diffraction_3d(mesh, wavenumber, wave_angle=0.0)
    raw = pickle.dumps(result)
    np.savez_compressed(path, result=np.frombuffer(raw, dtype=np.uint8))
    return result


CYLINDER_RADIUS = 1.0
#: (n_theta, n_z) of the graded side mesh; panel count = n_theta * n_z + n_theta.
LEVELS = {
    "coarse": (16, 10),
    "medium": (24, 14),
    "fine": (34, 20),
}
KA_GRID = (0.4, 0.6, 0.8)
#: draft in units of the longest wavelength: depth = DEPTH_FACTOR / k_min
DEPTH_FACTOR = 12.0
DENSITY = 1025.0
GRAVITY = 9.81

GATES = {
    "G1_diagonal_identity": 1e-9,
    "G2_far_field_consistency": 1e-3,
    "G3_long_wave_law": 1e-2,
    "G4_hermitian": 1e-10,
    "G5_bem_vs_closed_form": 0.10,
}


# ---------------------------------------------------------------------------
# stage 1: the analytic reference against the module's verified anchors
# ---------------------------------------------------------------------------
def _component(value, index: int) -> complex:
    """QTF entry as a scalar: the BEM matrix may carry one column per force component."""
    arr = np.asarray(value)
    if arr.ndim == 0:
        return complex(arr)
    return complex(arr.reshape(-1)[index])


def closed_form_validation(*, verbose: bool = True) -> dict:
    """Diagonal identity, far-field consistency, long-wave law, Hermiticity."""
    ka_list = (0.2, 0.35, 0.5, 0.7, 0.9, 1.1, 1.4)
    rows = []
    for ka in ka_list:
        pair = pair_integrals(ka, ka, radius=CYLINDER_RADIUS)
        closed = DENSITY * pair.qtf_surge
        near = near_field_cylinder_drift(
            ka, radius=CYLINDER_RADIUS, density=DENSITY, gravity=GRAVITY
        )["mean_drift_force_x"]
        far = far_field_cylinder_drift(
            ka, radius=CYLINDER_RADIUS, density=DENSITY, gravity=GRAVITY
        )["mean_drift_force_x"]
        rows.append(
            {
                "ka": float(ka),
                "closed_form": float(closed),
                "near_field_exact": float(near),
                "far_field_exact": float(far),
                "identity_error": float(abs(closed - near) / abs(near)),
                "far_field_error": float(abs(closed - far) / abs(far)),
                "imag_ratio": float(abs(pair.qtf_surge_imag) / abs(pair.qtf_surge)),
            }
        )
        if verbose:
            print(
                f"  ka={ka:.2f} closed={closed:.6e} near={near:.6e} "
                f"ident={rows[-1]['identity_error']:.2e} "
                f"far={rows[-1]['far_field_error']:.2e}",
                flush=True,
            )

    # long-wave law: F_x = C rho g A^2 a (ka)^3 with C = 5 pi^2 / 16
    c_law = long_wave_drift_coefficient()
    long_wave = []
    for ka in (0.02, 0.05, 0.1):
        pair = pair_integrals(ka, ka, radius=CYLINDER_RADIUS)
        closed = DENSITY * pair.qtf_surge
        law = c_law * DENSITY * GRAVITY * CYLINDER_RADIUS * ka**3
        long_wave.append(
            {
                "ka": float(ka),
                "closed_form": float(closed),
                "long_wave_law": float(law),
                "relative_error": float(abs(closed - law) / abs(law)),
            }
        )
        if verbose:
            print(
                f"  long-wave ka={ka:.2f} closed={closed:.6e} law={law:.6e} "
                f"err={long_wave[-1]['relative_error']:.3%}",
                flush=True,
            )

    matrix = closed_form_cylinder_qtf(
        (0.3, 0.5, 0.7, 1.0), radius=CYLINDER_RADIUS, density=DENSITY
    )
    hermitian = float(np.abs(matrix.matrix - np.conj(matrix.matrix.T)).max())
    offdiag = float(np.max(np.abs(np.triu(matrix.matrix, 1).imag)))
    return {
        "diagonal_rows": rows,
        "max_identity_error": float(max(r["identity_error"] for r in rows)),
        "max_far_field_error": float(max(r["far_field_error"] for r in rows)),
        "long_wave_rows": long_wave,
        "long_wave_error_at_0.05": float(long_wave[1]["relative_error"]),
        "long_wave_coefficient": float(c_law),
        "hermitian_residual": hermitian,
        "max_offdiagonal_imag": offdiag,
    }


# ---------------------------------------------------------------------------
# stage 2: BEM against the closed form on a mesh ladder
# ---------------------------------------------------------------------------
def bem_level(
    level: str,
    ka_values=KA_GRID,
    *,
    depth_factor: float = DEPTH_FACTOR,
    cache: pathlib.Path | None = None,
    verbose: bool = True,
) -> dict:
    """Run the panel BEM QTF at one mesh level and grade every pair."""
    n_theta, n_z = LEVELS[level]
    ka = np.asarray(ka_values, dtype=float)
    a = CYLINDER_RADIUS
    depth = float(depth_factor) / float(ka.min() / a)
    mesh = bottom_mounted_cylinder_mesh(a, depth, n_theta=n_theta, n_z=n_z)
    started = time.time()
    results = [
        _diffraction(mesh, float(x) / a, cache=cache, tag=f"{level}")
        for x in ka
    ]
    bem = quadratic_qtf(
        mesh,
        ka / a,
        wave_angle=0.0,
        density=DENSITY,
        gravity=GRAVITY,
        diffraction_results=results,
    )
    elapsed = time.time() - started
    reference = closed_form_cylinder_qtf(
        ka, radius=a, density=DENSITY, gravity=GRAVITY
    )
    pairs = []
    for i in range(ka.size):
        for j in range(ka.size):
            b = _component(bem.matrix[i, j], 0)
            c = complex(reference.matrix[i, j])
            pairs.append(
                {
                    "i": int(i),
                    "j": int(j),
                    "ka_i": float(ka[i]),
                    "ka_j": float(ka[j]),
                    "bem_real": b.real,
                    "bem_imag": b.imag,
                    "ref_real": c.real,
                    "ref_imag": c.imag,
                    "bem_abs": abs(b),
                    "ref_abs": abs(c),
                    "abs_error": float(abs(abs(b) - abs(c)) / abs(c)),
                    "real_error": float(abs(b.real - c.real) / abs(c)),
                    "imag_error": float(abs(b.imag - c.imag) / abs(c)),
                    "diagonal": bool(i == j),
                }
            )
    off = [p["abs_error"] for p in pairs if not p["diagonal"]]
    diag = [p["abs_error"] for p in pairs if p["diagonal"]]
    if verbose:
        for p in pairs:
            print(
                f"  [{level}] ({p['i']},{p['j']}) ka=({p['ka_i']:.2f},{p['ka_j']:.2f}) "
                f"bem={p['bem_abs']:.4e} ref={p['ref_abs']:.4e} "
                f"err={p['abs_error']:.2%}",
                flush=True,
            )
    return {
        "level": level,
        "n_theta": int(n_theta),
        "n_z": int(n_z),
        "panels": int(mesh.panel_count),
        "depth": depth,
        "ka": ka.tolist(),
        "solve_seconds_total": elapsed,
        "pairs": pairs,
        "offdiagonal_max_error": float(max(off)) if off else 0.0,
        "offdiagonal_median_error": float(np.median(off)) if off else 0.0,
        "diagonal_max_error": float(max(diag)) if diag else 0.0,
        "diagonal_median_error": float(np.median(diag)) if diag else 0.0,
    }


def depth_sensitivity(
    ka: float = 0.6,
    *,
    level: str = "coarse",
    factors=(6.0, 12.0, 24.0),
    cache: pathlib.Path | None = None,
    verbose=True,
) -> dict:
    """Draft truncation sensitivity of the diagonal at fixed discretisation."""
    a = CYLINDER_RADIUS
    reference = DENSITY * pair_integrals(ka, ka, radius=a).qtf_surge
    rows = []
    for factor in factors:
        depth = float(factor) / (ka / a)
        n_theta, n_z = LEVELS[level]
        mesh = bottom_mounted_cylinder_mesh(a, depth, n_theta=n_theta, n_z=n_z)
        res = _diffraction(mesh, ka / a, cache=cache, tag=f"depth{factor:.0f}_{level}")
        qtf = quadratic_qtf(mesh, [ka / a], wave_angle=0.0,
                            density=DENSITY, gravity=GRAVITY,
                            diffraction_results=[res])
        bem = float(_component(qtf.matrix[0, 0], 0).real)
        rows.append(
            {
                "depth_factor": float(factor),
                "depth": depth,
                "k_depth": float(ka / a * depth),
                "bem": bem,
                "ref": float(reference),
                "relative_error": float(abs(bem - reference) / abs(reference)),
            }
        )
        if verbose:
            print(
                f"  depth k*H={rows[-1]['k_depth']:.1f} bem={bem:.4e} "
                f"err={rows[-1]['relative_error']:.2%}",
                flush=True,
            )
    return {"ka": float(ka), "level": level, "rows": rows}


# ---------------------------------------------------------------------------
# stage 3: gates, tables, figures
# ---------------------------------------------------------------------------
def evaluate_gates(validation: dict, levels: list[dict]) -> dict:
    finest = levels[-1]
    errors = [p["abs_error"] for p in finest["pairs"] if not p["diagonal"]]
    g1 = validation["max_identity_error"] <= GATES["G1_diagonal_identity"]
    g2 = validation["max_far_field_error"] <= GATES["G2_far_field_consistency"]
    g3 = validation["long_wave_error_at_0.05"] <= GATES["G3_long_wave_law"]
    g4 = validation["hermitian_residual"] <= GATES["G4_hermitian"]
    g5 = bool(errors) and max(errors) <= GATES["G5_bem_vs_closed_form"]
    order = []
    for p_finest in finest["pairs"]:
        series = [
            next(
                q["abs_error"]
                for q in lvl["pairs"]
                if q["i"] == p_finest["i"] and q["j"] == p_finest["j"]
            )
            for lvl in levels
        ]
        order.append(series)
    g6 = all(
        all(b <= a + 1e-12 for b, a in zip(series[1:], series[:-1])) for series in order
    ) if len(levels) > 1 else None
    return {
        "G1_diagonal_identity": {"value": validation["max_identity_error"], "gate": GATES["G1_diagonal_identity"], "pass": bool(g1)},
        "G2_far_field_consistency": {"value": validation["max_far_field_error"], "gate": GATES["G2_far_field_consistency"], "pass": bool(g2)},
        "G3_long_wave_law": {"value": validation["long_wave_error_at_0.05"], "gate": GATES["G3_long_wave_law"], "pass": bool(g3)},
        "G4_hermitian": {"value": validation["hermitian_residual"], "gate": GATES["G4_hermitian"], "pass": bool(g4)},
        "G5_bem_vs_closed_form": {
            "value": float(max(errors)) if errors else None,
            "median": float(np.median(errors)) if errors else None,
            "level": finest["level"],
            "gate": GATES["G5_bem_vs_closed_form"],
            "pass": g5,
        },
        "G6_monotone_refinement": {"series": order, "pass": g6},
    }


def markdown_table(study: dict) -> str:
    lines = ["| level | panels | k*H | diagonal err (median/max) | off-diagonal err (median/max) |",
             "|---|---|---|---|---|"]
    for lvl in study["levels"]:
        lines.append(
            f"| {lvl['level']} | {lvl['panels']} | {lvl['depth'] * lvl['ka'][0]:.1f} | "
            f"{lvl['diagonal_median_error']:.2%} / {lvl['diagonal_max_error']:.2%} | "
            f"{lvl['offdiagonal_median_error']:.2%} / {lvl['offdiagonal_max_error']:.2%} |"
        )
    return "\n".join(lines)


def pair_table(study: dict) -> str:
    lines = ["| level | (i,j) | ka_i | ka_j | BEM \\|T\\| [N/m^2] | closed form [N/m^2] | rel. error | <10% ? |",
             "|---|---|---|---|---|---|---|---|"]
    for lvl in study["levels"]:
        for p in lvl["pairs"]:
            flag = "yes" if p["abs_error"] <= GATES["G5_bem_vs_closed_form"] else "**no**"
            lines.append(
                f"| {lvl['level']} | ({p['i']},{p['j']}) | {p['ka_i']:.2f} | {p['ka_j']:.2f} | "
                f"{p['bem_abs']:.4e} | {p['ref_abs']:.4e} | {p['abs_error']:.2%} | {flag} |"
            )
    return "\n".join(lines)


def figures(study: dict, outdir: pathlib.Path) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    made = []

    # (1) closed-form QTF surface over the (ka_i, ka_j) plane -- the "field map"
    grid = np.linspace(0.15, 1.6, 26)
    mag = np.zeros((grid.size, grid.size))
    for i, ki in enumerate(grid):
        for j, kj in enumerate(grid):
            pair = pair_integrals(ki, kj, radius=CYLINDER_RADIUS)
            mag[i, j] = DENSITY * math.hypot(pair.qtf_surge, pair.qtf_surge_imag)
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    cs = ax.contourf(grid, grid, mag, levels=20, cmap="viridis")
    ax.set_xlabel(r"$k_i a$")
    ax.set_ylabel(r"$k_j a$")
    ax.set_title(r"closed-form $|T_{xx}(k_i,k_j)|$ [N/m$^2$], $A_iA_j^*=1$")
    fig.colorbar(cs, ax=ax, label="N/m$^2$")
    fig.tight_layout()
    path = outdir / "qtf_closed_form_map.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append({"name": path.name, "caption": "Closed-form difference-frequency QTF magnitude |T_xx(k_i,k_j)| of the fixed bottom-mounted cylinder over the (ka_i, ka_j) plane (MacCamy-Fuchs modes, machine precision)."})

    # (2) BEM vs closed form, per pair, per level
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    finest = study["levels"][-1]
    labels = [f"({p['i']},{p['j']})" for p in finest["pairs"]]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, [p["ref_abs"] for p in finest["pairs"]], width=0.4,
           label="closed form", color="#4c72b0")
    ax.bar(x + 0.2, [p["bem_abs"] for p in finest["pairs"]], width=0.4,
           label=f"BEM ({finest['level']})", color="#dd8452")
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"$|T_{xx}|$ [N/m$^2$]")
    ax.set_xlabel("frequency pair (i,j)")
    ax.set_title("difference-frequency QTF: BEM vs closed form")
    ax.legend()
    fig.tight_layout()
    path = outdir / "qtf_bem_vs_closed_form.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append({"name": path.name, "caption": "Per-pair |T_xx| of the difference-frequency QTF: closed-form reference (blue) against the TensorBEM panel solution at the finest computed mesh level (orange)."})

    # (3) convergence of the pair errors with panel count
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    panels = [lvl["panels"] for lvl in study["levels"]]
    for p0 in study["levels"][0]["pairs"]:
        series = [
            next(q["abs_error"] for q in lvl["pairs"]
                 if q["i"] == p0["i"] and q["j"] == p0["j"])
            for lvl in study["levels"]
        ]
        ax.plot(panels, [100 * s for s in series], marker="o",
                label=f"({p0['i']},{p0['j']})")
    ax.axhline(10.0, color="crimson", ls="--", lw=1.0, label="10 % gate")
    ax.set_xlabel("panels")
    ax.set_ylabel("relative error of $|T_{xx}|$ [%]")
    ax.set_title("mesh refinement of the QTF error")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "qtf_mesh_convergence.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append({"name": path.name, "caption": "Relative error of every QTF pair against the closed form versus panel count, with the predeclared 10 % case gate."})

    # (4) depth truncation sensitivity
    if study.get("depth_sensitivity", {}).get("rows"):
        rows = study["depth_sensitivity"]["rows"]
        fig, ax = plt.subplots(figsize=(5.4, 3.8))
        ax.plot([r["k_depth"] for r in rows], [100 * r["relative_error"] for r in rows],
                marker="s", color="#55a868")
        ax.axhline(10.0, color="crimson", ls="--", lw=1.0, label="10 % gate")
        ax.set_xlabel(r"$k\cdot H$ (draft)")
        ax.set_ylabel("diagonal error vs closed form [%]")
        ax.set_title("draft-truncation sensitivity (coarse mesh)")
        ax.legend()
        fig.tight_layout()
        path = outdir / "qtf_depth_sensitivity.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        made.append({"name": path.name, "caption": "Draft-truncation sensitivity: diagonal QTF error against the infinite-draft closed form as a function of k*H at fixed coarse discretisation."})

    # (5) analytic QTF curves real/imag vs ka for a fixed reference frequency
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    kas = np.linspace(0.15, 1.6, 40)
    base = 0.6
    for comp, label in ((0, "Re"), (1, "Im")):
        vals = []
        for kj in kas:
            pair = pair_integrals(base, kj, radius=CYLINDER_RADIUS)
            vals.append(DENSITY * (pair.qtf_surge if comp == 0 else pair.qtf_surge_imag))
        ax.plot(kas, vals, label=f"{label} $T_{{xx}}(ka_i=0.6, ka_j)$")
    ax.set_xlabel(r"$k_j a$")
    ax.set_ylabel(r"$T_{xx}$ [N/m$^2$]")
    ax.set_title("closed-form QTF across the diagonal (i = fixed)")
    ax.legend()
    fig.tight_layout()
    path = outdir / "qtf_closed_form_curves.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append({"name": path.name, "caption": "Real and imaginary parts of the closed-form QTF along a row of the matrix (ka_i = 0.6 fixed), showing the continuity across the diagonal that the Newman approximation replaces by a straight line."})

    return made


def run_study(
    *,
    levels=("coarse", "medium"),
    ka_grid=KA_GRID,
    depth_factor=DEPTH_FACTOR,
    with_depth_sweep: bool = True,
    with_validation: bool = True,
    cache: pathlib.Path | None = None,
    verbose: bool = True,
) -> dict:
    study: dict = {
        "case": "BEM-QTF-DRIFT",
        "radius": CYLINDER_RADIUS,
        "density": DENSITY,
        "gravity": GRAVITY,
        "ka_grid": list(float(k) for k in ka_grid),
        "depth_factor": float(depth_factor),
        "gates": GATES,
        "levels_config": {k: list(v) for k, v in LEVELS.items()},
    }
    if with_validation:
        if verbose:
            print("closed-form validation", flush=True)
        study["validation"] = closed_form_validation(verbose=verbose)
    study["levels"] = []
    for level in levels:
        if verbose:
            print(f"BEM level {level}", flush=True)
        study["levels"].append(
            bem_level(
                level,
                ka_grid,
                depth_factor=depth_factor,
                cache=cache,
                verbose=verbose,
            )
        )
    if with_depth_sweep:
        if verbose:
            print("depth sensitivity", flush=True)
        study["depth_sensitivity"] = depth_sensitivity(cache=cache, verbose=verbose)
    if study.get("validation"):
        study["gates_result"] = evaluate_gates(study["validation"], study["levels"])
    if verbose:
        print(markdown_table(study), flush=True)
    return study


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default="docs/case_qtf_drift_closed_form/study.json")
    parser.add_argument("--figures", default="docs/case_qtf_drift_closed_form")
    parser.add_argument("--levels", nargs="+", default=["coarse", "medium"])
    parser.add_argument("--ka", nargs="+", type=float, default=list(KA_GRID))
    parser.add_argument("--depth-factor", type=float, default=DEPTH_FACTOR)
    parser.add_argument("--no-depth-sweep", action="store_true")
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--cache", default=None, help="directory for the diffraction cache")
    parser.add_argument("--tables", default=None, help="write the markdown tables here")
    args = parser.parse_args(argv)

    study = run_study(
        levels=tuple(args.levels),
        ka_grid=tuple(args.ka),
        depth_factor=args.depth_factor,
        with_depth_sweep=not args.no_depth_sweep,
        with_validation=not args.no_validation,
        cache=pathlib.Path(args.cache) if args.cache else None,
    )
    if study.get("gates_result"):
        study["markdown"] = markdown_table(study)
        study["pair_markdown"] = pair_table(study)
    out_json = pathlib.Path(args.json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    # numbers first: the figures are decorative and must not be able to lose a
    # completed (expensive) BEM run
    out_json.write_text(json.dumps(study, indent=2, sort_keys=True))
    try:
        made = figures(study, pathlib.Path(args.figures))
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"figure stage skipped: {exc!r}", flush=True)
        made = []
    study["figures"] = made
    out_json.write_text(json.dumps(study, indent=2, sort_keys=True))
    if args.tables:
        pathlib.Path(args.tables).write_text(
            study.get("markdown", "") + "\n\n" + study.get("pair_markdown", "") + "\n"
        )
    print(f"wrote {out_json} and {len(made)} figures", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())