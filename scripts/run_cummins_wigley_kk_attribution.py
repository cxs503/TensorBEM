"""Driver: attribute the declared Cummins / Kramers-Kronig residual (anchor
``CASE-BEM-CUMMINS-WIGLEY``) to an added-mass asymptote bias versus a genuine
shape (causality) error, and calibrate the estimator on exactly causal pairs.

Stages
------
``attribute``  load a radiation sweep (cached BEM points, or an archived
               ``results.json``) and decompose the declared G3 residual per DOF.
``calibrate``  run the estimator chain on exactly causal pairs: the Ursell
               closed-form cylinder (physics-faithful ``B(0) = 0``, ``A_inf = 0``
               known exactly) and a damped-oscillator surrogate fitted to the
               measured damping; plus the grid-refinement floor study.
``figures``    matplotlib panels for the report (from the JSON of the two stages).

Every number written to ``attribution.json`` is computed here; the report copies
nothing by hand.

Example::

    PYTHONPATH=src python3 scripts/run_cummins_wigley_kk_attribution.py \\
        --from-results docs/board_audit/m17340/subject_results.json \\
        --levels coarse,medium --outdir docs/case_bem_cummins_wigley_attribution
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from tensorbem import causality_attribution as ca
from tensorbem.analytic import ursell_cylinder_added_mass
from tensorbem.benchmarks import cummins_wigley_kk_causality as case

DOF_LABELS = case.DOF_LABELS


def load_sweep(path: pathlib.Path, level: str) -> dict:
    """Diagonal radiation sweep: from cached points if present, else archived."""
    points_dir = path / "points"
    if points_dir.is_dir():
        points = case.load_points(points_dir, level)
        cached = [w for w in case.OMEGA_GRID if round(float(w), 9) in points]
        if len(cached) >= 8:
            sweep = case.sweep_from_points(points, cached)
            return {
                "omega": np.asarray(sweep["omega"], dtype=float),
                "added_mass": np.asarray(sweep["added_mass"], dtype=float),
                "damping": np.asarray(sweep["damping"], dtype=float),
                "provenance": {"source": str(points_dir), "kind": "cached_points",
                               "nodes": len(cached)},
            }
    payload = json.loads(path.read_text())
    entry = payload["levels"][level]["sweep"]
    return {
        "omega": np.asarray(entry["omega"], dtype=float),
        "added_mass": np.diag(np.asarray(entry["added_mass_diag"], dtype=float)[0]).size
        and np.asarray(entry["added_mass_diag"], dtype=float),
        "damping": np.asarray(entry["damping_diag"], dtype=float),
        "provenance": {
            "source": str(path), "kind": "archived_results_json",
            "nodes": len(entry["omega"]),
            "cached_omega": payload["levels"][level].get("cached_omega"),
        },
    }


URSELL_RADIUS = 0.09375
URSELL_SUBMERGENCE = 0.35
URSELL_DENSITY = 1025.0


def ursell_asymptote(radius: float = URSELL_RADIUS,
                     density: float = URSELL_DENSITY) -> float:
    """High-frequency added mass of the deeply submerged cylinder.

    In the ``omega -> inf`` limit the free surface leaves the body alone, so the
    added mass is the infinite-fluid value ``rho*pi*a^2`` per unit span -- known
    in closed form, which is what makes the Ursell pair a calibrated control
    rather than another fit.
    """
    import math as _math

    return float(density * _math.pi * radius * radius)


def ursell_pair(omega: np.ndarray, *, radius: float = URSELL_RADIUS,
                submergence: float = URSELL_SUBMERGENCE,
                density: float = URSELL_DENSITY, gravity: float = 9.80665):
    """Exactly causal reference pair on the case's own frequency grid.

    Ursell's deeply submerged cylinder: ``B(omega) -> 0`` as ``omega -> 0`` (the
    physical low-frequency behaviour the Wigley sweep also has) and the
    infinite-frequency added mass is exactly zero (the free-surface condition
    degenerates to ``phi = 0`` and the body sees no free surface), so ``A_inf``
    is *known*, not fitted.
    """
    added = np.zeros_like(omega)
    damping = np.zeros_like(omega)
    for i, w in enumerate(omega):
        a, b = ursell_cylinder_added_mass(
            radius, submergence, float(w), density, gravity=gravity)
        added[i], damping[i] = a, b
    return added, damping


def calibrate(omega: np.ndarray, damping: np.ndarray, *, real_added: np.ndarray | None,
              label: str) -> dict:
    """Estimator floor on exactly causal pairs, on this very frequency grid."""
    out: dict = {"label": label, "grid": {"n": int(omega.size),
                                          "omega": omega.tolist()},
                 "pairs": {}}
    # (1) Ursell cylinder: analytic, B(0) = 0, A_inf = 0 exactly.  This is the
    #     physics-faithful estimator floor: the tube is deeply submerged, its
    #     damping vanishes at zero frequency exactly as a floating body's does.
    try:
        urs_added, urs_damping = ursell_pair(omega)
        a_inf_true = ursell_asymptote()
        entry = {
            "parameters": {"radius": URSELL_RADIUS,
                           "submergence": URSELL_SUBMERGENCE,
                           "density": URSELL_DENSITY,
                           "added_mass_inf_true": a_inf_true,
                           "added_mass_inf_source": "rho*pi*a^2 (infinite-fluid limit)"},
            "added_mass_range": [float(urs_added.min()), float(urs_added.max())],
            "damping_range": [float(urs_damping.min()), float(urs_damping.max())],
            "below_band_model_sensitivity": {},
        }
        for model in ("constant", "zero", "linear"):
            oracle = ca.kkr_decomposition(omega, urs_added, urs_damping,
                                          added_mass_inf=a_inf_true,
                                          below_band_model=model)
            entry["below_band_model_sensitivity"][model] = {
                "residual_total": float(oracle.residual_total),
                "residual_shape": float(oracle.residual_shape),
            }
        oracle = ca.kkr_decomposition(omega, urs_added, urs_damping,
                                      added_mass_inf=a_inf_true)
        fitted = ca.kkr_decomposition(omega, urs_added, urs_damping)
        entry["oracle_asymptote"] = {"residual_total": float(oracle.residual_total),
                                     "residual_shape": float(oracle.residual_shape)}
        entry["tail_fit_asymptote"] = {
            "added_mass_inf_fitted": float(fitted.added_mass_inf_used),
            "residual_total": float(fitted.residual_total),
            "residual_shape": float(fitted.residual_shape),
            "offset_bias": float(fitted.offset_bias),
        }
        entry["grid_refinement"] = calibrate_grid(
            rate=0.4, freq=1.0, amplitude=1.0, band=(float(omega[0]), float(omega[-1])),
            added_mass_inf=a_inf_true)
        out["pairs"]["ursell_cylinder"] = entry
    except Exception as exc:  # honest record, never a silent gap
        out["pairs"]["ursell_cylinder"] = {
            "error": f"{type(exc).__name__}: {exc}"}

    # (2) Damped-oscillator surrogate fitted to the measured damping.  Its
    #     kernel has non-zero mean (B(0) > 0), which a floating body's does not,
    #     so it is *not* a floor: it is a control on how much the below-band
    #     closure matters when the low-frequency behaviour is unphysical.
    if real_added is not None:
        decomposition = ca.kkr_decomposition(omega, real_added, damping)
        null = ca.null_calibration(
            omega, damping, added_mass_inf=float(decomposition.added_mass_inf_implied))
        null["note"] = ("oscillator family has B(0) > 0: upper bound on closure "
                        "sensitivity, not an estimator floor")
        out["pairs"]["oscillator_surrogate"] = null
    return out


def calibrate_grid(*, band: tuple[float, float], added_mass_inf: float,
                   node_counts=(22, 44, 88, 176)) -> dict:
    """Estimator floor versus frequency-grid resolution, on the Ursell pair."""
    rows = []
    for count in node_counts:
        grid = np.geomspace(band[0], band[1], int(count))
        added, damping = ursell_pair(grid)
        decomposition = ca.kkr_decomposition(grid, added, damping,
                                            added_mass_inf=added_mass_inf)
        fitted = ca.kkr_decomposition(grid, added, damping)
        rows.append({
            "nodes": int(count),
            "shape_residual_oracle": float(decomposition.residual_shape),
            "total_residual_tail_fit": float(fitted.residual_total),
            "added_mass_inf_fitted": float(fitted.added_mass_inf_used),
        })
    return {"pair": "ursell_cylinder", "band": [float(band[0]), float(band[1])],
            "added_mass_inf_true": float(added_mass_inf), "rows": rows}


def run_attribute(args) -> dict:
    path = pathlib.Path(args.from_results)
    results = {
        "case": "CASE-BEM-CUMMINS-WIGLEY",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": str(path),
        "levels": {},
    }
    for level in [v.strip() for v in args.levels.split(",") if v.strip()]:
        sweep = load_sweep(path, level)
        omega = sweep["omega"]
        added = sweep["added_mass"]
        damping = sweep["damping"]
        entry = {"provenance": sweep["provenance"], "per_dof": {}}
        for dof, name in enumerate(DOF_LABELS):
            a = added[:, dof] if added.ndim == 2 else added[:, dof, dof]
            b = damping[:, dof] if damping.ndim == 2 else damping[:, dof, dof]
            decomposition = ca.kkr_decomposition(omega, a, b)
            sum_rule = ca.sum_rule_asymptote(omega, a, b)
            negative = ca.kkr_decomposition(
                omega, 2.0 * decomposition.added_mass_inf_used - a, b)
            entry["per_dof"][name] = {
                "decomposition": decomposition.as_dict(),
                "sum_rule": sum_rule,
                "negative_control_sign_flip": {
                    "residual_total": float(negative.residual_total),
                    "residual_shape": float(negative.residual_shape),
                },
            }
        entry["calibration"] = calibrate(
            omega, damping[:, 2], real_added=added[:, 2], label=f"{level}:heave")
        results["levels"][level] = entry
        print(f"[attribute] level={level} nodes={omega.size} "
              f"worst_total={max(v['decomposition']['residual_total'] for v in entry['per_dof'].values()):.4f}",
              flush=True)
    return results


def make_figures(results: dict, outdir: pathlib.Path) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    figures = []
    levels = list(results["levels"])

    # Figure 1: declared residual split into offset and shape, per DOF per level.
    fig, axes = plt.subplots(1, len(levels), figsize=(5.2 * len(levels), 4.2),
                             squeeze=False)
    for ax, level in zip(axes[0], levels):
        per = results["levels"][level]["per_dof"]
        x = np.arange(len(DOF_LABELS))
        total = [per[n]["decomposition"]["residual_total"] * 100 for n in DOF_LABELS]
        shape = [per[n]["decomposition"]["residual_shape"] * 100 for n in DOF_LABELS]
        offset = [per[n]["decomposition"]["residual_offset"] * 100 for n in DOF_LABELS]
        ax.bar(x - 0.27, total, 0.27, label="declared G3 residual")
        ax.bar(x, shape, 0.27, label="shape (causality) part")
        ax.bar(x + 0.27, offset, 0.27, label="offset (A_inf) part")
        ax.axhline(10.0, color="k", ls="--", lw=1, label="gate tolerance 10 %")
        ax.set_xticks(x)
        ax.set_xticklabels(DOF_LABELS, rotation=45)
        ax.set_ylabel("residual / max|A - A_inf|  [%]")
        ax.set_title(f"{level}: {len(results['levels'][level]['per_dof'])} DOF")
        ax.legend(fontsize=7)
    fig.suptitle("KK residual: asymptote offset vs shape (CASE-BEM-CUMMINS-WIGLEY)")
    fig.tight_layout()
    path = outdir / "fig_attribution_bars.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figures.append({"name": path.name,
                    "caption": "Per-DOF split of the declared KK residual into the "
                               "added-mass asymptote offset and the shape residual."})

    # Figure 2: predicted vs measured added mass (heave, coarse/medium).
    fig, axes = plt.subplots(1, len(levels), figsize=(5.2 * len(levels), 4.0),
                             squeeze=False)
    for ax, level in zip(axes[0], levels):
        dec = results["levels"][level]["per_dof"]["heave"]["decomposition"]
        ax.plot(dec["omega_query"], dec["added_mass_measured"], "o-", ms=3,
                label="measured A33")
        ax.plot(dec["omega_query"],
                np.asarray(dec["added_mass_inf_implied"]) + np.asarray(
                    dec["predicted_offsetless"]), "-",
                label="A_inf*(implied) + Hilbert(B)")
        ax.plot(dec["omega_query"],
                np.asarray(dec["added_mass_inf_used"]) + np.asarray(
                    dec["predicted_offsetless"]), "--",
                label="A_inf(tail fit) + Hilbert(B)")
        ax.set_xlabel("omega [rad/s]")
        ax.set_ylabel("A33 [kg]")
        ax.set_title(f"heave, {level}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    path = outdir / "fig_heave_prediction.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figures.append({"name": path.name,
                    "caption": "Heave added mass against the Hilbert transform of the "
                               "measured damping, with the tail-fit and the "
                               "KK-implied asymptote."})

    # Figure 3: calibration floor vs frequency-grid size, plus the real residual.
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for level in levels:
        calib = results["levels"][level]["calibration"]["pairs"]["ursell_cylinder"]
        if "oracle_asymptote" not in calib:
            continue
        ax.axhline(calib["oracle_asymptote"]["residual_shape"] * 100, ls=":",
                   label=f"Ursell exact pair, {level} grid ({calib['grid']['n']} nodes)"
                   if "grid" in calib else None)
        refinement = calib.get("grid_refinement")
        if refinement:
            nodes = [row["nodes"] for row in refinement["rows"]]
            ax.semilogx(nodes, [row["shape_residual_oracle"] * 100
                                for row in refinement["rows"]], "o-",
                        label=f"Ursell exact pair, oracle A_inf ({level} band)")
    ax.axhline(10.0, color="k", ls="--", lw=1, label="gate tolerance 10 %")
    ax.set_xlabel("frequency-grid nodes")
    ax.set_ylabel("shape residual [%]")
    ax.set_title("Estimator floor on exactly causal pairs")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = outdir / "fig_calibration.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figures.append({"name": path.name,
                    "caption": "Shape residual produced by the same estimator chain on "
                               "exactly causal pairs, versus frequency-grid resolution."})
    return figures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-results", required=True,
                        help="directory with points/ or an archived results.json")
    parser.add_argument("--levels", default="coarse,medium")
    parser.add_argument("--outdir", default="docs/case_bem_cummins_wigley_attribution")
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args(argv)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results = run_attribute(args)
    if args.figures:
        results["figures"] = make_figures(results, outdir)
    path = outdir / "attribution.json"
    path.write_text(json.dumps(results, indent=1))
    print(f"[attribute] -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())