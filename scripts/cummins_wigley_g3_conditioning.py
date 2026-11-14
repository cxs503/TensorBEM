"""Where does the G3 (Kramers-Kronig) residual actually come from?

Why this script exists
----------------------
``CASE-BEM-CUMMINS-WIGLEY`` has failed G3 on every round since round 2, at
40-82 % on several DOFs, and each round explained part of it in prose:

* round 2: "band / scale effect" (normalisation by ``max|A - A_inf|``, band
  truncated at the lowest tabulated frequency);
* round 3: the *sampling* story (``kk_band_floor``), which removed the
  piecewise-linear model from the estimator's input by shape-preserving
  re-expression -- but without extended-band solves it could not separate band
  from discretisation;
* round 4: a synthetic arm that reported a 45 % residual "floor" which survived
  350x refinement, and therefore *withdrew* the sampling explanation.  That
  floor turned out to belong to the round-4 closed-form pair, not to the
  estimator: the pair carried a ``1/pi`` prefactor on ``B`` that makes
  ``(A, B)`` non-causal by ``1 - 2/pi = 36.3 %``
  (``tests/test_cummins_wigley_kk_causality.py::
  test_damped_oscillator_pair_is_kramers_kronig_consistent``).

So the residual has (at least) four candidate sources, and they ask for
different responses:

``sampling``
    the damping peak of a lightly damped heave resonance is ~0.074 rad/s wide;
    a 0.2-0.5 rad/s global grid cannot represent it, and the principal-value
    integral of a mis-represented peak is wrong *at the peak*.  Response: solve
    a resonance cluster (what :func:`cummins_wigley_kk_causality.closure_grid`
    prescribes).
``a_inf``
    G3 normalises by ``max|A - A_inf|`` and takes ``A_inf`` from
    :func:`tensorbem.causality.fit_added_mass_asymptote` (a two-term tail fit).
    The scale is a *difference* of two nearly equal quantities on the
    translation/rotation DOFs, so a 1 % bias in the fit is a large fraction of
    it.  Response: better ``A_inf``, not a better solve.
``band``
    the tails outside ``[w_min, w_max]``.  Response: new solves at both ends.
``physics``
    the panel solution is genuinely non-causal.  Response: solver fix.

This script measures the first three on the *same* datum, with a synthetic arm
where the ground truth is known, so the decomposition procedure itself is
validated before it is applied to panel data:

1. ``predeclared``  -- exactly what G3 does (fitted ``A_inf``, tabulated grid).
2. ``refined``      -- same estimator on the PCHIP re-expression (sampling).
3. ``exact_a_inf``  -- the same, with ``A_inf`` supplied (``a_inf`` component).
4. ``a_inf_scan``   -- residual as a function of a uniform ``A_inf`` bias, which
   makes the *conditioning* of G3 explicit: ``d(residual)/d(A_inf) = 1/scale``
   in the worst case, so the gate needs ``|dA_inf| < 0.1*scale`` to be decidable.
5. ``band``         -- optional: the same estimator on a wider sweep of the same
   DOF (``--extended-npz``), reported as ``band_attributable``.

Nothing here redefines a gate.  G3 keeps its predeclared definition; the numbers
below say which of the four sources the failure belongs to.

Usage::

    PYTHONPATH=src python3 scripts/cummins_wigley_g3_conditioning.py \\
        --sweep-npz docs/m18249_case_bem_cummins_wigley/sweep_coarse.npz \\
        --outdir artifacts/case_bem_cummins_wigley_g3_conditioning

    # or from a sharded point cache (the driver's own format)
    PYTHONPATH=src python3 scripts/cummins_wigley_g3_conditioning.py \\
        --points-dir artifacts/case_bem_cummins_wigley_closure/points --level coarse
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from tensorbem import causality
from tensorbem.benchmarks import cummins_wigley_kk_causality as case
from tensorbem.benchmarks import kk_band_floor

DOF_LABELS = case.DOF_LABELS


# ---------------------------------------------------------------------------
# data access
# ---------------------------------------------------------------------------


def load_npz(path: pathlib.Path | str) -> dict:
    """Load a sweep ``.npz`` written by earlier rounds (omega, A, B)."""
    data = np.load(path, allow_pickle=True)
    return {
        "label": f"npz:{pathlib.Path(path).name}",
        "omega": np.asarray(data["omega"], dtype=float),
        "added_mass": np.asarray(data["added_mass"], dtype=float),
        "damping": np.asarray(data["damping"], dtype=float),
    }


def load_points(directory: pathlib.Path | str, level: str) -> dict:
    """Load the full sorted point cache of one mesh level (driver format)."""
    points = case.load_points(directory, level)
    if not points:
        raise SystemExit(f"no cached points for level {level!r} in {directory!r}")
    omega = np.asarray(sorted(points), dtype=float)
    sweep = case.sweep_from_points(points, omega)
    return {
        "label": f"points:{level}:n={omega.size}",
        "omega": omega,
        "added_mass": np.asarray(sweep["added_mass"], dtype=float),
        "damping": np.asarray(sweep["damping"], dtype=float),
    }


# ---------------------------------------------------------------------------
# the decomposition
# ---------------------------------------------------------------------------


def _diag(sweep: dict, dof: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(sweep["omega"], dtype=float),
        np.asarray(sweep["added_mass"], dtype=float)[:, dof, dof],
        np.asarray(sweep["damping"], dtype=float)[:, dof, dof],
    )


def implied_a_inf(
    omega: np.ndarray, added: np.ndarray, damping: np.ndarray
) -> dict:
    """The KK-implied ``A_inf`` profile (see :class:`tensorbem.causality.AInfProfile`).

    Eq. (2) of :mod:`tensorbem.causality` is **affine in ``A_inf`` with unit
    slope**, so the G3 residual

    ``max_i |A_pred(w_i; a) - A_i| / scale = max_i |a - e_i| / scale``,
    with ``e_i = A_i - q_i`` and ``q_i`` the principal-value integral of ``B``,

    is a max of absolute affine functions: convex, piecewise linear, and minimised
    in closed form by the **midrange** of ``e``
    (:func:`tensorbem.causality.profile_added_mass_inf`).  Hence

    * the profile residual is the part that no choice of ``A_inf`` can remove --
      the spread of the implied ``A_inf`` along the band, i.e. the measured
      causality inconsistency of the tabulated pair;
    * the installed ``A_inf`` (two-term tail fit, what G3 uses) enters as a pure
      translation: a bias of ``f * scale`` adds ``|f|``.  The scan below measures
      that slope, which is 1.00 for every DOF of the panel sweep.
    """
    profile = causality.profile_added_mass_inf(omega, added, damping)
    return {
        "omega_mid": profile.sample_omegas,
        "implied_a_inf": profile.implied_added_mass_inf,
        "best_a_inf": profile.best_added_mass_inf,
        "best_residual": profile.residual,
        "scale": profile.scale,
        "installed_a_inf": profile.installed_added_mass_inf,
        "installed_residual": profile.installed_residual,
        "spread": float(np.max(profile.implied_added_mass_inf)
                        - np.min(profile.implied_added_mass_inf)),
    }


def a_inf_scan(
    omega: np.ndarray,
    added: np.ndarray,
    damping: np.ndarray,
    *,
    centre: float,
    scale: float,
    fractions=(-1.0, -0.5, -0.2, -0.1, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02,
               0.05, 0.1, 0.2, 0.5, 1.0),
) -> list[dict]:
    """G3 residual as a function of a uniform ``A_inf`` bias, in units of ``scale``.

    Measured, not derived: the predicted ``A`` shifts by exactly the bias, so the
    slope of this curve against ``fraction`` should be 1.0 for every DOF.  That
    prediction is what :func:`implied_a_inf` relies on, and the slope column of
    the report is the check.
    """
    rows = []
    for fraction in fractions:
        residual = kk_band_floor.kk_residual(
            omega, added, damping,
            added_mass_inf=float(centre) + float(fraction) * float(scale),
            scale=float(scale),
        )
        rows.append({"fraction_of_scale": float(fraction), "max": residual["max"]})
    return rows


def dof_report(
    sweep: dict,
    dof: int,
    *,
    refined_points: int = 601,
    extended: dict | None = None,
) -> dict:
    """The decomposition of G3 for one DOF."""
    omega, added, damping = _diag(sweep, dof)
    predeclared = kk_band_floor.kk_residual(omega, added, damping)
    a_inf = float(predeclared["added_mass_inf"])
    scale = float(predeclared["scale"])
    implied = implied_a_inf(omega, added, damping)

    dense_omega, dense_a, dense_b = kk_band_floor.refine_sweep(
        omega, added, damping, points=refined_points)
    refined = kk_band_floor.kk_residual(
        dense_omega, dense_a, dense_b, added_mass_inf=a_inf, scale=scale)
    implied_refined = implied_a_inf(dense_omega, dense_a, dense_b)

    bias = (a_inf - implied["best_a_inf"]) / scale
    report = {
        "dof": dof,
        "dof_label": DOF_LABELS[dof],
        "band": [float(omega[0]), float(omega[-1])],
        "points": int(omega.size),
        "predeclared": {"max": predeclared["max"], "rms": predeclared["rms"],
                        "argmax_omega": predeclared["argmax_omega"]},
        "refined": {"max": refined["max"], "rms": refined["rms"],
                    "argmax_omega": refined["argmax_omega"]},
        "added_mass_inf": a_inf,
        "scale": scale,
        "best_a_inf": implied["best_a_inf"],
        "best_a_inf_shift": implied["best_a_inf"] - a_inf,
        "best_a_inf_shift_fraction_of_a_inf": (
            (implied["best_a_inf"] - a_inf) / a_inf if a_inf else float("nan")),
        "calibrated": {"max": implied["best_residual"],
                       "refined_max": implied_refined["best_residual"]},
        "calibration_component": abs(bias),
        "a_inf_scan": a_inf_scan(omega, added, damping, centre=a_inf, scale=scale),
    }
    scan = report["a_inf_scan"]
    lo = next(row for row in scan if row["fraction_of_scale"] == -0.01)
    hi = next(row for row in scan if row["fraction_of_scale"] == 0.01)
    report["a_inf_sensitivity"] = (hi["max"] - lo["max"]) / 0.02
    # the two readings must agree: predeclared = |bias| + calibrated (up to the
    # sign of the bias, since the residual is a max of absolute values)
    report["closure_error"] = float(
        abs(predeclared["max"] - (abs(bias) + implied["best_residual"])))
    if extended is not None:
        ext_omega, ext_a, ext_b = _diag(extended, dof)
        ext = kk_band_floor.kk_residual(
            ext_omega, ext_a, ext_b, added_mass_inf=a_inf, scale=scale)
        ext_implied = implied_a_inf(ext_omega, ext_a, ext_b)
        report["extended"] = {
            "band": [float(ext_omega[0]), float(ext_omega[-1])],
            "points": int(ext_omega.size),
            "max": ext["max"], "rms": ext["rms"],
            "argmax_omega": ext["argmax_omega"],
            "calibrated_max": ext_implied["best_residual"],
            "best_a_inf": ext_implied["best_a_inf"],
        }
        report["band_attributable"] = float(
            max(implied_refined["best_residual"] - ext_implied["best_residual"], 0.0))
    return report


def synthetic_control() -> dict:
    """The same decomposition on the closed-form pair, where ``A_inf`` is known.

    This is the validation of the *procedure*: on a datum whose causal partner is
    analytic, a residual that survives refinement can only be an ``A_inf`` bias,
    and the script below should show the two contributions adding up.
    """
    sigma, omega_n = 0.037, 7.378504598323515
    grid = np.asarray(case.closure_grid(omega_n), dtype=float)
    added, damping = kk_band_floor.damped_oscillator_pair(
        grid, sigma=sigma, omega_n=omega_n)
    sweep = {
        "label": f"synthetic:closure({grid.size})",
        "omega": grid,
        "added_mass": np.tile(added.reshape(-1, 1, 1), (1, 6, 6)),
        "damping": np.tile(damping.reshape(-1, 1, 1), (1, 6, 6)),
    }
    report = dof_report(sweep, 2, refined_points=601)
    report["a_inf_true"] = 0.0
    report["exact_a_inf"] = kk_band_floor.kk_residual(
        grid, added, damping, added_mass_inf=0.0,
        scale=float(np.max(np.abs(added))))["max"]
    # the procedure's own validation: with the truth known, the *calibrated*
    # residual must reproduce the exact-A_inf residual, and the shift the
    # minimiser asks for must recover A_inf = 0
    report["procedure_error"] = float(
        abs(report["calibrated"]["max"] - report["exact_a_inf"]))
    report["a_inf_bias_over_scale"] = report["added_mass_inf"] / report["scale"]
    report["best_a_inf_relative_to_true"] = float(
        report["best_a_inf"] - report["a_inf_true"])
    return report


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def _figure(rows: list[dict], synthetic: dict, path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    labels = [row["dof_label"] for row in rows]
    x = np.arange(len(rows))
    width = 0.27
    axes[0].bar(x - width, [row["predeclared"]["max"] for row in rows], width,
                label="predeclared G3 (fitted $A_\\infty$)")
    axes[0].bar(x, [row["refined"]["max"] for row in rows], width,
                label="refined (sampling removed)")
    axes[0].bar(x + width, [row["calibrated"]["max"] for row in rows], width,
                label="calibrated ($A_\\infty$ = exact Chebyshev centre)")
    axes[0].axhline(case.GATES["G3_kramers_kronig"]["tol"], color="k", ls="--",
                    lw=1.0, label="G3 tolerance 10 %")
    axes[0].set_xticks(x, labels, rotation=30)
    axes[0].set_ylabel("KK residual (max, /max$|A-A_\\infty|$)")
    axes[0].set_title("panel data: three readings of the same datum")
    axes[0].legend(fontsize=7)

    scan = np.array([[row["fraction_of_scale"] for row in synthetic["a_inf_scan"]],
                     [row["max"] for row in synthetic["a_inf_scan"]]])
    axes[1].plot(scan[0], scan[1], "o-", label="synthetic pair, fitted $A_\\infty$")
    axes[1].axhline(case.GATES["G3_kramers_kronig"]["tol"], color="k", ls="--", lw=1.0)
    axes[1].axhline(synthetic["exact_a_inf"], color="C3", ls=":",
                    label=f"exact $A_\\infty$: {synthetic['exact_a_inf']:.2e}")
    axes[1].axvline(synthetic["a_inf_bias_over_scale"], color="C2", ls="-.",
                    label=f"tail-fit bias {synthetic['a_inf_bias_over_scale']:+.3f}")
    axes[1].set_xlabel("uniform $A_\\infty$ bias / max$|A-A_\\infty|$")
    axes[1].set_ylabel("KK residual (max)")
    axes[1].set_title("synthetic control: residual vs $A_\\infty$ bias")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sweep-npz", help="a sweep .npz with omega/added_mass/damping")
    source.add_argument("--points-dir", help="driver point cache directory")
    parser.add_argument("--level", default="coarse", help="mesh level for --points-dir")
    parser.add_argument("--extended-npz", default=None,
                        help="optional wider sweep of the *same* mesh, for band_attributable")
    parser.add_argument("--extended-points-dir", default=None,
                        help="optional point cache of a wider sweep")
    parser.add_argument("--outdir", default="artifacts/case_bem_cummins_wigley_g3_conditioning")
    parser.add_argument("--out", default="g3_conditioning.json")
    args = parser.parse_args(argv)

    if args.sweep_npz:
        sweep = load_npz(args.sweep_npz)
    else:
        sweep = load_points(args.points_dir, args.level)
    extended = None
    if args.extended_npz:
        extended = load_npz(args.extended_npz)
    elif args.extended_points_dir:
        extended = load_points(args.extended_points_dir, args.level)

    started = time.time()
    synthetic = synthetic_control()
    rows = [dof_report(sweep, dof, extended=extended) for dof in range(6)]

    payload = {
        "case": "CASE-BEM-CUMMINS-WIGLEY",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": "which component of the G3 residual is which",
        "source": sweep["label"],
        "omega": np.asarray(sweep["omega"], dtype=float).tolist(),
        "gate": case.GATES["G3_kramers_kronig"],
        "synthetic_control": synthetic,
        "dofs": rows,
        "wall_seconds": time.time() - started,
    }
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / args.out
    path.write_text(json.dumps(payload, indent=1))

    print(f"[g3] source        {sweep['label']}")
    print(f"[g3] synthetic     predeclared={synthetic['predeclared']['max']:.4f} "
          f"exact-A_inf={synthetic['exact_a_inf']:.4f} "
          f"A_inf bias/scale={synthetic['a_inf_bias_over_scale']:+.4f}")
    print(f"[g3] procedure     calibrated={synthetic['calibrated']['max']:.4f} "
          f"exact-A_inf={synthetic['exact_a_inf']:.4f} "
          f"err={synthetic['procedure_error']:.2e} "
          f"A_inf*={synthetic['best_a_inf']:+.4f} (truth 0)")
    for row in rows:
        line = (f"[g3] {row['dof_label']:5s} predeclared={row['predeclared']['max']:.4f} "
                f"refined={row['refined']['max']:.4f} "
                f"calibrated={row['calibrated']['max']:.4f} "
                f"(A_inf {row['added_mass_inf']:.1f}->{row['best_a_inf']:.1f}, "
                f"shift {100*row['best_a_inf_shift_fraction_of_a_inf']:+.1f}%, "
                f"slope {row['a_inf_sensitivity']:.2f}, "
                f"closure {row['closure_error']:.1e})")
        if "band_attributable" in row:
            line += f" band_attr={row['band_attributable']:.4f}"
        print(line)
    figure = outdir / "fig_g3_conditioning.png"
    _figure(rows, synthetic, figure)
    print(f"[g3] -> {path}")
    print(f"[g3] -> {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())