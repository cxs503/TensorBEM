"""Reproducible study of the surface-gradient recovery under data error.

``docs/three_dimensional_bem.md`` limitation 9 says the recovery path "depends
on the smoothness of the boundary potential: for real piecewise-constant input
(for example shell pressure data) it must be smoothed first".  That sentence
was never quantified and no preprocessing was implemented.  This study answers
the three questions it leaves open, all against an exact field:

``order``
    The recovery of an *exact* trace, refined over four geodesic meshes.  The
    recovery module's docstring claims second-order accuracy; the measurement
    is first order, so the docstring was corrected rather than the number.
``amplification``
    A panel-to-panel (white) data error of size ``sigma`` turned into a
    gradient error.  Measured on a fixed ``sigma`` and on ``sigma = h_mean``;
    the resulting amplification is ``sigma / h`` times a geometric constant,
    which is what makes raw piecewise-constant input unusable.
``smoothing``
    The area-weighted panel -> vertex -> panel smoother of
    :mod:`tensorbem.panel_smoothing`, at 1, 2 and 3 iterations.  It lowers the
    noise-driven error by a fixed factor and *degrades* exact data, and it does
    not remove the ``sigma / h`` scaling: smoothing is necessary but not
    sufficient.
``correlated``
    The same amplitude as an error that varies smoothly over the surface.  This
    is benign by two orders of magnitude, so the amplification is a property of
    *non-smooth* data error, not of the operator alone.
``indicator``
    :func:`tensorbem.recovery_uncertainty.estimate_recovery_uncertainty`
    predicts the gradient error from the data alone (patch residuals propagated
    through the fit).  The prediction is compared with the measured error for
    every case above; the raw cases agree within a few percent, the smoothed
    cases are under-predicted by a factor of about 3 because smoothing
    correlates the residuals that the estimator reads.

Run: ``python -m tensorbem.benchmarks.recovery_robustness --levels 1 2 3 4``.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np

from tensorbem.geometry3d import TriangleMesh, icosphere
from tensorbem.panel_smoothing import smooth_panel_field
from tensorbem.recovery import recover_tangent_gradient, tangent_basis
from tensorbem.recovery_uncertainty import estimate_recovery_uncertainty

LEVELS = (1, 2, 3, 4)
SMOOTHING_ITERATIONS = (1, 2, 3)
FIXED_SIGMA = 1.0e-2
CORRELATED_AMPLITUDE = 1.0e-2


# --------------------------------------------------------------------------
# exact field: exterior potential of a unit sphere translating along +x
# --------------------------------------------------------------------------
def translation_trace(points: np.ndarray) -> np.ndarray:
    """Boundary potential ``u = -x / (2 r)`` of the translating unit sphere."""
    radius = np.linalg.norm(points, axis=1)
    return -0.5 * points[:, 0] / radius


def translation_gradient(points: np.ndarray) -> np.ndarray:
    """Exact gradient of :func:`translation_trace` (tangential everywhere)."""
    radius = np.linalg.norm(points, axis=1)
    unit_x = np.array([1.0, 0.0, 0.0])
    radial = points / radius[:, None]
    return -0.5 * (
        unit_x[None, :] / radius[:, None]
        - points * points[:, :1] / radius[:, None] ** 3
    ) - 0.5 * 0.0 * radial  # radial term vanishes identically


def correlated_error(points: np.ndarray, amplitude: float) -> np.ndarray:
    """A smooth, non-constant perturbation of the given amplitude."""
    return amplitude * np.cos(3.0 * points[:, 0]) * (1.0 + 0.5 * points[:, 1])


def correlated_error_gradient(points: np.ndarray, amplitude: float) -> np.ndarray:
    """Exact gradient of :func:`correlated_error`."""
    x, y = points[:, 0], points[:, 1]
    factor = 1.0 + 0.5 * y
    d_dx = -3.0 * amplitude * np.sin(3.0 * x) * factor
    d_dy = 0.5 * amplitude * np.cos(3.0 * x) * np.ones_like(x)
    zeros = np.zeros_like(x)
    return np.stack([d_dx, d_dy, zeros], axis=1)


def tangential_reference(
    mesh: TriangleMesh, gradient: np.ndarray
) -> np.ndarray:
    """Project a 3-D gradient onto each panel plane: the recoverable quantity."""
    normals = mesh.normals
    return gradient - np.sum(gradient * normals, axis=1)[:, None] * normals


def _rms(vectors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(vectors**2, axis=1))))


def _relative(computed: np.ndarray, exact: np.ndarray) -> float:
    denominator = _rms(exact)
    if denominator == 0.0:
        return _rms(computed - exact)
    return _rms(computed - exact) / denominator


def _order(coarse: float, fine: float, ratio: float = 2.0) -> float:
    if coarse <= 0.0 or fine <= 0.0:
        return float("nan")
    return math.log(coarse / fine) / math.log(ratio)


def _patch_step(mesh: TriangleMesh) -> float:
    """Ratio of the mean panel size between two icosphere levels (2.0)."""
    return 2.0


# --------------------------------------------------------------------------
# cases
# --------------------------------------------------------------------------
def _case(mesh: TriangleMesh, exact: np.ndarray, values: np.ndarray, iterations: int) -> dict[str, Any]:
    field = smooth_panel_field(mesh, values, iterations) if iterations else values
    recovered = recover_tangent_gradient(mesh, field)
    uncertainty = estimate_recovery_uncertainty(mesh, field)
    error = _relative(recovered, exact)
    return {
        "iterations": int(iterations),
        "absolute_rms_error": _rms(recovered - exact),
        "relative_l2_error": error,
        "predicted_rms": uncertainty.predicted_rms,
        "sigma_hat_rms": uncertainty.sigma_rms,
        "prediction_ratio": (
            _rms(recovered - exact) / uncertainty.predicted_rms
            if uncertainty.predicted_rms > 0.0
            else float("nan")
        ),
    }


def run_study(levels: tuple[int, ...] = LEVELS) -> dict[str, Any]:
    """Return the JSON-serialisable evidence for the study."""
    rng = np.random.default_rng(20261103)
    clean: list[dict[str, Any]] = []
    biased: list[dict[str, Any]] = []
    white_fixed: list[dict[str, Any]] = []
    white_scaled: list[dict[str, Any]] = []
    correlated: list[dict[str, Any]] = []

    for level in levels:
        mesh = icosphere(level, radius=1.0)
        centroids = mesh.centroids
        exact = tangential_reference(mesh, translation_gradient(centroids))
        trace = translation_trace(centroids)

        row = {
            "level": int(level),
            "panels": int(mesh.panel_count),
            "h_mean": float(mesh.h_mean),
            "h_max": float(mesh.h_max),
            "patch_step": _patch_step(mesh),
        }
        clean_row = dict(row)
        clean_row["cases"] = [
            _case(mesh, exact, trace, iterations)
            for iterations in (0,) + SMOOTHING_ITERATIONS
        ]
        clean.append(clean_row)

        noise = rng.standard_normal(mesh.panel_count)
        fixed_row = dict(row)
        fixed_row["sigma"] = FIXED_SIGMA
        noisy = trace + FIXED_SIGMA * noise
        fixed_row["cases"] = [
            _case(mesh, exact, noisy, iterations) for iterations in (0, 1, 2)
        ]
        white_fixed.append(fixed_row)

        scaled_row = dict(row)
        scaled_row["sigma"] = float(mesh.h_mean)
        scaled = trace + mesh.h_mean * noise
        scaled_row["cases"] = [
            _case(mesh, exact, scaled, iterations) for iterations in (0, 1, 2)
        ]
        white_scaled.append(scaled_row)

        perturbed = trace + correlated_error(centroids, CORRELATED_AMPLITUDE)
        correlated_exact = tangential_reference(
            mesh,
            translation_gradient(centroids)
            + correlated_error_gradient(centroids, CORRELATED_AMPLITUDE),
        )
        corr_row = dict(row)
        corr_row["amplitude"] = CORRELATED_AMPLITUDE
        corr_row["cases"] = [
            _case(mesh, correlated_exact, perturbed, iterations)
            for iterations in (0, 1)
        ]
        correlated.append(corr_row)

        # The same smoother applied for its own sake, to show the bias it adds
        # to exact data: this is the "cost" side of the trade-off.
        biased.append(
            {
                "level": int(level),
                "panels": int(mesh.panel_count),
                "h_mean": float(mesh.h_mean),
            }
        )

    # amplification of a fixed sigma, in units of sigma / h_mean
    for row in white_fixed:
        raw = next(c for c in row["cases"] if c["iterations"] == 0)
        smoothed = next(c for c in row["cases"] if c["iterations"] == 1)
        row["amplification_raw"] = raw["absolute_rms_error"] * row["h_mean"] / row["sigma"]
        row["amplification_smoothed"] = (
            smoothed["absolute_rms_error"] * row["h_mean"] / row["sigma"]
        )
        row["smoothing_gain"] = raw["absolute_rms_error"] / smoothed["absolute_rms_error"]

    summary = _summarise(clean, white_fixed, white_scaled, correlated)
    return {
        "benchmark": "recovery_robustness",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "exact_field": "u = -x/(2r), exact tangential gradient -0.5(e_x/r - x x_1/r^3)",
        "levels": [int(level) for level in levels],
        "clean": clean,
        "white_fixed_sigma": white_fixed,
        "white_scaled_sigma": white_scaled,
        "correlated": correlated,
        "summary": summary,
    }


def _slope(rows: list[dict[str, Any]], key: str) -> float:
    """Least-squares slope of log(value) against log(h_mean)."""
    xs = np.log([row["h_mean"] for row in rows])
    ys = np.log([row[key] for row in rows])
    if len(xs) < 2:
        return float("nan")
    return float(np.polyfit(xs, ys, 1)[0])


def _summarise(
    clean: list[dict[str, Any]],
    white_fixed: list[dict[str, Any]],
    white_scaled: list[dict[str, Any]],
    correlated: list[dict[str, Any]],
) -> dict[str, Any]:
    clean_errors = [
        next(c for c in row["cases"] if c["iterations"] == 0)["absolute_rms_error"]
        for row in clean
    ]
    clean_orders = [
        _order(coarse, fine)
        for coarse, fine in zip(clean_errors, clean_errors[1:])
    ]
    clean_smoothed = [
        next(c for c in row["cases"] if c["iterations"] == 1)["absolute_rms_error"]
        for row in clean
    ]
    fixed_raw = [
        next(c for c in row["cases"] if c["iterations"] == 0)["absolute_rms_error"]
        for row in white_fixed
    ]
    fixed_smoothed = [
        next(c for c in row["cases"] if c["iterations"] == 1)["absolute_rms_error"]
        for row in white_fixed
    ]
    fixed_predicted = [
        next(c for c in row["cases"] if c["iterations"] == 0)["predicted_rms"]
        for row in white_fixed
    ]
    corr_raw = [
        next(c for c in row["cases"] if c["iterations"] == 0)["absolute_rms_error"]
        for row in correlated
    ]
    return {
        "clean_errors": clean_errors,
        "clean_observed_orders": clean_orders,
        "clean_error_after_one_smoothing": clean_smoothed,
        "clean_smoothing_penalty": [
            after / before for after, before in zip(clean_smoothed, clean_errors)
        ],
        "white_fixed_sigma_errors": fixed_raw,
        "white_fixed_sigma_errors_smoothed": fixed_smoothed,
        "white_fixed_sigma_predicted": fixed_predicted,
        "white_fixed_sigma_loglog_slope": _slope(white_fixed, "amplification_raw"),
        "white_fixed_sigma_error_slope": float(
            np.polyfit(
                np.log([row["h_mean"] for row in white_fixed]),
                np.log(fixed_raw),
                1,
            )[0]
        ),
        "white_scaled_sigma_errors": [
            next(c for c in row["cases"] if c["iterations"] == 0)["absolute_rms_error"]
            for row in white_scaled
        ],
        "white_scaled_sigma_smoothing_gain": [
            row["cases"][0]["absolute_rms_error"] / row["cases"][1]["absolute_rms_error"]
            for row in white_scaled
        ],
        "correlated_errors": corr_raw,
        "correlated_vs_white_ratio": [
            white / corr for white, corr in zip(fixed_raw, corr_raw)
        ],
        "prediction_ratio_raw": [
            next(c for c in row["cases"] if c["iterations"] == 0)["prediction_ratio"]
            for row in white_fixed
        ],
        "prediction_ratio_smoothed": [
            next(c for c in row["cases"] if c["iterations"] == 1)["prediction_ratio"]
            for row in white_fixed
        ],
    }


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def render_markdown(study: dict[str, Any]) -> str:
    summary = study["summary"]
    lines = [
        "# Recovery robustness: data error, smoothing, and a computable indicator",
        "",
        "Exact field: boundary potential of a unit sphere translating along +x;",
        "the reference is the exact gradient projected on each panel plane.",
        "",
        "## 1. Recovery of an exact trace (order)",
        "",
        "| level | panels | h_mean | absolute RMS error | relative L2 | observed order |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    orders = summary["clean_observed_orders"]
    for position, row in enumerate(study["clean"]):
        case = next(c for c in row["cases"] if c["iterations"] == 0)
        order = f"{orders[position - 1]:.3f}" if position else "--"
        lines.append(
            f"| {row['level']} | {row['panels']} | {row['h_mean']:.4f} | "
            f"{case['absolute_rms_error']:.4e} | {case['relative_l2_error']:.4e} | {order} |"
        )
    lines += [
        "",
        "## 2. White (panel-to-panel) data error",
        "",
        "| level | panels | h_mean | sigma | error, raw | error, 1 smoothing | gain | "
        "amplification, raw (sigma/h) | predicted | actual/predicted |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in study["white_fixed_sigma"]:
        raw, smoothed = row["cases"][0], row["cases"][1]
        lines.append(
            f"| {row['level']} | {row['panels']} | {row['h_mean']:.4f} | {row['sigma']:.1e} | "
            f"{raw['absolute_rms_error']:.3e} | {smoothed['absolute_rms_error']:.3e} | "
            f"{row['smoothing_gain']:.2f} | {row['amplification_raw']:.3f} | "
            f"{raw['predicted_rms']:.3e} | {raw['prediction_ratio']:.2f} |"
        )
    lines += [
        "",
        "## 3. Correlated (smooth) data error of the same amplitude",
        "",
        "| level | panels | amplitude | error, raw | error after smoothing |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in study["correlated"]:
        lines.append(
            f"| {row['level']} | {row['panels']} | {row['amplitude']:.1e} | "
            f"{row['cases'][0]['absolute_rms_error']:.3e} | "
            f"{row['cases'][1]['absolute_rms_error']:.3e} |"
        )
    lines += [
        "",
        "## 4. Summary",
        "",
        f"- observed order of the recovery on exact data: "
        f"{', '.join(f'{value:.2f}' for value in summary['clean_observed_orders'])}",
        f"- slope of log(error) against log(h) for a fixed white sigma: "
        f"{summary['white_fixed_sigma_error_slope']:.3f} (pure 1/h is -1)",
        f"- smoothing gain on white noise: "
        f"{', '.join(format(row['smoothing_gain'], '.2f') for row in study['white_fixed_sigma'])}",
        f"- smoothing penalty on exact data: "
        f"{', '.join(f'{value:.2f}' for value in summary['clean_smoothing_penalty'])}",
        f"- correlated error is smaller than the white error of the same amplitude by: "
        f"{', '.join(f'{value:.1f}' for value in summary['correlated_vs_white_ratio'])}",
        f"- indicator on raw noisy data (actual/predicted): "
        f"{', '.join(f'{value:.2f}' for value in summary['prediction_ratio_raw'])}",
        f"- indicator on smoothed noisy data (actual/predicted): "
        f"{', '.join(f'{value:.2f}' for value in summary['prediction_ratio_smoothed'])}",
    ]
    return "\n".join(lines) + "\n"


def render_figure(study: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    left, right = axes

    clean = study["clean"]
    h = [row["h_mean"] for row in clean]
    clean_error = [
        next(c for c in row["cases"] if c["iterations"] == 0)["absolute_rms_error"]
        for row in clean
    ]
    clean_smoothed = [
        next(c for c in row["cases"] if c["iterations"] == 1)["absolute_rms_error"]
        for row in clean
    ]
    fixed_error = [
        next(c for c in row["cases"] if c["iterations"] == 0)["absolute_rms_error"]
        for row in study["white_fixed_sigma"]
    ]
    fixed_smoothed = [
        next(c for c in row["cases"] if c["iterations"] == 1)["absolute_rms_error"]
        for row in study["white_fixed_sigma"]
    ]
    correlated = [
        next(c for c in row["cases"] if c["iterations"] == 0)["absolute_rms_error"]
        for row in study["correlated"]
    ]

    left.loglog(h, clean_error, "o-", label="exact trace (no data error)")
    left.loglog(h, clean_smoothed, "o--", label="exact trace, 1 smoothing (bias)")
    left.loglog(h, fixed_error, "s-", label=r"white $\sigma=10^{-2}$, raw")
    left.loglog(h, fixed_smoothed, "s--", label=r"white $\sigma=10^{-2}$, smoothed")
    left.loglog(h, correlated, "^-", label=r"correlated $\epsilon=10^{-2}$")
    reference = clean_error[-1] * (h[-1] / np.array(h))
    left.loglog(h, reference, "k:", label=r"slope $-1$")
    left.set_xlabel(r"mean panel size $h$ [-]")
    left.set_ylabel("recovered-gradient RMS error")
    left.set_title("data error, smoothing and refinement")
    left.legend(fontsize=7, loc="best")
    left.grid(True, which="both", alpha=0.3)

    actual = []
    predicted = []
    for row in study["white_fixed_sigma"]:
        case = row["cases"][0]
        actual.append(case["absolute_rms_error"])
        predicted.append(case["predicted_rms"])
    for row in study["clean"]:
        case = row["cases"][0]
        actual.append(case["absolute_rms_error"])
        predicted.append(case["predicted_rms"])
    actual = np.array(actual)
    predicted = np.array(predicted)
    right.loglog(predicted, actual, "o", label="raw / exact data")
    span = np.array([predicted.min() * 0.5, predicted.max() * 2.0])
    right.loglog(span, span, "k-", label="prediction = actual")
    right.loglog(span, 3.0 * span, "k:", label="factor 3")
    right.loglog(span, span / 3.0, "k:", label="factor 1/3")
    right.set_xlabel("predicted RMS error (patch residual propagated)")
    right.set_ylabel("measured RMS error")
    right.set_title("computable indicator vs truth")
    right.legend(fontsize=7, loc="best")
    right.grid(True, which="both", alpha=0.3)

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, nargs="+", default=list(LEVELS))
    parser.add_argument("--output", type=Path, default=Path("outputs/recovery_robustness"))
    arguments = parser.parse_args(argv)
    study = run_study(tuple(arguments.levels))
    arguments.output.mkdir(parents=True, exist_ok=True)
    json_path = arguments.output / "recovery_robustness.json"
    json_path.write_text(json.dumps(study, indent=1) + "\n")
    (arguments.output / "recovery_robustness.md").write_text(render_markdown(study))
    render_figure(study, arguments.output / "recovery_robustness.png")
    print(json.dumps(study["summary"], indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())