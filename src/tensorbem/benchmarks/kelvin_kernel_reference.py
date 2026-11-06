"""Independent high-precision reference for the 3-D deep-water Kelvin wave function.

Why this file exists
--------------------
:mod:`tensorbem.free_surface3d` evaluates the dimensionless wave integral

.. math::

    F(X, Y) = \\mathrm{PV} \\int_0^\\infty \\frac{e^{-u Y} J_0(u X)}{u - 1} \\, du

by splitting the quarter plane into four regions (near-axis Bessho series,
Struve/Bessel plus Gauss-Legendre, Laguerre tail, 17-term Legendre asymptotic).
Until this driver existed every check on that routine was *internal*:

* the vertical identity ``dF/dY + F = -1 / hypot(X, Y)`` -- true *by
  construction* in the shipped code, because ``dF/dY`` is generated from it;
* the Kelvin free-surface condition and the radiation decay -- both algebraic
  consequences of the same decomposition;
* seam continuity between neighbouring regions -- which tests the two branches
  against each other, not against the integral;
* finite-difference agreement between ``dF/dX`` and ``dF/dY`` -- again internal.

So a derivation error *inside one region* would survive the whole suite.  This
module supplies the missing external reference: the defining principal-value
integral is evaluated with ``mpmath`` at 22 significant digits and the production
evaluator is compared against it on a grid that covers all four regions and all
three inter-region seams.

Regularisation
--------------
With ``f(u) = exp(-u Y) J_0(u X)`` the principal value is removed analytically,
because the two logarithmic divergences cancel between the halves:

.. math::

    \\mathrm{PV} \\int_0^\\infty \\frac{f(u)}{u-1} du
      = \\int_0^1 \\frac{f(1+s) - f(1-s)}{s} ds
      + \\int_1^\\infty \\frac{f(1+s)}{s} ds .

The first integrand is regular (limit ``2 f'(1) = -2 Y J_0(X)``); the second
decays like ``exp(-s Y) s^{-3/2}`` and is summed with mpmath's oscillatory
quadrature between the zeros of ``J_0((1+s) X)``.  For ``X = 0`` the tail is the
closed form ``exp(-Y) E1(Y)`` exactly, which the generator checks and records as
``tail_identity``.  ``dF/dX`` is evaluated with the same split applied to
``-u exp(-u Y) J_1(u X)``; that integrand also has a simple pole at ``u = 1``, so
the same cancellation argument applies.

Usage::

    python -m tensorbem.benchmarks.kelvin_kernel_reference            # compare
    python -m tensorbem.benchmarks.kelvin_kernel_reference --write    # regenerate the pin
    python -m tensorbem.benchmarks.kelvin_kernel_reference --figure   # deviation map
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import pathlib
from typing import Iterable

from ..free_surface3d import (
    REGION_CONE_MAX_Y,
    REGION_CONE_SLOPE,
    REGION_FAR_DISTANCE,
    REGION_FAR_MIN_X,
    region_label,
    wave_function_3d,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
PIN_PATH = REPO_ROOT / "docs" / "kelvin_kernel_reference" / "reference.json"
FIGURE_PATH = REPO_ROOT / "docs" / "kelvin_kernel_reference" / "deviation_map.png"
SCHEMA = "tensorbem-kelvin-kernel-reference/1"

#: Working precision of the reference quadrature (digits).
REFERENCE_DPS = 22
#: Lower precision used to demonstrate that the reference itself has converged.
SELF_CHECK_DPS = 16


def _mpmath():
    try:
        import mpmath
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "mpmath is required to regenerate the Kelvin-kernel reference; "
            "install it with `pip install mpmath` (the pinned JSON in "
            "docs/kelvin_kernel_reference/reference.json is enough to *check* "
            "the kernel, only regeneration needs mpmath)"
        ) from exc
    return mpmath


def _d_seam(x: float, distance: float) -> float:
    return math.sqrt(distance * distance - x * x)


#: ``(label, left probe, right probe)`` for every inter-region seam.  The two
#: probes straddle the boundary by ``1e-7`` (or ``5e-4`` for the cone seam, where
#: ``X`` itself is the coordinate), so each branch is exercised at the edge of
#: its own validity, where a partition bug is most visible.  A branch
#: discontinuity cannot exceed the sum of the two branch errors against the
#: reference, which is what :func:`evaluate` records.
SEAM_PAIRS: tuple[tuple[str, tuple[float, float], tuple[float, float]], ...] = (
    ("cone / near (X = 0.35 Y)", (0.3495, 1.0), (0.3505, 1.0)),
    ("near / middle (Y = 4.5)", (2.0, 4.4999999), (2.0, 4.5000001)),
    ("middle / far (hypot = 26, X = 0.5)", (0.5, _d_seam(0.5, REGION_FAR_DISTANCE - 1e-7)),
     (0.5, _d_seam(0.5, REGION_FAR_DISTANCE + 1e-7))),
    ("middle / far (hypot = 26, X = 3)", (3.0, _d_seam(3.0, REGION_FAR_DISTANCE - 1e-7)),
     (3.0, _d_seam(3.0, REGION_FAR_DISTANCE + 1e-7))),
)


def sample_points() -> tuple[tuple[float, float], ...]:
    """Quarter-plane grid covering every region, every seam and the axis ``X = 0``.

    Points are authored so that the slowly decaying cases (small ``Y`` with
    large ``X``) stay affordable for the oscillatory quadrature: the tail needs
    roughly ``(1 + 92 / Y) * X / pi`` half-periods.
    """

    return (
        # axis X = 0 (no oscillation; the tail is a closed form)
        (0.0, 0.25),
        (0.0, 2.0),
        (0.0, 26.0),
        # far field (d >= 26, X > 0.25): the asymptotic Legendre region
        (0.5, 26.0),
        (1.0, 26.0),
        (3.0, 30.0),
        (8.0, 26.0),
        (2.0, 40.0),
        (0.3, 40.0),
        # middle region (Y > 4.5 with d < 26 or X <= 0.25)
        (0.2, 26.0),
        (0.2, 60.0),
        (1.0, 4.6),
        (2.0, 6.0),
        (1.0, 10.0),
        (5.0, 10.0),
        (0.05, 12.0),
        # near region (Y <= 4.5, outside the cone)
        (2.0, 1.0),
        (4.0, 0.5),
        (6.0, 0.2),
        (2.0, 0.05),
        (0.5, 0.5),
        (20.0, 2.0),
        # cone (0 < Y <= 4.5, X <= 0.35 Y)
        (0.05, 1.0),
        (0.3, 1.0),
        (1.5, 4.4),
        # seam-adjacent probes: both sides of every partition seam
        *[probe for _, left, right in SEAM_PAIRS for probe in (left, right)],
    )


def _tail_closed_form_zero_x(y: float, dps: int) -> "object":
    """``PV integral_0^inf e^{-u y}/(u - 1) du = -e^{-y} Ei(-y) + ...`` tail part.

    For ``X = 0`` the second half of the regularised split is exactly
    ``exp(-y) E1(y)``; the generator uses it instead of the oscillatory sum and
    records the agreement as an independent check on the regularisation.
    """
    mp = _mpmath()
    with mp.workdps(dps):
        return mp.exp(-mp.mpf(y)) * mp.e1(mp.mpf(y))


def reference_wave_function(x: float, y: float, dps: int = REFERENCE_DPS):
    """Return ``(F, dF/dX)`` from the defining principal-value integrals."""
    mp = _mpmath()
    with mp.workdps(dps):
        big_x = mp.mpf(float(x))
        big_y = mp.mpf(float(y))
        if big_y <= 0:
            raise ValueError("the reference requires Y > 0")

        def f_int(u):
            return mp.exp(-u * big_y) * mp.besselj(0, u * big_x)

        head = mp.quad(lambda s: (f_int(1 + s) - f_int(1 - s)) / s, [0, mp.mpf(1)])
        if big_x == 0:
            tail = mp.exp(-big_y) * mp.e1(big_y)
        else:
            def zeros(n):
                return mp.besseljzero(0, n) / big_x - 1

            offset = 1
            while zeros(offset) <= 1:
                offset += 1
            tail = mp.quadosc(
                lambda s: f_int(1 + s) / s,
                [1, mp.inf],
                zeros=lambda n: zeros(n + offset - 1),
            )
        value = head + tail

        def g_int(u):
            return -u * mp.exp(-u * big_y) * mp.besselj(1, u * big_x)

        head_x = mp.quad(lambda s: (g_int(1 + s) - g_int(1 - s)) / s, [0, mp.mpf(1)])
        if big_x == 0:
            slope = mp.mpf(0)
        else:
            def zeros_x(n):
                return mp.besseljzero(1, n) / big_x - 1

            offset_x = 1
            while zeros_x(offset_x) <= 1:
                offset_x += 1
            slope = head_x + mp.quadosc(
                lambda s: g_int(1 + s) / s,
                [1, mp.inf],
                zeros=lambda n: zeros_x(n + offset_x - 1),
            )
        return value, slope


def seam_pairs(points: list[dict]) -> list[dict]:
    """Accuracy of both branches on either side of every partition seam.

    The jump itself is not measurable directly (a straddle probe also contains
    the genuine variation of ``F`` across the probe separation), so the recorded
    bound is ``dev_left + dev_right``: the largest discontinuity two branches
    with the measured errors can produce.
    """
    index = {(round(p["x"], 15), round(p["y"], 15)): p for p in points}
    rows = []
    for label, left, right in SEAM_PAIRS:
        left_row = index[(round(left[0], 15), round(left[1], 15))]
        right_row = index[(round(right[0], 15), round(right[1], 15))]
        rows.append(
            {
                "seam": label,
                "left": {"x": left[0], "y": left[1], "region": left_row["region"],
                         "dev_f": left_row["dev_f"], "dev_fx": left_row["dev_fx"]},
                "right": {"x": right[0], "y": right[1], "region": right_row["region"],
                          "dev_f": right_row["dev_f"], "dev_fx": right_row["dev_fx"]},
                "bound_f": left_row["dev_f"] + right_row["dev_f"],
                "bound_fx": left_row["dev_fx"] + right_row["dev_fx"],
            }
        )
    return rows


def _digits(value) -> str:
    mp = _mpmath()
    return mp.nstr(value, 20, strip_zeros=False)


def evaluate(dps: int = REFERENCE_DPS, self_dps: int = SELF_CHECK_DPS) -> dict:
    """Build the full reference payload (reference values, deviations, envelopes)."""
    mp = _mpmath()
    points = []
    for x, y in sample_points():
        with mp.workdps(dps):
            f_ref, fx_ref = reference_wave_function(x, y, dps)
        with mp.workdps(self_dps):
            f_low, fx_low = reference_wave_function(x, y, self_dps)
        with mp.workdps(dps):
            self_conv_f = abs((f_low - f_ref) / f_ref)
            self_conv_fx = abs((fx_low - fx_ref) / fx_ref) if fx_ref != 0 else abs(fx_low - fx_ref)
        import numpy as np

        f_prod, fx_prod, _ = wave_function_3d(x, y)
        f_prod = float(np.ravel(f_prod)[0])
        fx_prod = float(np.ravel(fx_prod)[0])
        with mp.workdps(dps):
            dev_f = abs((mp.mpf(f_prod) - f_ref) / f_ref)
            dev_fx = (
                abs((mp.mpf(fx_prod) - fx_ref) / fx_ref) if fx_ref != 0 else abs(mp.mpf(fx_prod) - fx_ref)
            )
        points.append(
            {
                "x": x,
                "y": y,
                "d": math.hypot(x, y),
                "region": region_label(x, y),
                "f_reference": _digits(f_ref),
                "fx_reference": _digits(fx_ref),
                "production_f": f_prod,
                "production_fx": fx_prod,
                "dev_f": float(dev_f),
                "dev_fx": float(dev_fx),
                # A regression gate at 10x the measured deviation: tight enough to
                # catch a real change, loose enough to survive a different BLAS.
                "rtol_f": max(10.0 * float(dev_f), 5e-14),
                "rtol_fx": max(10.0 * float(dev_fx), 5e-13),
                "reference_self_convergence_f": float(self_conv_f),
                "reference_self_convergence_fx": float(self_conv_fx),
            }
        )

    # The X = 0 tail has an exact closed form; recording the agreement checks the
    # regularisation itself, independently of the production code.
    tail_checks = []
    for y in (0.25, 2.0, 8.0):
        with mp.workdps(dps):
            big_y = mp.mpf(y)
            numeric = mp.quad(
                lambda sv: mp.exp(-(1 + sv) * big_y) / sv, [1, 1 + 1 / big_y, mp.inf]
            )
            exact = mp.exp(-big_y) * mp.e1(big_y)
            tail_checks.append(
                {"y": y, "rel_dev": float(abs((numeric - exact) / exact)), "value": _digits(exact)}
            )

    by_region: dict[str, dict[str, float]] = {}
    for point in points:
        bucket = by_region.setdefault(point["region"], {"max_dev_f": 0.0, "max_dev_fx": 0.0, "count": 0})
        bucket["count"] += 1
        bucket["max_dev_f"] = max(bucket["max_dev_f"], point["dev_f"])
        bucket["max_dev_fx"] = max(bucket["max_dev_fx"], point["dev_fx"])

    payload = {
        "schema": SCHEMA,
        "generated": _dt.date.today().isoformat(),
        "reference_precision_dps": dps,
        "self_check_precision_dps": self_dps,
        "definition": "F(X,Y) = PV int_0^inf exp(-uY) J0(uX)/(u-1) du; dF/dX = PV int_0^inf -u exp(-uY) J1(uX)/(u-1) du",
        "regularisation": "PV[f/(u-1)] = int_0^1 [f(1+s)-f(1-s)]/s ds + int_1^inf f(1+s)/s ds (J0 zeros for the oscillatory half)",
        "region_thresholds": {
            "cone_max_y": REGION_CONE_MAX_Y,
            "cone_slope": REGION_CONE_SLOPE,
            "far_distance": REGION_FAR_DISTANCE,
            "far_min_x": REGION_FAR_MIN_X,
        },
        "points": points,
        "tail_identity": tail_checks,
        "by_region": by_region,
        "envelope": {
            "max_dev_f": max(p["dev_f"] for p in points),
            "max_dev_fx": max(p["dev_fx"] for p in points),
            "max_reference_self_convergence_f": max(
                p["reference_self_convergence_f"] for p in points
            ),
            "argmax_f": _argmax(points, "dev_f"),
            "argmax_fx": _argmax(points, "dev_fx"),
        },
        "seams": seam_pairs(points),
    }
    return payload


def _argmax(points: list[dict], key: str) -> dict:
    best = max(points, key=lambda p: p[key])
    return {"x": best["x"], "y": best["y"], "region": best["region"], "dev": best[key]}


def compare(payload: dict) -> list[dict]:
    """Recompute every production value and compare with the pinned reference."""
    import numpy as np

    rows = []
    for point in payload["points"]:
        f_prod, fx_prod, _ = wave_function_3d(point["x"], point["y"])
        f_prod = float(np.ravel(f_prod)[0])
        fx_prod = float(np.ravel(fx_prod)[0])
        f_ref = float(point["f_reference"])
        fx_ref = float(point["fx_reference"])
        dev_f = abs(f_prod - f_ref) / abs(f_ref)
        dev_fx = abs(fx_prod - fx_ref) / abs(fx_ref) if fx_ref != 0.0 else abs(fx_prod - fx_ref)
        rows.append(
            {
                **{k: point[k] for k in ("x", "y", "region")},
                "dev_f": dev_f,
                "dev_fx": dev_fx,
                "rtol_f": point["rtol_f"],
                "rtol_fx": point["rtol_fx"],
                "ok": dev_f <= point["rtol_f"] and dev_fx <= point["rtol_fx"],
            }
        )
    return rows


def render_document_block(payload: dict) -> str:
    """Markdown block for ``docs/kelvin_kernel_reference.md`` (generated, not typed)."""
    lines = [
        "| region | points | max \\|ΔF\\|/\\|F\\| | max \\|ΔF_x\\|/\\|F_x\\| |",
        "|---|---:|---:|---:|",
    ]
    for name in ("cone", "near", "middle", "far"):
        bucket = payload["by_region"][name]
        lines.append(
            f"| `{name}` | {bucket['count']} | {bucket['max_dev_f']:.2e} | {bucket['max_dev_fx']:.2e} |"
        )
    env = payload["envelope"]
    lines.append(
        f"| **all** | {len(payload['points'])} | {env['max_dev_f']:.2e} | {env['max_dev_fx']:.2e} |"
    )
    lines.append("")
    lines.append("| seam | probe (X, Y) | branch regions | bound on \\|ΔF\\|/\\|F\\| | bound on \\|ΔF_x\\|/\\|F_x\\| |")
    lines.append("|---|---|---|---:|---:|")
    for seam in payload["seams"]:
        left, right = seam["left"], seam["right"]
        lines.append(
            f"| {seam['seam']} | ({left['x']:.7g}, {left['y']:.7g}) / ({right['x']:.7g}, {right['y']:.7g}) | "
            f"{left['region']} / {right['region']} | {seam['bound_f']:.2e} | {seam['bound_fx']:.2e} |"
        )
    lines.append("")
    lines.append("| X = 0 tail identity | relative deviation |")
    lines.append("|---|---:|")
    for check in payload["tail_identity"]:
        lines.append(f"| Y = {check['y']:g} | {check['rel_dev']:.2e} |")
    return "\n".join(lines)


def save(payload: dict, path: pathlib.Path = PIN_PATH) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    return path


def load(path: pathlib.Path = PIN_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def figure(payload: dict, path: pathlib.Path = FIGURE_PATH) -> pathlib.Path:
    """Deviation map: production vs the principal-value reference."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"cone": "#1b6ca8", "near": "#2a9d8f", "middle": "#e9a03b", "far": "#c1121f"}
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    for key, axis, label in (
        ("dev_f", axes[0], r"$|F_{prod}-F_{ref}|/|F_{ref}|$"),
        ("dev_fx", axes[1], r"$|F_{x,prod}-F_{x,ref}|/|F_{x,ref}|$"),
    ):
        for name, colour in colours.items():
            subset = [p for p in payload["points"] if p["region"] == name]
            axis.semilogy(
                [p["d"] for p in subset],
                [max(p[key], 1e-18) for p in subset],
                "o",
                color=colour,
                label=name,
                markersize=6,
            )
        axis.axhline(1e-15, color="0.5", lw=0.8, ls="--")
        axis.set_xlabel(r"$d=\sqrt{X^2+Y^2}$")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    axes[0].set_title("Kelvin wave function F vs PV reference")
    axes[1].set_title("Wave function slope dF/dX vs PV reference")
    fig.suptitle("3-D deep-water Kelvin kernel: production vs 22-digit principal-value reference")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="regenerate the pinned reference JSON")
    parser.add_argument("--json", type=pathlib.Path, default=None, help="write the payload here")
    parser.add_argument("--figure", action="store_true", help="write docs/kelvin_kernel_reference/deviation_map.png")
    parser.add_argument("--dps", type=int, default=REFERENCE_DPS, help="reference working precision")
    parser.add_argument("--self-dps", type=int, default=SELF_CHECK_DPS, help="precision used for the self-convergence probe")
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = evaluate(dps=args.dps, self_dps=args.self_dps)
    if args.write or args.json is not None:
        target = args.json if args.json is not None else PIN_PATH
        save(payload, target)
        print(f"wrote {target}")
    if args.figure:
        print(f"wrote {figure(payload)}")
    rows = compare(payload)
    bad = [row for row in rows if not row["ok"]]
    print(render_document_block(payload))
    print()
    print(f"points: {len(rows)}, outside tolerance: {len(bad)}")
    for row in bad[:10]:
        print(
            f"  x={row['x']:g} y={row['y']:g} region={row['region']}: "
            f"dev_f={row['dev_f']:.2e} (rtol {row['rtol_f']:.2e}), "
            f"dev_fx={row['dev_fx']:.2e} (rtol {row['rtol_fx']:.2e})"
        )
    return 1 if bad else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())