"""Elevation geometry of the deep-water deviation ladder: two ladders, one kernel.

Two published documents in this tree report a *different* number for the same
``nu``, the same horizontal separation and (nominally) the same ``h`` ladder of
the finite-depth kernel's deviation from the deep-water kernel:

* ``docs/finite_depth3d_regime.md`` §3 (m3409) —
  ``|G/G_deep - 1|`` = 8.225e-02 at ``h = 2``, 5.900e-03 at ``h = 5``,
  9.334e-05 at ``h = 20``;
* ``docs/finite_depth3d_deepwater_arbitration.md`` §3.3 (m4776) — the same
  entries re-measured as 2.049e-01, 7.225e-02, 1.607e-02, and therefore
  declared a *stale ledger* ("three of the four quoted numbers are off by 1-2
  orders of magnitude on current main").

Both runs are correct; they are not the same geometry.

``tensorbem.benchmarks.finite_depth3d_regime.deep_water_regime`` pins the
elevation at the **absolute** value ``z = zeta = -0.5 m`` for every depth
(``Z_MID``), which equals ``-h/2`` only at ``h = 1 m``. m4776 read the prose
word "mid-depth" as *scaled* mid-depth ``z = zeta = -h/2`` and re-measured on
that geometry. At ``h = 1`` the two definitions coincide — which is exactly why
m4776 saw one rung reproduce "exactly" and three rungs "drift" — and for
``h > 1`` they are different points of the same kernel, so no commit needs to
have moved anything.

This module measures both ladders from the current kernel, reports the two
metrics side by side (raw ``|G/G_deep - 1|`` and, as m4776 recommends for
cancelling points, the deviation normalised by the local Rankine scale), and
states the verdict as data. It is the evidence behind the m6384 erratum in
``docs/finite_depth3d_deepwater_arbitration.md`` §8.

Run as::

    PYTHONPATH=src python3 -m tensorbem.benchmarks.deepwater_ladder_geometry
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tensorbem.finite_depth3d import finite_depth_wavenumber, green_function_3d
from tensorbem.free_surface3d import green_function_3d as deep_water_kernel

#: Wavenumber and horizontal separation of both published ladders.
NU = 50.0
RADIUS = 1.0

#: Depth ladder shared by m3409 and m4776.
DEPTHS = (1.0, 2.0, 5.0, 20.0, 100.0, 800.0)

#: Absolute elevation held fixed by ``finite_depth3d_regime.Z_MID``.
Z_FIXED = -0.5

#: ``|G/G_deep - 1|`` as printed in ``docs/finite_depth3d_regime.md`` §3.
M3409_DOCUMENTED = (
    (1.0, 3.653e-01),
    (2.0, 8.225e-02),
    (5.0, 5.900e-03),
    (20.0, 9.334e-05),
    (100.0, 7.459e-07),
    (800.0, 1.455e-09),
)

#: ``|G/G_deep - 1|`` re-measured in ``docs/finite_depth3d_deepwater_arbitration.md`` §3.3.
M4776_RERUN = (
    (1.0, 3.653e-01),
    (2.0, 2.049e-01),
    (5.0, 7.225e-02),
    (20.0, 1.607e-02),
)

OUTPUT = Path("docs/m6384/deepwater_ladder_geometry.json")
FIGURE = Path("docs/m6384/deepwater_ladder_geometry.png")


def _deviation(depth: float, elevation: str, nu: float = NU, radius: float = RADIUS) -> dict:
    """One ``h`` rung of the ladder at the requested elevation convention."""
    if elevation == "fixed":
        z = Z_FIXED
    elif elevation == "mid-depth":
        z = -0.5 * depth
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown elevation convention {elevation!r}")
    target = np.array([radius, 0.0, z])
    source = np.array([0.0, 0.0, z])
    finite = complex(green_function_3d(target, source, nu, depth))
    deep = complex(deep_water_kernel(target, source, nu))
    # Local Rankine scale 1/(4 pi) (1/r + 1/r1) with r1 the free-surface image.
    r = float(np.linalg.norm(target - source))
    r_image = math.hypot(radius, 2.0 * z)
    rankine = (1.0 / (4.0 * math.pi)) * (1.0 / r + 1.0 / r_image)
    return {
        "depth": depth,
        "k0h": finite_depth_wavenumber(nu, depth) * depth,
        "z": z,
        "z_over_h": z / depth,
        "finite_depth_real": finite.real,
        "finite_depth_imag": finite.imag,
        "deep_water_real": deep.real,
        "deep_water_imag": deep.imag,
        "relative_deviation": abs(finite / deep - 1.0),
        "rankine_scale": rankine,
        "deviation_over_rankine_scale": abs(finite - deep) / rankine,
        "finite": bool(np.isfinite(finite)),
    }


def ladder(elevation: str, depths=DEPTHS) -> list[dict]:
    """A whole ladder for one elevation convention (``fixed`` or ``mid-depth``)."""
    return [_deviation(depth, elevation) for depth in depths]


def _compare(rows, documented) -> dict:
    by_depth = {row["depth"]: row for row in rows}
    worst = 0.0
    table = []
    for depth, quoted in documented:
        measured = by_depth[depth]["relative_deviation"]
        rel = abs(measured - quoted) / quoted
        worst = max(worst, rel)
        table.append({"depth": depth, "quoted": quoted, "measured": measured, "rel_to_quoted": rel})
    return {"worst_relative_vs_quoted": worst, "rows": table, "reproduces": worst < 5.0e-3}


def audit(depths=DEPTHS) -> dict:
    """Measure both ladders, compare each with its own document, explain the gap."""
    fixed = ladder("fixed", depths)
    scaled = ladder("mid-depth", depths)
    coincide = []
    for a, b in zip(fixed, scaled):
        coincide.append(
            {
                "depth": a["depth"],
                "relative_gap_between_ladders": abs(a["relative_deviation"] - b["relative_deviation"])
                / max(a["relative_deviation"], b["relative_deviation"]),
                "same_elevation": abs(a["z"] - b["z"]) < 1.0e-15,
            }
        )
    verdict = {
        "m3409_ladder_reproduces_at_fixed_elevation": _compare(fixed, M3409_DOCUMENTED),
        "m4776_rerun_reproduces_at_scaled_mid_depth": _compare(scaled, M4776_RERUN),
        "ladders_coincide_only_where_z_equals_minus_h_over_2": all(
            row["same_elevation"] == (row["relative_gap_between_ladders"] < 1.0e-12) for row in coincide
        ),
        "coincident_depths": [row["depth"] for row in coincide if row["same_elevation"]],
    }
    return {
        "case": {
            "kernel": "tensorbem.finite_depth3d.green_function_3d",
            "deep_reference": "tensorbem.free_surface3d.green_function_3d",
            "nu": NU,
            "radius": RADIUS,
            "z_fixed_absolute": Z_FIXED,
            "elevation_conventions": {
                "fixed": "z = zeta = -0.5 m absolute at every depth (finite_depth3d_regime.Z_MID)",
                "mid-depth": "z = zeta = -h/2, scaled with the depth (m4776 arbitration reading)",
            },
            "python": __import__("platform").python_version(),
            "numpy": np.__version__,
        },
        "fixed_elevation_ladder": fixed,
        "scaled_mid_depth_ladder": scaled,
        "coincidence": coincide,
        "verdict": verdict,
    }


def figure(payload: dict, path: Path) -> None:
    """Deviation vs ``k0 h`` for both elevation conventions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure_, axes = plt.subplots(figsize=(7.0, 4.6))
    for key, label, style in (
        ("fixed_elevation_ladder", "z = -0.5 m fixed (m3409 ledger)", "o-"),
        ("scaled_mid_depth_ladder", "z = -h/2 scaled (m4776 re-run)", "s--"),
    ):
        rows = payload[key]
        axes.loglog(
            [row["k0h"] for row in rows],
            [row["relative_deviation"] for row in rows],
            style,
            label=label,
        )
    axes.axhline(1.0e-2, color="tab:red", lw=0.8, ls=":", label="1 % band")
    axes.set_xlabel(r"$k_0 h$ (fixed $\nu = 50\ \mathrm{m^{-1}}$, $R = 1\ \mathrm{m}$)")
    axes.set_ylabel(r"$|G/G_{\mathrm{deep}} - 1|$")
    axes.set_title("Same kernel, two elevation conventions: they meet only at $h = 1$ m")
    axes.grid(alpha=0.3, which="both")
    axes.legend(fontsize=8)
    figure_.tight_layout()
    figure_.savefig(path, dpi=140)
    plt.close(figure_)


def main(output: Path = OUTPUT, figure_path: Path | None = FIGURE) -> dict:
    payload = audit()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=_encode) + "\n", encoding="utf-8")
    if figure_path is not None:
        figure(payload, figure_path)
    return payload


def _encode(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, complex):
        return [value.real, value.imag]
    raise TypeError(type(value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    result = main(args.output, None if args.no_figures else FIGURE)
    print(json.dumps(result["verdict"], indent=2))