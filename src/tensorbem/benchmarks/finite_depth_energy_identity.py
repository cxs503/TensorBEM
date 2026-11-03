"""Finite-depth energy identity: attribution of the far/near damping offset.

:meth:`docs/finite_depth_radiation.md` §5 (limitation 1) recorded an open
verification debt: at 64 panels the finite-depth solver's far-field damping
``B_far = 2*rho*g*c_g*|eta_a|^2`` exceeds the near-field surface-pressure
damping ``B_near = omega*rho*Im(force)`` by about 1.06 %, while the deep-water
solver on the same mesh is about 1.04 % *below* the near field.  The two
offsets have the same magnitude and opposite signs, and the cause was left
unattributed.

This benchmark closes that debt.  It refines the panel mesh and tracks the
ratio ``B_far / B_near`` for both solvers.  The attribution logic is:

* if the offset ``ratio - 1`` decays like ``C / N`` (first order in the panel
  size) and tends to zero, the far-field *constant* is correct and the residual
  is a panel discretisation error - the same kind of error as the ~1.7 %
  added-mass error at 64 panels;
* if the offset tends to a non-zero constant, one of the two far-field
  normalisations carries a systematic factor and must be corrected.

The two solvers extract the far field with different (mathematically
equivalent) formulae - the deep-water Haskind surface integral versus the
finite-depth source-density integral - so a correct pair of formulae is
expected to bracket the truth from opposite sides and converge to the same
limit.  That bracketing is itself evidence that neither formula carries a
hidden constant.

Usage::

    PYTHONPATH=src python3 -m tensorbem.benchmarks.finite_depth_energy_identity \
        --json outputs/finite_depth_energy_identity/study.json \
        --report docs/finite_depth_energy_identity.md \
        --figure outputs/finite_depth_energy_identity/convergence.png
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from tensorbem.benchmarks.finite_depth_radiation import (
    MOTIONS,
    submerged_circle_mesh,
)
from tensorbem.finite_depth_solver import solve_radiation
from tensorbem.free_surface_solver import (
    far_field_damping as deep_far_field_damping,
)
from tensorbem.free_surface_solver import solve_radiation_indirect

GRAVITY = 9.80665
KA = 0.6
RADIUS = 1.0
SUBMERGENCE = 2.0
DENSITY = 1025.0


def _nu() -> float:
    return KA / RADIUS


def finite_depth_ratio(
    depth: float,
    panels: int,
    motion: str = "heave",
    ka: float = KA,
    radius: float = RADIUS,
    submergence: float = SUBMERGENCE,
    density: float = DENSITY,
) -> dict:
    """One finite-depth solve, returning the far/near damping ratio."""
    mesh = submerged_circle_mesh(panels, radius, submergence)
    result, far = solve_radiation(
        depth=depth, mesh=mesh, nu=_nu(), direction=motion,
        density=density, gravity=GRAVITY,
    )
    index = MOTIONS[motion]
    near = float(result.damping[index])
    ratio = float(far) / near
    return {
        "solver": "finite_depth",
        "depth": float(depth),
        "panels": int(panels),
        "motion": motion,
        "B_near": near,
        "B_far": float(far),
        "ratio": ratio,
        "offset": ratio - 1.0,
        "residual": float(result.relative_residual),
    }


def deep_water_ratio(
    panels: int,
    motion: str = "heave",
    ka: float = KA,
    radius: float = RADIUS,
    submergence: float = SUBMERGENCE,
    density: float = DENSITY,
) -> dict:
    """One deep-water solve, returning the far/near damping ratio."""
    mesh = submerged_circle_mesh(panels, radius, submergence)
    result = solve_radiation_indirect(
        mesh=mesh, nu=_nu(), direction=motion, density=density
    )
    index = MOTIONS[motion]
    near = float(result.damping[index])
    far = float(
        deep_far_field_damping(
            mesh, result.potential, result.normal_velocity, _nu(), density
        )
    )
    ratio = far / near
    return {
        "solver": "deep_water",
        "depth": None,
        "panels": int(panels),
        "motion": motion,
        "B_near": near,
        "B_far": far,
        "ratio": ratio,
        "offset": ratio - 1.0,
        "residual": float(result.relative_residual),
    }


def observed_orders(offsets, panels) -> list[float]:
    """Observed convergence order of ``offset`` against the panel count.

    ``order_i = log2(|off_i| / |off_{i+1}|) / log2(N_{i+1} / N_i)``; a value of
    1.0 means the offset halves for every doubling of the panel count.
    """
    orders = []
    for i in range(len(offsets) - 1):
        h_ratio = math.log2(panels[i + 1] / panels[i])
        o_ratio = math.log2(abs(offsets[i]) / abs(offsets[i + 1]))
        orders.append(o_ratio / h_ratio)
    return orders


def run(
    refinement_depth: float = 5.0,
    refinement_panels: tuple[int, ...] = (64, 128, 256, 512),
    depth_sweep: tuple[float, ...] = (3.2, 10.0, 50.0),
    sweep_panels: tuple[int, ...] = (64, 128),
    deep_panels: tuple[int, ...] = (64, 128, 256),
    motion: str = "heave",
) -> dict:
    """Run the refinement ladder, the depth sweep and the deep-water ladder."""
    cases: list[dict] = []
    started = time.time()
    for panels in refinement_panels:
        cases.append(finite_depth_ratio(refinement_depth, panels, motion))
    for depth in depth_sweep:
        for panels in sweep_panels:
            cases.append(finite_depth_ratio(depth, panels, motion))
    for panels in deep_panels:
        cases.append(deep_water_ratio(panels, motion))

    fd_ref = [c for c in cases
              if c["solver"] == "finite_depth" and c["depth"] == refinement_depth]
    fd_ref.sort(key=lambda c: c["panels"])
    fd_orders = observed_orders(
        [c["offset"] for c in fd_ref], [c["panels"] for c in fd_ref]
    )
    dw = [c for c in cases if c["solver"] == "deep_water"]
    dw.sort(key=lambda c: c["panels"])
    dw_orders = observed_orders([c["offset"] for c in dw], [c["panels"] for c in dw])

    return {
        "benchmark": "finite_depth_energy_identity",
        "ka": KA,
        "radius": RADIUS,
        "submergence": SUBMERGENCE,
        "density": DENSITY,
        "gravity": GRAVITY,
        "motion": motion,
        "refinement_depth": refinement_depth,
        "cases": cases,
        "finite_depth_refinement_orders": fd_orders,
        "deep_water_orders": dw_orders,
        "elapsed_s": time.time() - started,
    }


def _markdown(payload: dict) -> str:
    lines = [
        "# Finite-depth energy identity: far/near damping offset attribution",
        "",
        f"Generated by `tensorbem.benchmarks.finite_depth_energy_identity` "
        f"({payload['elapsed_s']:.0f} s).  All numbers below are emitted by the "
        "CLI; nothing is transcribed by hand.",
        "",
        f"Geometry: circle radius {payload['radius']}, axis at "
        f"-{payload['submergence']}, ka = {payload['ka']}, "
        f"rho = {payload['density']}, g = {payload['gravity']}, "
        f"motion = {payload['motion']}.",
        "",
        "## Panel refinement (finite depth, h/a = "
        f"{payload['refinement_depth']})",
        "",
        "| N | B_near | B_far | B_far/B_near | offset | residual |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for c in [x for x in payload["cases"]
              if x["solver"] == "finite_depth"
              and x["depth"] == payload["refinement_depth"]]:
        lines.append(
            f"| {c['panels']} | {c['B_near']:.2f} | {c['B_far']:.2f} | "
            f"{c['ratio']:.6f} | {c['offset']:+.6f} | {c['residual']:.1e} |"
        )
    lines += [
        "",
        "Observed convergence order of the offset: "
        + " / ".join(f"{o:.2f}" for o in payload["finite_depth_refinement_orders"]),
        "",
        "## Depth sweep (finite depth)",
        "",
        "| h/a | N | B_far/B_near | offset |",
        "|---:|---:|---:|---:|",
    ]
    for c in [x for x in payload["cases"]
              if x["solver"] == "finite_depth" and x["depth"] != payload["refinement_depth"]]:
        lines.append(
            f"| {c['depth']} | {c['panels']} | {c['ratio']:.6f} | {c['offset']:+.6f} |"
        )
    lines += [
        "",
        "## Deep-water comparison (same mesh)",
        "",
        "| N | B_near | B_far | B_far/B_near | offset |",
        "|---:|---:|---:|---:|---:|",
    ]
    for c in [x for x in payload["cases"] if x["solver"] == "deep_water"]:
        lines.append(
            f"| {c['panels']} | {c['B_near']:.2f} | {c['B_far']:.2f} | "
            f"{c['ratio']:.6f} | {c['offset']:+.6f} |"
        )
    lines += [
        "",
        "Observed convergence order of the deep-water offset: "
        + " / ".join(f"{o:.2f}" for o in payload["deep_water_orders"]),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--figure", type=Path, default=None)
    parser.add_argument("--panels", type=int, nargs="*",
                        default=[64, 128, 256, 512])
    parser.add_argument("--depths", type=float, nargs="*",
                        default=[3.2, 10.0, 50.0])
    args = parser.parse_args(argv)

    payload = run(
        refinement_panels=tuple(args.panels),
        depth_sweep=tuple(args.depths),
    )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(_markdown(payload))
    if args.figure is not None:
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        _figure(payload, args.figure)
    print(json.dumps({
            "finite_depth_refinement_orders": payload["finite_depth_refinement_orders"],
            "deep_water_orders": payload["deep_water_orders"],
        }, indent=2))
    return 0


def _figure(payload: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fd = [c for c in payload["cases"]
            if c["solver"] == "finite_depth"
            and c["depth"] == payload["refinement_depth"]]
    fd.sort(key=lambda c: c["panels"])
    dw = [c for c in payload["cases"] if c["solver"] == "deep_water"]
    dw.sort(key=lambda c: c["panels"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.semilogy([c["panels"] for c in fd], [abs(c["offset"]) for c in fd],
                "o-", label="finite depth (h/a=%.1f)" % payload["refinement_depth"])
    ax.semilogy([c["panels"] for c in dw], [abs(c["offset"]) for c in dw],
                "s--", label="deep water")
    ref = fd[0]["offset"]
    ax.semilogy([fd[0]["panels"], fd[-1]["panels"]],
                 [abs(ref), abs(ref) * (fd[-1]["panels"] / fd[0]["panels"]) ** -1],
                 "k:", label="O(1/N) reference")
    ax.set_xlabel("panel count N")
    ax.set_ylabel("|B_far/B_near - 1|")
    ax.set_title("Energy-identity offset vs mesh")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[1]
    for c in fd:
        ax.plot(c["panels"], c["ratio"], "o-",
                label="finite depth" if c["depth"] == payload["refinement_depth"]
                else None)
    for c in dw:
        ax.plot(c["panels"], c["ratio"], "s--", label="deep water")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlabel("panel count N")
    ax.set_ylabel("B_far / B_near")
    ax.set_title("Far/near damping ratio")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
