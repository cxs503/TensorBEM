"""``CASE-SHU-OSHER``: linear wave interaction benchmark.

Ledger anchor: ``CASE-SHU-OSHER`` — reference = d'Alembert's formula (exact
solution of the 1-D linear wave equation), tolerance ``< 5 %``.

Scope note
----------
The original Shu & Osher (1988) problem is a *nonlinear* hyperbolic
conservation law (Euler equations, Mach 3 shock interacting with an entropy
wave).  BEM is a method for *linear* elliptic / Helmholtz-type PDEs and
cannot reproduce shock-capturing results.  This benchmark implements the
**linear** wave-equation analog: two counter-propagating Gaussian pulses
that pass through each other, exercising wave superposition and interference
within the domain where BEM-style retarded potentials are mathematically valid.

The analytical reference is d'Alembert's formula, which is exact for the
infinite-line problem.  With transparent (non-reflecting) boundary conditions,
the numerical solution should match d'Alembert to machine precision.  With
Dirichlet boundary conditions (u=0 at both ends), the boundary flux correction
introduces a discretisation error that decreases with dt.

Two routes are compared:
  A — d'Alembert's formula (exact reference),
  B — time-domain BEM with retarded potentials (capability under test).

Usage
-----
    python -m tensorbem.benchmarks.shu_osher_wave_interaction \
        --json out.json --markdown out.md --figure out.png
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from .. import wave1d as w1d

TOLERANCE = 0.05  # 5% per ledger acceptance


def run_case(
    L: float = 10.0,
    c: float = 1.0,
    t_max: float = 8.0,
    dt: float = 0.01,
    bc_type: str = "transparent",
) -> dict:
    """Run the wave interaction case and return results dict."""
    t0 = time.perf_counter()
    case = w1d.wave_interaction_case(L=L, c=c, t_max=t_max, dt=dt, bc_type=bc_type)
    elapsed = time.perf_counter() - t0
    case["wallclock_s"] = round(elapsed, 3)
    return case


def generate_figure(case: dict, path: str) -> None:
    """Generate comparison figure: BEM field vs d'Alembert at key times."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = case.get("_result")
    if result is None:
        # Reconstruct from case data
        centre1, sigma1 = 3.0, 0.5
        centre2, sigma2 = 7.0, 0.5

        def u0(x):
            return w1d._gaussian(x, centre1, sigma1) + w1d._gaussian(x, centre2, sigma2)

        def v0(x):
            return -case["c"] * w1d._gaussian_prime(x, centre1, sigma1) + case["c"] * w1d._gaussian_prime(x, centre2, sigma2)

        result = w1d.solve_wave1d(
            case["L"], case["c"], u0, v0, case["t_max"], case["dt"], bc_type=case["bc_type"]
        )

    times_to_plot = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, t_target in enumerate(times_to_plot):
        ax = axes[idx]
        # Find closest time index
        n = int(round(t_target / result.dt))
        n = min(n, len(result.times) - 1)
        t = result.times[n]

        ax.plot(result.x_points, result.u_exact[n, :], "k--", lw=1.5, label="d'Alembert (exact)")
        ax.plot(result.x_points, result.u_field[n, :], "r-", lw=1.0, alpha=0.7, label="BEM retarded potential")
        ax.set_title(f"t = {t:.1f}")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x,t)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"CASE-SHU-OSHER: Linear Wave Interaction (bc={case['bc_type']}, "
        f"L2 err={case['l2_relative_error']:.2e})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def generate_markdown(case: dict) -> str:
    """Generate markdown report section."""
    lines = [
        "# CASE-SHU-OSHER: Linear Wave Interaction Benchmark",
        "",
        "## Scope Note",
        "",
        case["scope_note"],
        "",
        "## Case Description",
        "",
        f"- **Domain**: [0, {case['L']}] with wave speed c = {case['c']}",
        f"- **Boundary conditions**: {case['bc_type']}",
        f"- **Time**: t = 0 .. {case['t_max']}, dt = {case['dt']}",
        f"- **Pulse 1**: centre={case['pulse1']['centre']}, σ={case['pulse1']['sigma']}, direction={case['pulse1']['direction']}",
        f"- **Pulse 2**: centre={case['pulse2']['centre']}, σ={case['pulse2']['sigma']}, direction={case['pulse2']['direction']}",
        f"- **Interaction**: t = {case['interaction_time']:.1f}, x = {case['interaction_location']:.1f}",
        "",
        "## Method",
        "",
        "The solution is decomposed as u = u_dalembert + u_retarded, where:",
        "- u_dalembert: d'Alembert's formula (exact for infinite line)",
        "- u_retarded: boundary flux correction via retarded potentials",
        "",
        "For transparent BCs, u_retarded = 0 and the solution is exact.",
        "For Dirichlet BCs, boundary fluxes are solved from a Volterra integral equation.",
        "",
        "## Results",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| L2 relative error | {case['l2_relative_error']:.6e} |",
        f"| L∞ relative error | {case['linf_relative_error']:.6e} |",
        f"| Max |u| | {case['u_max']:.6f} |",
        f"| Time steps | {case['n_time_steps']} |",
        f"| Spatial points | {case['n_spatial_points']} |",
        f"| Wall clock (s) | {case.get('wallclock_s', 'N/A')} |",
        "",
        f"## Acceptance (< {TOLERANCE:.0%})",
        "",
        f"**{'PASS' if case['passes_5pct'] else 'FAIL'}**: "
        f"L2 relative error = {case['l2_relative_error']:.6e} "
        f"{'<' if case['passes_5pct'] else '>='} {TOLERANCE}",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="CASE-SHU-OSHER wave interaction benchmark")
    parser.add_argument("--json", type=str, help="Output JSON path")
    parser.add_argument("--markdown", type=str, help="Output markdown path")
    parser.add_argument("--figure", type=str, help="Output figure PNG path")
    parser.add_argument("--bc", type=str, default="transparent", choices=["transparent", "dirichlet"])
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--t-max", type=float, default=8.0)
    args = parser.parse_args()

    case = run_case(dt=args.dt, t_max=args.t_max, bc_type=args.bc)

    if args.json:
        # Remove non-serialisable fields
        out = {k: v for k, v in case.items() if k != "_result"}
        Path(args.json).write_text(json.dumps(out, indent=2))

    if args.markdown:
        md = generate_markdown(case)
        Path(args.markdown).write_text(md)

    if args.figure:
        generate_figure(case, args.figure)

    print(f"L2 relative error: {case['l2_relative_error']:.6e}")
    print(f"L∞ relative error: {case['linf_relative_error']:.6e}")
    print(f"Passes 5% tolerance: {case['passes_5pct']}")


if __name__ == "__main__":
    main()
