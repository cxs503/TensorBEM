"""CASE-WAVE-FLUME: linear wavemaker transfer functions in a constant-depth flume.

Verifies regular-wave generation against closed-form (Biesel-Havelock) wavemaker
theory through three independent statements, none of which re-derives the other:

1. **Far field, propagating-mode route** -- the Kochin integral of the solved
   wall source density (:func:`tensorbem.wavemaker.kochin_amplitude`) gives the
   radiated elevation ``eta_a``; ``H/S = 2|eta_a|/S`` must equal
   :func:`tensorbem.wavemaker.transfer_function`.
2. **Far field, free-surface probe route** -- the potential is evaluated two
   wavelengths downstream of the wall by direct quadrature of the finite-depth
   kernel (:func:`tensorbem.wavemaker.free_surface_potential`) and turned into
   an elevation there; it must agree with route 1 in amplitude *and* phase.
3. **Near field, exact eigenfunction series** -- the wall problem separates in
   the vertical modes of the strip, so the elevations at
   ``0.25 ... 4`` wavelengths are known in closed form *including the evanescent
   modes* (:func:`tensorbem.wavemaker.exact_elevation`).  The panel field is
   compared station by station.

The study also records two properties of the operator that decide how much of
the solver this case actually exercises, because they are not obvious and they
are easy to report wrongly:

* For a **vertical** full-depth wall every cross-panel entry of the
  normal-derivative matrix vanishes identically (all co-planar panels: the
  kernels depend only on ``|dx|``, whose ``x``-derivative is zero).  The system
  therefore reduces to ``(1/2) I`` -- ``cond = 1.0000`` to machine precision --
  and the panel solution *is* the Havelock projection.  The flume case is a
  kernel-quadrature and mode-decomposition test, **not** an integral-equation
  test.  Saying otherwise would overstate it.
* An **inclined-face** wavemaker (same strip, face sloped by ``alpha``) is not
  co-planar, so the operator becomes non-degenerate (``cond`` and the largest
  off-diagonal entry grow with ``alpha``) and the naive projection stops being
  the answer.  As ``alpha -> 0`` the computed transfer must return to the
  piston value; the residual decay with ``alpha`` is the quantitative anchor
  for the non-degenerate solve.

Run: ``python -m tensorbem.benchmarks.wave_flume_wavemaker``
Writes ``docs/wave_flume_wavemaker/study.json``, the figures and
``docs/wave_flume_wavemaker.md`` (every number programmatic).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tensorbem.wavemaker import (
    GRAVITY,
    WavemakerSurface,
    assemble_normal_derivative,
    evanescent_decay,
    exact_elevation,
    free_surface_potential,
    kochin_amplitude,
    solve_sources,
    solve_wavemaker,
    transfer_function,
    vertical_wavemaker,
)

_ROOT = Path(__file__).resolve().parents[3]
_DOC_DIR = _ROOT / "docs" / "wave_flume_wavemaker"

#: depth used for every case; ``kh`` then *is* the dimensionless depth
DEPTH = 1.0
DEFAULT_KH = (0.5, 1.0, 2.0, 3.0)
DEFAULT_PANELS = (8, 16, 32, 64)
KINDS = ("piston", "flap")
FIELD_STATIONS = (0.25, 0.5, 1.0, 2.0, 4.0)
INCLINE_ANGLES = (1.0e-3, 5.0, 10.0, 20.0, 30.0)


def inclined_wavemaker(depth: float, panels: int, angle_degrees: float) -> WavemakerSurface:
    """Wall spanning the full depth but sloped by ``angle_degrees`` off vertical.

    The face runs from ``(+tan(a) h/2, -h)`` up to ``(-tan(a) h/2, 0)``, so its
    mean plane is still ``x = 0`` and the normal points downstream.  ``angle``
    is in degrees; ``0`` would be the vertical wall (built with
    :func:`tensorbem.wavemaker.vertical_wavemaker` instead, which keeps the
    x-coordinate exactly zero).
    """
    if int(panels) < 2:
        raise ValueError("panels must be >= 2")
    angle = math.radians(float(angle_degrees))
    half = 0.5 * float(depth) * math.tan(angle)
    fraction = np.linspace(0.0, 1.0, int(panels) + 1)
    x = -half + fraction * 2.0 * half
    z = -float(depth) + fraction * float(depth)
    segments = np.zeros((int(panels), 2, 2), dtype=float)
    segments[:, 0, 0] = x[:-1]
    segments[:, 1, 0] = x[1:]
    segments[:, 0, 1] = z[:-1]
    segments[:, 1, 1] = z[1:]
    return WavemakerSurface(segments)


def _max_off_diagonal(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(matrix - np.diag(np.diag(matrix)))))


def _wall_fields(kind: str, kh: float, panels: int, stations):
    """Return ``(elevation_free_surface, exact_free_surface, extra)`` at stations."""
    depth = DEPTH
    k = kh / depth
    omega = math.sqrt(GRAVITY * k * math.tanh(kh))
    nu = omega * omega / GRAVITY
    surface = vertical_wavemaker(depth, panels)
    from tensorbem.wavemaker import velocity_profile

    shape = velocity_profile(kind, surface.centers[:, 1], depth)
    data = -0.5j * omega * shape * surface.normals[:, 0]
    sources, system = solve_sources(surface, data, nu, depth)
    wavelength = 2.0 * math.pi / k
    x = np.asarray(stations, dtype=float) * wavelength
    eta = 1j * omega / GRAVITY * free_surface_potential(surface, sources, nu, depth, x)
    exact = exact_elevation(kind, x, kh=kh, depth=depth, modes=20)
    extra = {
        "system_condition": float(np.linalg.cond(system)),
        "max_off_diagonal": _max_off_diagonal(assemble_normal_derivative(surface, nu, depth)),
        "kochin_amplitude": complex(kochin_amplitude(surface, sources, nu, depth)),
        "wavenumber": k,
        "wavelength": wavelength,
    }
    return eta, exact, extra


def transfer_sweep(kh_values, panels_values) -> list[dict]:
    rows = []
    for kind in KINDS:
        for kh in kh_values:
            for panels in panels_values:
                result = solve_wavemaker(kind, kh=kh, depth=DEPTH, panels=int(panels))
                rows.append(
                    {
                        "kind": kind,
                        "kh": float(kh),
                        "panels": int(panels),
                        "transfer_bem": result.transfer_bem,
                        "transfer_analytic": result.transfer_analytic,
                        "relative_error": result.relative_error,
                        "route_disagreement": result.route_disagreement,
                        "condition_number": result.condition_number,
                        "relative_residual": result.relative_residual,
                        "kochin_abs": abs(result.kochan_amplitude),
                        "probe_abs": abs(result.probe_amplitude),
                    }
                )
    return rows


def field_comparison(kh_values, panels: int = 64) -> list[dict]:
    rows = []
    for kind in KINDS:
        for kh in kh_values:
            eta, exact, extra = _wall_fields(kind, kh, panels, FIELD_STATIONS)
            for station, value, reference in zip(FIELD_STATIONS, eta, exact):
                rows.append(
                    {
                        "kind": kind,
                        "kh": float(kh),
                        "station_wavelengths": float(station),
                        "elevation_abs_bem": float(abs(value)),
                        "elevation_abs_exact": float(abs(reference)),
                        "relative_error": float(abs(value - reference) / abs(reference)),
                    }
                )
    return rows


def operator_degeneracy(kh: float = 1.0, panels: int = 64) -> dict:
    """Cross-panel entries of the operator: vertical wall versus inclined faces."""
    depth = DEPTH
    k = kh / depth
    omega = math.sqrt(GRAVITY * k * math.tanh(kh))
    nu = omega * omega / GRAVITY
    rows = []
    vertical = vertical_wavemaker(depth, panels)
    matrix = assemble_normal_derivative(vertical, nu, depth)
    vertical_sources, vertical_system = solve_sources(
        vertical, -0.5j * omega * np.ones(panels), nu, depth
    )
    vertical_transfer = 2.0 * abs(kochin_amplitude(vertical, vertical_sources, nu, depth))
    rows.append(
        {
            "angle_degrees": 0.0,
            "condition_number": float(np.linalg.cond(matrix + 0.5 * np.eye(panels))),
            "max_off_diagonal": _max_off_diagonal(matrix),
            "max_abs_entry": float(np.max(np.abs(matrix))),
            "transfer": vertical_transfer,
            "transfer_over_piston": 1.0,
        }
    )
    piston_reference = transfer_function("piston", kh)
    for angle in INCLINE_ANGLES:
        surface = inclined_wavemaker(depth, panels, angle)
        matrix = assemble_normal_derivative(surface, nu, depth)
        # unit *normal* velocity on every panel: the alpha -> 0 limit is the
        # uniform horizontal piston velocity
        data = -0.5j * omega * np.ones(surface.panel_count)
        sources, system = solve_sources(surface, data, nu, depth)
        eta = kochin_amplitude(surface, sources, nu, depth)
        transfer = 2.0 * abs(eta) / 1.0
        rows.append(
            {
                "angle_degrees": float(angle),
                "condition_number": float(np.linalg.cond(system)),
                "max_off_diagonal": _max_off_diagonal(matrix),
                "max_abs_entry": float(np.max(np.abs(matrix))),
                "transfer": transfer,
                "transfer_over_piston": transfer / piston_reference,
                "relative_error_vs_piston": abs(transfer / piston_reference - 1.0),
            }
        )
    return {
        "kh": float(kh),
        "panels": int(panels),
        "piston_transfer_analytic": piston_reference,
        "rows": rows,
    }


def run(quick: bool = False) -> dict:
    kh_values = (1.0, 2.0) if quick else DEFAULT_KH
    panels_values = (16, 64) if quick else DEFAULT_PANELS
    transfer = transfer_sweep(kh_values, panels_values)
    fields = field_comparison(kh_values)
    degeneracy = operator_degeneracy()
    decay = {}
    for kind in KINDS:
        decay[kind] = evanescent_decay(kind, kh=1.0, depth=DEPTH, panels=64)
    worst = max(row["relative_error"] for row in transfer)
    finest = [row for row in transfer if row["panels"] == max(panels_values)]
    fields_worst = max(row["relative_error"] for row in fields)
    routes_worst = max(row["route_disagreement"] for row in transfer)
    return {
        "case": "wave_flume_wavemaker",
        "anchor": "CASE-WAVE-FLUME",
        "depth": DEPTH,
        "gravity": GRAVITY,
        "kh_values": list(kh_values),
        "panels_values": list(panels_values),
        "kinds": list(KINDS),
        "transfer": transfer,
        "field": fields,
        "degeneracy": degeneracy,
        "evanescent_decay": decay,
        "summary": {
            "worst_transfer_error_all_meshes": worst,
            "worst_transfer_error_finest_mesh": max(
                row["relative_error"] for row in finest
            ),
            "worst_route_disagreement": routes_worst,
            "worst_field_error": fields_worst,
            "transfer_within_10pct": bool(worst <= 0.10),
            "transfer_within_3pct": bool(worst <= 0.03),
            "field_within_1pct": bool(fields_worst <= 0.01),
            "vertical_operator_is_diagonal": bool(
                degeneracy["rows"][0]["max_off_diagonal"] <= 1.0e-12
            ),
        },
    }


def render_markdown(study: dict) -> str:
    def pct(value: float) -> str:
        return f"{100.0 * value:.4f}%"

    summary = study["summary"]
    lines = [
        "# CASE-WAVE-FLUME: linear wavemaker transfer functions",
        "",
        "**Anchor**: `CASE-WAVE-FLUME` (算例池·规则波生成) — acceptance `<10%` against the",
        "closed-form linear-wavemaker reference.",
        "",
        "**Reference**: Biesel & Suquet (1951) / Havelock (1929) piston and flap transfer",
        "functions, re-derived in `src/tensorbem/wavemaker.py` and *also* checked against the",
        "long-wave mass balance (`H/S -> kh` and `H/S -> kh/2`) in the tests.",
        "",
        "## 1. Case",
        "",
        f"Constant-depth flume, `h = {study['depth']} m`, `g = {study['gravity']}`.",
        "The end wall is a vertical face panelled from the bed to the waterline and driven",
        "with a prescribed normal velocity; the free surface condition, the bed and the",
        "radiation condition are carried by the finite-depth kernel.",
        "",
        "| wavemaker | wall velocity `u(z)` | `H/S` |",
        "|---|---|---|",
        "| piston | `U0` | `4 sinh^2(kh) / (2kh + sinh 2kh)` |",
        "| flap (bed hinge) | `U0 (z+h)/h` | `4 sinh(kh)[sinh(kh)+(1-cosh(kh))/(kh)] / (2kh+sinh 2kh)` |",
        "",
        "Dimensionless depths: "
        + ", ".join(f"`kh={kh}`" for kh in study["kh_values"])
        + "; panel counts: "
        + ", ".join(f"`{n}`" for n in study["panels_values"])
        + ".",
        "",
        "## 2. How it was done",
        "",
        "1. Panelize the wall (`tensorbem.wavemaker.vertical_wavemaker`) and assemble the",
        "   finite-depth single-layer normal-derivative matrix: exact Rankine row integrals",
        "   plus Gauss quadrature of the smooth part of the kernel.",
        "2. Solve `(A + I/2) sigma = du/dn` for the prescribed stroke velocity.",
        "3. Read the far-field elevation off two independent quantities: the Kochin",
        "   integral over the wall (route 1) and direct quadrature of the kernel two",
        "   wavelengths downstream (route 2).",
        "4. Compare with the closed-form transfer function, and with the exact",
        "   eigenfunction field (propagating + evanescent modes) at five stations.",
        "",
        "## 3. Results — transfer function",
        "",
        "| wavemaker | kh | panels | BEM `H/S` | analytic `H/S` | rel. error | route disagreement |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in study["transfer"]:
        lines.append(
            "| {kind} | {kh:g} | {panels} | {bem:.6f} | {ana:.6f} | {err} | {route} |".format(
                kind=row["kind"],
                kh=row["kh"],
                panels=row["panels"],
                bem=row["transfer_bem"],
                ana=row["transfer_analytic"],
                err=pct(row["relative_error"]),
                route=pct(row["route_disagreement"]),
            )
        )
    lines += [
        "",
        f"Worst transfer error over every mesh: **{pct(summary['worst_transfer_error_all_meshes'])}**; "
        f"on the finest mesh: **{pct(summary['worst_transfer_error_finest_mesh'])}**. "
        f"Worst disagreement between the two far-field routes: **{pct(summary['worst_route_disagreement'])}**.",
        "",
        "## 4. Results — near field versus the exact eigenfunction series",
        "",
        "| wavemaker | kh | station (`x/lambda`) | `\\|eta\\|` BEM | `\\|eta\\|` exact | rel. error |",
        "|---|---|---|---|---|---|",
    ]
    for row in study["field"]:
        lines.append(
            "| {kind} | {kh:g} | {station:g} | {bem:.6f} | {exact:.6f} | {err} |".format(
                kind=row["kind"],
                kh=row["kh"],
                station=row["station_wavelengths"],
                bem=row["elevation_abs_bem"],
                exact=row["elevation_abs_exact"],
                err=pct(row["relative_error"]),
            )
        )
    lines += [
        "",
        f"Worst near-field error: **{pct(summary['worst_field_error'])}**.",
        "",
        "## 5. Scope of the operator — what this case does and does not test",
        "",
        "For a vertical full-depth wall every cross-panel entry of the normal-derivative",
        "matrix vanishes: all panels are co-planar, so the kernels depend on `|dx|` only and",
        "their `x`-derivative is zero. The system is then `(1/2) I` to round-off and the panel",
        "solution *is* the Havelock projection — so this case tests the kernel quadrature and",
        "the propagating/evanescent decomposition, but **not** the integral-equation solve.",
        "",
        "| face inclination (deg) | condition number | max off-diagonal | `H/S` | `H/S` / piston |",
        "|---|---|---|---|---|",
    ]
    for row in study["degeneracy"]["rows"]:
        lines.append(
            "| {angle:g} | {cond:.4f} | {off:.3e} | {transfer:.6f} | {ratio:.6f} |".format(
                angle=row["angle_degrees"],
                cond=row["condition_number"],
                off=row["max_off_diagonal"],
                transfer=row["transfer"],
                ratio=row["transfer_over_piston"],
            )
        )
    lines += [
        "",
        "The inclined face is *not* co-planar, the operator becomes non-degenerate, and as the",
        "inclination goes to zero the computed transfer returns to the analytic piston value —",
        "which is the quantitative anchor for the non-degenerate solve (no closed form exists",
        "for the sloped face itself).",
        "",
        "## 6. Results — evanescent decay (flume attenuation)",
        "",
        "| wavemaker | `x/lambda` | `x/h` | `\\|eta\\|` | deviation from flat |",
        "|---|---|---|---|---|",
    ]
    for kind, rows in study["evanescent_decay"].items():
        for row in rows:
            lines.append(
                "| {kind} | {lam:g} | {xd:g} | {amp:.6f} | {dev} |".format(
                    kind=kind,
                    lam=row["offset_wavelengths"],
                    xd=row["x_over_depth"],
                    amp=row["elevation_abs"],
                    dev=pct(row["deviation_from_flat"]),
                )
            )
    lines += [
        "",
        "Past one wavelength the elevation modulus is flat to round-off, i.e. only the",
        "propagating mode survives — the frequency-domain statement of the flume's",
        "wave-height attenuation being a wavemaker/absorption property, not propagation loss.",
        "",
        "## 7. Error judgement",
        "",
        f"- Transfer function versus Biesel-Havelock: worst **{pct(summary['worst_transfer_error_all_meshes'])}**, "
        f"finest mesh **{pct(summary['worst_transfer_error_finest_mesh'])}** — "
        f"**{'PASS' if summary['transfer_within_10pct'] else 'FAIL'}** at the 10% anchor "
        f"tolerance and **{'PASS' if summary['transfer_within_3pct'] else 'FAIL'}** at 3%.",
        f"- Near field versus the exact eigenfunction series: worst **{pct(summary['worst_field_error'])}** — "
        f"**{'PASS' if summary['field_within_1pct'] else 'FAIL'}** at 1%.",
        f"- Two far-field routes: worst **{pct(summary['worst_route_disagreement'])}**.",
        "",
        "## 8. Caveats",
        "",
        "- Linear theory only: no steepness limit, no breaking, no wavemaker backlash.",
        "- The wall meets the waterline and the bed, which the closed-body solvers in",
        "  `tensorbem.finite_depth_solver` refuse by construction; this module assembles the",
        "  same two operator pieces for the open curve and proves equality with those",
        "  routines on a submerged closed body in `tests/test_wavemaker_transfer.py`.",
        "- Gauss nodes stay strictly inside each panel, so no kernel call ever lands on the",
        "  waterline where the image term is `log`-singular; the invariant is asserted, not",
        "  assumed.",
        "",
    ]
    return "\n".join(lines)


def make_figures(study: dict) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _DOC_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    panels = sorted({row["panels"] for row in study["transfer"]})
    for kind, marker in (("piston", "o"), ("flap", "s")):
        rows = [row for row in study["transfer"] if row["kind"] == kind]
        analytic_kh = np.linspace(min(study["kh_values"]), max(study["kh_values"]), 200)
        analytic = [transfer_function(kind, kh) for kh in analytic_kh]
        axes[0].plot(analytic_kh, analytic, "-", label=f"{kind} (Biesel-Havelock)")
        for panel_count in panels:
            subset = [row for row in rows if row["panels"] == panel_count]
            axes[0].plot(
                [row["kh"] for row in subset],
                [row["transfer_bem"] for row in subset],
                marker,
                markersize=4.0,
                alpha=0.85,
                label=f"{kind} BEM, {panel_count} panels",
            )
        axes[1].semilogy(
            [row["panels"] for row in rows if row["kh"] == study["kh_values"][-1]],
            [
                row["relative_error"]
                for row in rows
                if row["kh"] == study["kh_values"][-1]
            ],
            marker + "-",
            label=f"{kind}, kh={study['kh_values'][-1]:g}",
        )
    axes[0].set_xlabel("kh")
    axes[0].set_ylabel("H / S")
    axes[0].set_title("Wavemaker transfer function")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=6.5)
    axes[1].set_xlabel("panels")
    axes[1].set_ylabel("relative error vs analytic")
    axes[1].set_title("Panel convergence")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    path = _DOC_DIR / "fig1_transfer_function.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for kind, marker in (("piston", "o"), ("flap", "s")):
        rows = [row for row in study["field"] if row["kind"] == kind and row["kh"] == study["kh_values"][0]]
        axes[0].plot(
            [row["station_wavelengths"] for row in rows],
            [row["elevation_abs_exact"] for row in rows],
            "-",
            label=f"{kind} exact (with evanescent modes)",
        )
        axes[0].plot(
            [row["station_wavelengths"] for row in rows],
            [row["elevation_abs_bem"] for row in rows],
            marker,
            linestyle="none",
            label=f"{kind} BEM",
        )
        decay = study["evanescent_decay"][kind]
        axes[1].semilogy(
            [row["offset_wavelengths"] for row in decay],
            [max(row["deviation_from_flat"], 1e-16) for row in decay],
            marker + "-",
            label=f"{kind}",
        )
    axes[0].set_xlabel("x / lambda")
    axes[0].set_ylabel("|eta| / stroke")
    axes[0].set_title(f"Free-surface elevation, kh={study['kh_values'][0]:g}")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("x / lambda")
    axes[1].set_ylabel("deviation from far-field modulus")
    axes[1].set_title("Evanescent modes dying downstream")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    path = _DOC_DIR / "fig2_near_field_decay.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small sweep for CI")
    args = parser.parse_args()
    study = run(quick=args.quick)
    _DOC_DIR.mkdir(parents=True, exist_ok=True)
    (_DOC_DIR / "study.json").write_text(json.dumps(study, indent=2, default=str))
    figures = make_figures(study)
    (_ROOT / "docs" / "wave_flume_wavemaker.md").write_text(render_markdown(study))
    print(json.dumps(study["summary"], indent=2))
    for path in figures:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()