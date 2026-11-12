"""Accuracy envelope of the sphere Helmholtz BEM on the ``ka`` plane.

Ledger anchor: ``CASE-ACOUSTIC-PULSE`` (reference = literature transient solution,
tolerance ``< 8 %``).  The case module :mod:`tensorbem.acoustic3d_pulse` proves the
transient bridge is correct *for one pulse band on one mesh*.  It does not answer
the question a user actually asks before spending CPU hours:

    "for a sphere of radius ``a`` at wavenumber ``k``, how many panels per
     wavelength do I need before the BEM harmonic amplitude is inside 8 % (the
     case tolerance) or inside 3 % (the repository's usual engineering bar)?"

That question is the difference between a *validated code* and a *code with a
published validity domain*.  This driver measures it directly.

Method (one dense Helmholtz solve per configuration, no transient synthesis):

* body = geodesic sphere of radius ``a = 0.5 m`` (``icosphere(subdivisions)``,
  ``20 * 4**s`` flat triangles);
* monopole Neumann data ``U = 1 m/s`` (radial), receiver on the x axis at
  ``r = 4 a`` (well outside the geometric near field of the mesh);
* ``ka in {1, 2, 3, 4}`` -- i.e. ``a / lambda = 1/2pi .. 2/pi``;
* ``subdivisions in {1, 2, 3}`` -- 80, 320, 1280 panels;
* the exact reference is the closed-form pulsating-sphere pressure
  :func:`tensorbem.acoustic3d.pulsating_sphere_pressure` (no far-field
  approximation), so every recorded error is BEM discretisation error only.

The two output axes that matter are *amplitude* relative error and *panels per
wavelength* ``lambda / h`` with the area-equivalent panel size
``h = sqrt(4 pi a^2 / N)``.  For each threshold the driver reports the coarsest
mesh level that meets it, and the corresponding ``lambda / h``.  Transient use is
covered by a bound, not by a second sweep: the L2 waveform error of a bin
synthesis cannot exceed its worst per-bin amplitude error, which is exactly what
``BemPulseSolution.max_per_bin_error`` reports (pinned by
``tests/test_acoustic3d_pulse.py``).

Everything printed to the markdown artifact is injected from the JSON payload;
no number in the document is retyped by hand.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from .. import acoustic3d as ac
from ..geometry3d import icosphere

FloatArray = NDArray[np.float64]

RADIUS = 0.5
RECEIVER_RANGE_FACTOR = 4.0
DEFAULT_KA = (1.0, 2.0, 3.0, 4.0)
DEFAULT_SUBDIVISIONS = (1, 2, 3)
THRESHOLDS = (0.03, 0.08)
DOC_ANCHOR = "# CASE-ACOUSTIC-PULSE validity envelope"


def panel_size(radius: float, panel_count: int) -> float:
    """Area-equivalent panel edge length of a sphere mesh."""

    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if panel_count <= 0:
        raise ValueError("panel_count must be positive")
    return math.sqrt(4.0 * math.pi * radius * radius / float(panel_count))


def panels_per_wavelength(ka: float, radius: float, panel_count: int) -> float:
    """``lambda / h`` for the area-equivalent panel size."""

    if ka <= 0.0:
        raise ValueError("ka must be positive")
    wavelength = 2.0 * math.pi * radius / ka
    return wavelength / panel_size(radius, panel_count)


def harmonic_error(
    ka: float,
    subdivisions: int,
    *,
    radius: float = RADIUS,
    receiver_range_factor: float = RECEIVER_RANGE_FACTOR,
    density: float = ac.DENSITY_SEAWATER,
    sound_speed: float = ac.SOUND_SPEED_SEAWATER,
    n_chief: int = 5,
) -> dict[str, float]:
    """Relative error of the BEM monopole pressure at one ``(ka, mesh)`` point."""

    if ka <= 0.0:
        raise ValueError("ka must be positive")
    mesh = icosphere(int(subdivisions), radius=radius)
    k = float(ka) / radius
    receiver = np.array([[receiver_range_factor * radius, 0.0, 0.0]], dtype=float)
    solution = ac.solve_exterior_neumann(mesh, k, complex(1.0, 0.0), n_chief=n_chief)
    potential = ac.field_potential(
        mesh, k, solution.phi, solution.normal_velocity, receiver
    )
    bem = complex(ac.acoustic_pressure(potential[0], k, density=density,
                                      sound_speed=sound_speed))
    exact = complex(
        ac.pulsating_sphere_pressure(
            radius, complex(1.0, 0.0), k, receiver,
            density=density, sound_speed=sound_speed,
        )[0]
    )
    if abs(exact) == 0.0:
        raise ValueError("exact pressure vanished; check the receiver point")
    return {
        "ka": float(ka),
        "subdivisions": int(subdivisions),
        "panel_count": int(mesh.panel_count),
        "panel_size_m": panel_size(radius, mesh.panel_count),
        "panels_per_wavelength": panels_per_wavelength(ka, radius, mesh.panel_count),
        "bem_amplitude": abs(bem),
        "exact_amplitude": abs(exact),
        "phase_error_rad": float(np.angle(bem / exact)),
        "relative_error": abs(bem - exact) / abs(exact),
    }


def coarsest_passing(rows: Sequence[dict[str, float]], threshold: float) -> dict | None:
    """Coarsest mesh level inside ``threshold`` at a fixed ``ka`` (None if none)."""

    passing = [row for row in rows if row["relative_error"] <= threshold]
    if not passing:
        return None
    return min(passing, key=lambda row: row["panel_count"])


def run_study(
    ka_values: Iterable[float] = DEFAULT_KA,
    subdivisions: Sequence[int] = DEFAULT_SUBDIVISIONS,
    *,
    radius: float = RADIUS,
) -> dict:
    """Sweep ``ka`` x mesh and return the injectable payload."""

    ka_list = [float(ka) for ka in ka_values]
    sub_list = [int(s) for s in subdivisions]
    if not ka_list or not sub_list:
        raise ValueError("ka_values and subdivisions must be non empty")

    rows: list[dict[str, float]] = []
    for ka in ka_list:
        for subdivisions_level in sub_list:
            rows.append(harmonic_error(ka, subdivisions_level, radius=radius))

    per_ka: list[dict] = []
    for ka in ka_list:
        level_rows = sorted(
            (row for row in rows if row["ka"] == ka), key=lambda row: row["panel_count"]
        )
        entry = {
            "ka": ka,
            "wavelength_over_radius": 2.0 * math.pi / ka,
            "rows": level_rows,
            "requirements": [],
        }
        for threshold in THRESHOLDS:
            passing = coarsest_passing(level_rows, threshold)
            entry["requirements"].append(
                {
                    "threshold": threshold,
                    "min_panel_count": None if passing is None else passing["panel_count"],
                    "min_panels_per_wavelength":
                        None if passing is None else passing["panels_per_wavelength"],
                    "unreachable_on_this_ladder": passing is None,
                }
            )
        per_ka.append(entry)

    finest: list[dict] = [
        max(entry["rows"], key=lambda row: row["panel_count"]) for entry in per_ka
    ]
    return {
        "case": "CASE-ACOUSTIC-PULSE validity envelope",
        "statement": (
            "Relative error of the frequency-domain Helmholtz BEM (rigid sphere, "
            "monopole Neumann data) against the closed-form pulsating-sphere "
            "pressure, as a function of ka and panels per wavelength."
        ),
        "radius_m": radius,
        "receiver_range_factor": RECEIVER_RANGE_FACTOR,
        "density_kg_m3": ac.DENSITY_SEAWATER,
        "sound_speed_m_s": ac.SOUND_SPEED_SEAWATER,
        "ka_values": ka_list,
        "subdivisions": sub_list,
        "thresholds": list(THRESHOLDS),
        "rows": rows,
        "per_ka": per_ka,
        "finest_mesh": finest,
        "tolerance": 0.08,
    }


def _fmt(value: float | None, spec: str = "{:.3e}") -> str:
    return "—" if value is None else spec.format(value)


def render_markdown(payload: dict) -> str:
    """Publication form of the payload; every number is injected."""

    lines: list[str] = []
    lines.append(DOC_ANCHOR)
    lines.append("")
    lines.append(
        "> Generated by `python -m tensorbem.benchmarks.acoustic3d_pulse_envelope "
        "--markdown docs/acoustic3d_pulse_envelope.md`; the numbers below are "
        "injected from the JSON payload, never retyped."
    )
    lines.append("")
    lines.append(
        f"Body: rigid sphere, radius `a = {payload['radius_m']} m`, receiver at "
        f"`r = {payload['receiver_range_factor']} a`, density "
        f"`{payload['density_kg_m3']} kg/m3`, sound speed "
        f"`{payload['sound_speed_m_s']} m/s`. Reference: closed-form pulsating-sphere "
        "pressure (no far-field approximation), so the recorded error is panel "
        "discretisation error only."
    )
    lines.append("")
    lines.append("## 1. Measured error per (ka, mesh)")
    lines.append("")
    lines.append("| ka | panels | lambda/h | |p_bem| | |p_exact| | rel. error |")
    lines.append("|---|---|---|---|---|---|")
    for row in payload["rows"]:
        lines.append(
            f"| {row['ka']:.2f} | {row['panel_count']} | "
            f"{row['panels_per_wavelength']:.2f} | {row['bem_amplitude']:.6e} | "
            f"{row['exact_amplitude']:.6e} | {row['relative_error']:.3%} |"
        )
    lines.append("")
    lines.append("## 2. Mesh requirement per threshold")
    lines.append("")
    header = "| ka | " + " | ".join(
        f"coarsest mesh inside {threshold:.0%}" for threshold in payload["thresholds"]
    ) + " |"
    lines.append(header)
    lines.append("|---|" + "---|" * len(payload["thresholds"]))
    for entry in payload["per_ka"]:
        cells = []
        for requirement in entry["requirements"]:
            if requirement["min_panel_count"] is None:
                cells.append("not reached on this ladder")
            else:
                cells.append(
                    f"{requirement['min_panel_count']} panels "
                    f"(λ/h = {requirement['min_panels_per_wavelength']:.2f})"
                )
        lines.append(f"| {entry['ka']:.2f} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## 3. Reading the table")
    lines.append("")
    lines.append(
        "* The error is a smooth function of `ka` and of `lambda/h`; refinement "
        "buys accuracy at every `ka` (see the test file for the per-level "
        "monotonicity assertions)."
    )
    lines.append(
        f"* `CASE-ACOUSTIC-PULSE` accepts `< {payload['tolerance']:.0%}`; the "
        "`3 %` column is the repository's engineering bar, not a case requirement."
    )
    lines.append(
        "* Transient use inherits this envelope through the per-bin bound: the "
        "L2 waveform error of a bin synthesis cannot exceed the worst per-bin "
        "amplitude error (`BemPulseSolution.max_per_bin_error`)."
    )
    lines.append(
        "* Limitation: the ladder tops out at 1280 panels, so a `ka` row whose "
        "`3 %` cell is empty means *this ladder* cannot resolve it, not that no "
        "mesh can."
    )
    lines.append("")
    return "\n".join(lines)


def render_figure(payload: dict, path: pathlib.Path) -> pathlib.Path:
    """Error vs panels per wavelength, one curve per ka."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(7.0, 4.5))
    for entry in payload["per_ka"]:
        x = [row["panels_per_wavelength"] for row in entry["rows"]]
        y = [row["relative_error"] for row in entry["rows"]]
        axes.loglog(x, y, "o-", label=f"ka = {entry['ka']:.2f}")
    for threshold, style in ((0.03, "--"), (0.08, ":")):
        axes.axhline(threshold, color="k", linestyle=style, linewidth=1.0)
        axes.annotate(f"{threshold:.0%}", xy=(axes.get_xlim()[0], threshold),
                      xytext=(2, 2), textcoords="offset points", fontsize=8)
    axes.set_xlabel(r"panels per wavelength  $\lambda/h$")
    axes.set_ylabel("relative amplitude error")
    axes.set_title("Sphere Helmholtz BEM: accuracy envelope (CASE-ACOUSTIC-PULSE)")
    axes.grid(True, which="both", alpha=0.3)
    axes.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the payload")
    parser.add_argument("--write", type=pathlib.Path, help="write the JSON payload")
    parser.add_argument("--markdown", type=pathlib.Path, help="write the markdown report")
    parser.add_argument("--figure", type=pathlib.Path, help="write the PNG figure")
    parser.add_argument("--ka", type=float, nargs="*", default=list(DEFAULT_KA))
    parser.add_argument("--subdivisions", type=int, nargs="*",
                        default=list(DEFAULT_SUBDIVISIONS))
    args = parser.parse_args(argv)

    payload = run_study(args.ka, args.subdivisions)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(payload), encoding="utf-8")
    if args.figure:
        render_figure(payload, args.figure)
    if args.json or not (args.write or args.markdown or args.figure):
        print(json.dumps(payload, indent=2))
    for entry in payload["per_ka"]:
        print(
            f"ka={entry['ka']:.2f}: " + ", ".join(
                f"{requirement['threshold']:.0%} -> "
                + ("none" if requirement["min_panel_count"] is None
                   else f"{requirement['min_panel_count']} panels "
                        f"(lambda/h={requirement['min_panels_per_wavelength']:.2f})")
                for requirement in entry["requirements"]
            )
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())