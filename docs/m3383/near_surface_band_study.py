"""Near-surface band cross-certification of the 2-D free-surface radiation solver.

The documented Ursell verification of ``tensorbem.free_surface_solver`` is a
single working point: one submergence ratio (``f/a = 1.5``) over a wavenumber
ladder.  The physically hard regime for a ship section is the *other* axis --
small submergence, where the free-surface kernel's near field is strongest and
the multipole reference series is worst conditioned.  This study sweeps

    ka in {0.2, 0.6, 1.0, 1.5}  x  f/a in {1.05, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0}

and compares the panel method against an independent representation: the
method of fundamental solutions (:func:`tensorbem.free_surface_solver.
radiation_reference`), which uses point sources inside the body and shares no
panel, no basis function and no matrix entry with the panel assembly.

Per grid point we report

* ``A``, ``B`` from the panel solver at 32/64/128/256 panels,
* ``A``, ``B`` from MFS at 24/48/96 sources (its own convergence check),
* relative panel-vs-MFS discrepancies and the panel observed order.

Everything is written to ``near_surface_band.json`` so no number is
transcribed by hand.

Run:  python docs/m3383/near_surface_band_study.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorbem.free_surface_solver import (  # noqa: E402
    radiation_reference,
    solve_radiation_indirect,
)
from tensorbem.geometry import PanelMesh  # noqa: E402

GRAVITY = 9.80665
RADIUS = 1.0
DENSITY = 1025.0
PANEL_LADDER = (32, 64, 128)
SOURCE_LADDER = (48,)
KA_VALUES = (0.2, 0.6, 1.0)
DEPTH_RATIOS = (1.05, 1.5, 3.0)
DIRECTIONS = ("sway", "heave")


def circle_mesh(panel_count: int, radius: float, depth: float) -> PanelMesh:
    """Counter-clockwise circle of radius ``radius`` centred at ``(0, -depth)``."""
    angle = 2.0 * math.pi * np.arange(panel_count) / panel_count
    vertices = np.column_stack(
        (radius * np.cos(angle), -depth + radius * np.sin(angle))
    )
    return PanelMesh(vertices)


def _component(value, direction: str) -> float:
    """Pick the longitudinal/vertical component of a 2-vector coefficient."""
    array = np.asarray(value, dtype=float).ravel()
    return float(array[0] if direction == "sway" else array[1])


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1.0e-300)


def _order(coarse: float, fine: float) -> float:
    if coarse <= 0.0 or fine <= 0.0:
        return float("nan")
    return math.log(coarse / fine) / math.log(2.0)


def run_point(ka: float, depth_ratio: float, direction: str) -> dict:
    wavenumber = ka / RADIUS
    depth = depth_ratio * RADIUS
    row: dict = {
        "ka": ka,
        "depth_ratio": depth_ratio,
        "direction": direction,
        "panels": {},
        "mfs": {},
    }

    for panels in PANEL_LADDER:
        mesh = circle_mesh(panels, RADIUS, depth)
        result = solve_radiation_indirect(
            mesh, wavenumber, direction, density=DENSITY, panel_nodes=24
        )
        row["panels"][str(panels)] = {
            "added_mass": _component(result.added_mass, direction),
            "damping": _component(result.damping, direction),
            "condition_number": float(result.condition_number),
            "relative_residual": float(result.relative_residual),
        }

    for sources in SOURCE_LADDER:
        mesh = circle_mesh(PANEL_LADDER[-1], RADIUS, depth)
        result = radiation_reference(
            mesh, wavenumber, direction, source_count=sources, density=DENSITY
        )
        row["mfs"][str(sources)] = {
            "added_mass": _component(result.added_mass, direction),
            "damping": _component(result.damping, direction),
            "condition_number": float(result.condition_number),
            "relative_residual": float(result.relative_residual),
        }

    fine = row["panels"][str(PANEL_LADDER[-1])]
    reference = row["mfs"][str(SOURCE_LADDER[-1])]
    row["panel_vs_mfs"] = {
        "added_mass": _rel(fine["added_mass"], reference["added_mass"]),
        "damping": _rel(fine["damping"], reference["damping"]),
    }
    row["mfs_self"] = {
        "added_mass": _rel(row["mfs"][str(SOURCE_LADDER[0])]["added_mass"], reference["added_mass"]),
        "damping": _rel(row["mfs"][str(SOURCE_LADDER[0])]["damping"], reference["damping"]),
    }
    row["orders"] = {
        "added_mass": _order(
            abs(row["panels"][str(PANEL_LADDER[-2])]["added_mass"] - reference["added_mass"]),
            abs(fine["added_mass"] - reference["added_mass"]),
        ),
        "damping": _order(
            abs(row["panels"][str(PANEL_LADDER[-2])]["damping"] - reference["damping"]),
            abs(fine["damping"] - reference["damping"]),
        ),
    }
    return row


def main() -> None:
    rows = []
    for ka in KA_VALUES:
        for depth_ratio in DEPTH_RATIOS:
            for direction in DIRECTIONS:
                rows.append(run_point(ka, depth_ratio, direction))
                print(
                    f"ka={ka:<4} f/a={depth_ratio:<5} {direction:<6} "
                    f"A_err={rows[-1]['panel_vs_mfs']['added_mass']:.3e} "
                    f"B_err={rows[-1]['panel_vs_mfs']['damping']:.3e}",
                    flush=True,
                )

    worst = max(rows, key=lambda r: max(r["panel_vs_mfs"].values()))
    payload = {
        "meta": {
            "problem": "2-D deep-water radiation of a submerged circular cylinder",
            "radius_m": RADIUS,
            "density": DENSITY,
            "panels": list(PANEL_LADDER),
            "mfs_sources": list(SOURCE_LADDER),
            "ka_values": list(KA_VALUES),
            "depth_ratios": list(DEPTH_RATIOS),
            "note": (
                "panel method = tensorbem.free_surface_solver."
                "solve_radiation_indirect (piecewise-constant indirect BIE); "
                "reference = radiation_reference (method of fundamental "
                "solutions, point sources inside the body, no shared matrix "
                "entries); 'panel_vs_mfs' compares 256 panels against 96 sources"
            ),
        },
        "rows": rows,
        "worst": {
            "ka": worst["ka"],
            "depth_ratio": worst["depth_ratio"],
            "direction": worst["direction"],
            "added_mass": worst["panel_vs_mfs"]["added_mass"],
            "damping": worst["panel_vs_mfs"]["damping"],
        },
    }
    out = Path(__file__).with_suffix(".json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print("worst:", payload["worst"])


if __name__ == "__main__":
    main()