"""W25: B4-70 open-water error surface against Oosterveld & van Oossanen (1975).

Research question (mission ``paper-exp-propeller-openwater-bem-7``)
------------------------------------------------------------------
1. On the advance-ratio range ``J = 0.1 .. 1.2`` (10 points), how far is the
   repository's vortex-lattice open-water kernel from the Wageningen B4-70
   regression (Oosterveld & van Oossanen 1975, NSMB Publ. 479), and how much of
   that distance is owned by the two axes we can actually move -- panel density
   and the trailing-vortex (wake) model?
2. At the KP505 design point, where does the historically recorded ~29 %
   deviation come from: blade geometry, trailing-vortex model, or friction?

Predeclared criteria (fixed *before* the sweep was run, per the W25 contract)
----------------------------------------------------------------------------
C1  for every grid point with ``J <= 0.8`` the relative deviations of ``K_T``
    and ``K_Q`` satisfy ``|d| < 15 %`` -- the mission's "unacceptable above
    15 %" standard -- evaluated on the highest-fidelity configuration available
    in the module (fine lattice, induced wake, ITTC-78 friction);
C2  ``|d eta_0| < 15 %`` on the same points and configuration;
C3  the ablation axes are quantified and the parts add up: for a three-factor
    Shapley decomposition ``|sum_i phi_i - (v(all) - v(none))| / scale < 1e-9``
    (exactness check), and the unattributed remainder ``v(none)`` -- the
    model-form term (assumed chord and radial pitch distribution, zero camber,
    no hub or thickness blockage) -- is reported instead of being hidden.

A criterion that fails is recorded as ``honest FAIL`` and is *not* retuned
afterwards.

How the anchor is used
----------------------
Every anchor value comes from ``benchmarks/anchors/wageningen_b_series.json``
(39-term KT / 47-term KQ regression, machine-extracted -- see
``tools/build_wageningen_anchor.py``) evaluated through
:class:`tensorbem.wageningen.B4Series`.  No coefficient is transcribed by hand.

Friction comparability: the regression is defined at ``Re = 2e6`` referred to the
0.75 R chord, so the B4-70 configuration uses ``re_match=True``, which derives
the rev/s of every point from ``Re = W_nd n c_nd D^2 / nu`` (recorded per point).

Deviation from the first predeclaration (recorded, with evidence)
---------------------------------------------------------------
The fine level was predeclared as "32 x 2 lattice + induced helix L = 2.0 D / 48
steps, chosen by a 400-panel budget".  The pre-sweep stability probe
(``stability_probe``, grid x mesh x wake model) showed that the *non-geometric*
helices (kinematic = free-stream, induced = force-free aligned) return a
near-singular induction solve at low advance ratio on this blade, e.g.
``K_T = -9.6e4`` at ``J = 0.1`` with a 16 x 2 lattice -- a numerical breakdown,
not physics (the same lattice with the geometric helix gives ``K_T = 0.53``).
The fine configuration is therefore *derived by rule* from the probe instead of
being asserted: highest wake fidelity that is numerically stable at every grid
point, then the largest lattice under the 400-panel budget.  The rule yields a
geometric helix (L = 2.0 D, 48 steps) on a 24 x 2 lattice, and the family
difference between the three wake models is reported as a sensitivity study over
the part of the grid where the non-geometric helices are stable (J >= 0.4),
never dropped silently.

Run::

    python -m tensorbem.benchmarks.b4_70_error_surface [--outdir DIR] [--quick]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..propeller import TWO_PI, WATER_NU, BladeGeometry, open_water_point
from ..wageningen import B4Series, anchor_payload, b4_70, verify_tables
from .kp505_openwater_bem import (
    ANCHOR as KP505_ANCHOR,
    ANCHOR_SOURCE as KP505_ANCHOR_SOURCE,
    DESIGN_ADVANCE as KP505_DESIGN_ADVANCE,
    GEOMETRY_SOURCE as KP505_GEOMETRY_SOURCE,
)

# ---------------------------------------------------------------------------
# Predeclared run constants
# ---------------------------------------------------------------------------

#: Mission grid: 10 evenly spaced points spanning J = 0.1 .. 1.2 inclusive.
PRIMARY_GRID = [round(0.1 + (1.2 - 0.1) * i / 9.0, 10) for i in range(10)]

#: Supplementary round-J grid over the same span (12 points) for readers.
ROUND_GRID = [round(0.1 * i, 10) for i in range(1, 13)]

CRITERION_J_MAX = 0.8
KT_GATE_PCT = 15.0
KQ_GATE_PCT = 15.0
ETA_GATE_PCT = 15.0

#: Reynolds reference of the B-series regression (0.75 R chord).
ANCHOR_RE = 2.0e6

#: Lattice / wake levels.  ``MESH_FINE`` and ``WAKE_FINE`` are *derived* from the
#: stability probe by :func:`select_fine_configuration` and cross-checked against
#: these constants (the constants only save the reader a lookup; the artifact
#: carries the derivation).
MESH_BASE = {"n_radial": 16, "n_chord": 2}
MESH_FINE = {"n_radial": 24, "n_chord": 2}

#: Pre-sweep numerical-stability probe: lattice x wake model, over the J grid.
STABILITY_PROBE_MESHES = [(8, 1), (16, 2), (24, 2)]
STABILITY_PROBE_WAKES = [
    {"wake_model": "geometric", "wake_length": 1.0, "wake_steps": 24},
    {"wake_model": "geometric", "wake_length": 2.0, "wake_steps": 48},
    {"wake_model": "kinematic", "wake_length": 1.0, "wake_steps": 24},
    {"wake_model": "induced", "wake_length": 2.0, "wake_steps": 48},
]
STABILITY_PROBE_J = [0.1, 0.3, 0.5, 0.8, 1.0, 1.2]

#: Numerical-stability acceptance rule.  A blow-up of this lattice returns
#: ``|K_T|`` of order 1e2..1e6 and ``|K_Q|`` of order 1e1..1e10, while a genuine
#: open-water point has ``|K_T| <= 1`` and ``|K_Q| <= 0.2`` even past zero
#: thrust; the rule is written on magnitude only so that physically negative
#: thrust near the zero-thrust advance ratio is *not* mislabelled.
KT_PLAUSIBLE_MAX = 1.0
KQ_PLAUSIBLE_MAX = 0.2
MESH_LADDER = [
    {"n_radial": 8, "n_chord": 1},
    {"n_radial": 8, "n_chord": 2},
    {"n_radial": 16, "n_chord": 1},
    {"n_radial": 16, "n_chord": 2},
    {"n_radial": 24, "n_chord": 2},
    {"n_radial": 32, "n_chord": 2},
]

WAKE_BASE = {"wake_model": "geometric", "wake_length": 1.0, "wake_steps": 24}
WAKE_FINE = {"wake_model": "geometric", "wake_length": 2.0, "wake_steps": 48}
WAKE_LADDER = [
    {"wake_model": "geometric", "wake_length": 1.0, "wake_steps": 24},
    {"wake_model": "geometric", "wake_length": 2.0, "wake_steps": 48},
    {"wake_model": "kinematic", "wake_length": 1.0, "wake_steps": 24},
    {"wake_model": "kinematic", "wake_length": 2.0, "wake_steps": 48},
    {"wake_model": "induced", "wake_length": 1.0, "wake_steps": 24},
    {"wake_model": "induced", "wake_length": 2.0, "wake_steps": 48},
]

FRICTION_BASE = {"viscous": False}
FRICTION_FINE = {"viscous": True}


# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """One open-water model configuration of the lattice kernel."""

    blades: int
    expanded_area_ratio: float
    pitch_ratio: float
    hub_ratio: float = 0.18
    thickness_ratio: float = 0.06
    root_chord_offset: float = 0.12
    n_radial: int = 16
    n_chord: int = 2
    wake_model: str = "geometric"
    wake_length: float = 1.0
    wake_steps: int = 24
    viscous: bool = True
    radial_spacing: str = "cosine"
    rev_per_second: float = 10.0
    re_match: bool = False

    def geometry(self) -> BladeGeometry:
        return BladeGeometry(
            diameter=1.0,
            blades=self.blades,
            hub_ratio=self.hub_ratio,
            pitch_ratio=self.pitch_ratio,
            expanded_area_ratio=self.expanded_area_ratio,
            root_chord_offset=self.root_chord_offset,
            thickness_ratio=self.thickness_ratio,
            n_radial=self.n_radial,
            n_chord=self.n_chord,
            radial_spacing=self.radial_spacing,
        )

    def revs_for(self, advance: float, re_target: float = ANCHOR_RE) -> float:
        """Rev/s that puts the 0.75 R section at ``Re = re_target``.

        ``Re = W_phys c_phys / nu`` with ``W_phys = W_nd n D`` and
        ``c_phys = c_nd D``; the lattice is non-dimensionalised with ``D = 1``.
        """

        if not self.re_match:
            return self.rev_per_second
        geometry = self.geometry()
        radius = geometry.radius
        chord = float(geometry.chord(np.array([0.75 * radius]))[0])
        w_nd = math.hypot(advance, TWO_PI * 0.75 * radius)
        return re_target * WATER_NU / (chord * w_nd * geometry.diameter**2)


B4_70_GEOMETRY_DECLARED = {
    "Z": 4,
    "Ae_Ao": 0.70,
    "P_D": 1.00,
    "hub_ratio": 0.18,
    "hub_ratio_source": (
        "B-series hub-ratio band 0.18-0.20 (O&O 1975 validity domain); 0.18 is the "
        "series' stated hub ratio"
    ),
    "thickness_ratio": 0.06,
    "section": "flat zero-camber lattice panel (B4-70 sections are NACA66)",
    "chord_distribution": "parametric hump (root_chord_offset 0.12, exponents 0.80/0.50)",
    "radial_pitch": (
        "constant at the mean P/D (the B-series radial pitch distribution is not "
        "available offline; it is not invented here)"
    ),
}

B4_70_CONFIG = Config(
    blades=4,
    expanded_area_ratio=0.70,
    pitch_ratio=1.00,
    rev_per_second=3.0,
    re_match=True,
)

B4_70_FINE = replace(B4_70_CONFIG, **MESH_FINE, **WAKE_FINE, **FRICTION_FINE)
B4_70_BASE = replace(B4_70_CONFIG)


# ---------------------------------------------------------------------------
# Kernel wrapper
# ---------------------------------------------------------------------------


def solve(config: Config, advance: float) -> dict:
    """Run one open-water point and return the non-dimensional coefficients.

    Results are memoised on ``(config, J)`` -- the Windows/ablation design calls
    the same configuration many times (sweeps, ladders, Shapley coalitions), and
    the cache makes those repeated calls free without changing any number.
    """

    return dict(_solve_cached(config, float(advance)))


@lru_cache(maxsize=8192)
def _solve_cached(config: Config, advance: float) -> dict:
    geometry = config.geometry()
    revs = config.revs_for(advance)
    result = open_water_point(
        geometry,
        advance,
        wake_model=config.wake_model,
        wake_length=config.wake_length,
        wake_steps=config.wake_steps,
        rev_per_second=revs,
        viscous=config.viscous,
    )
    resisting_torque = -result.torque  # the lattice torque is negative (drag)
    kq = resisting_torque
    efficiency = (
        advance * result.thrust / (TWO_PI * kq) if abs(kq) > 1e-14 else float("nan")
    )
    return {
        "J": float(advance),
        "KT": float(result.thrust),
        "KQ": float(kq),
        "10KQ": float(10.0 * kq),
        "eta0": float(efficiency),
        "n_panels_per_blade": int(result.n_panels),
        "rev_per_second": float(revs),
        "viscous": bool(config.viscous),
        "wake_model": config.wake_model,
        "wake_length": config.wake_length,
        "wake_steps": config.wake_steps,
        "n_radial": config.n_radial,
        "n_chord": config.n_chord,
        "physical": bool(result.thrust > 0.0 and kq > 0.0 and 0.0 < efficiency <= 1.5),
    }


def _reynolds_at_075(config: Config, advance: float) -> float:
    """Section Reynolds number at 0.75 R implied by the run's rev/s."""

    geometry = config.geometry()
    radius = geometry.radius
    chord = float(geometry.chord(np.array([0.75 * radius]))[0])
    w_nd = math.hypot(advance, TWO_PI * 0.75 * radius)
    revs = config.revs_for(advance)
    return w_nd * revs * chord * geometry.diameter**2 / WATER_NU


def _rel(value: float, anchor: float) -> float:
    if anchor is None or not np.isfinite(anchor) or abs(anchor) < 1e-12:
        return float("nan")
    return 100.0 * (value - anchor) / anchor


def score_point(config: Config, advance: float, series: B4Series,
                kt_ref0: float, kq_ref0: float) -> dict:
    """One BEM-versus-anchor comparison row."""

    row = solve(config, advance)
    kt_ref = series.kt(advance)
    kq_ref = series.kq(advance)
    eta_ref = series.eta0(advance)
    row.update(
        {
            "reynolds_075R": _reynolds_at_075(config, advance),
            "KT_anchor": float(kt_ref),
            "KQ_anchor": float(kq_ref),
            "10KQ_anchor": float(10.0 * kq_ref),
            "eta0_anchor": float(eta_ref),
            "dKT_pct": float(_rel(row["KT"], kt_ref)),
            "dKQ_pct": float(_rel(row["KQ"], kq_ref)),
            "d10KQ_pct": float(_rel(row["10KQ"], 10.0 * kq_ref)),
            "deta0_pct": float(_rel(row["eta0"], eta_ref)),
            # anchor-free absolute measures that stay finite where the
            # regression itself crosses zero
            "dKT_norm_by_KT0": float((row["KT"] - kt_ref) / kt_ref0),
            "dKQ_norm_by_KQ0": float((row["KQ"] - kq_ref) / kq_ref0),
            "in_criterion_range": bool(advance <= CRITERION_J_MAX + 1e-12),
            "anchor_KT_positive": bool(kt_ref > 0.0),
            "anchor_KQ_positive": bool(kq_ref > 0.0),
        }
    )
    return row


def sweep(config: Config, grid, series: B4Series) -> list:
    kt_ref0 = series.kt(0.0)
    kq_ref0 = series.kq(0.0)
    return [score_point(config, advance, series, kt_ref0, kq_ref0) for advance in grid]


def stability_probe() -> dict:
    """Numerical-stability envelope of the lattice over (lattice x wake model x J).

    Run *before* the error surface is scored, because the envelope decides which
    ablation levels are admissible.  Every row is kept, including the unstable
    ones: they are evidence about the kernel, not noise to be dropped.
    """

    rows = []
    for n_radial, n_chord in STABILITY_PROBE_MESHES:
        for wake in STABILITY_PROBE_WAKES:
            unstable_at = []
            for advance in STABILITY_PROBE_J:
                config = replace(
                    B4_70_CONFIG, n_radial=n_radial, n_chord=n_chord, **wake
                )
                row = solve(config, advance)
                stable = bool(
                    np.isfinite(row["KT"])
                    and np.isfinite(row["KQ"])
                    and abs(row["KT"]) <= KT_PLAUSIBLE_MAX
                    and abs(row["KQ"]) <= KQ_PLAUSIBLE_MAX
                )
                rows.append(
                    {
                        "J": advance,
                        "n_radial": n_radial,
                        "n_chord": n_chord,
                        "n_panels_per_blade": n_radial * n_chord,
                        "wake_model": wake["wake_model"],
                        "wake_length": wake["wake_length"],
                        "wake_steps": wake["wake_steps"],
                        "KT": row["KT"],
                        "10KQ": row["10KQ"],
                        "numerically_stable": stable,
                    }
                )
                if not stable:
                    unstable_at.append(
                        {"J": advance, "KT": row["KT"], "10KQ": row["10KQ"]}
                    )
            rows[-1]["_unstable_at"] = unstable_at  # placeholder removed below
    for row in rows:
        row.pop("_unstable_at", None)

    # summarise per (lattice, wake) combination
    combinations = {}
    for row in rows:
        key = (
            row["n_radial"],
            row["n_chord"],
            row["wake_model"],
            row["wake_length"],
            row["wake_steps"],
        )
        entry = combinations.setdefault(
            key,
            {
                "n_radial": row["n_radial"],
                "n_chord": row["n_chord"],
                "n_panels_per_blade": row["n_panels_per_blade"],
                "wake_model": row["wake_model"],
                "wake_length": row["wake_length"],
                "wake_steps": row["wake_steps"],
                "stable_at_all_J": True,
                "unstable_points": [],
            },
        )
        if not row["numerically_stable"]:
            entry["stable_at_all_J"] = False
            entry["unstable_points"].append(
                {"J": row["J"], "KT": row["KT"], "10KQ": row["10KQ"]}
            )
    return {
        "probe_J": list(STABILITY_PROBE_J),
        "rule": (
            f"numerically stable iff |KT| <= {KT_PLAUSIBLE_MAX} and "
            f"|KQ| <= {KQ_PLAUSIBLE_MAX} (blow-ups are 1e2..1e6; physical points "
            "stay inside these bounds even past zero thrust)"
        ),
        "rows": rows,
        "combinations": sorted(
            combinations.values(),
            key=lambda e: (e["n_radial"], e["wake_model"], e["wake_length"]),
        ),
    }


#: Wake-fidelity order used to derive the fine configuration (highest first):
#: the induced helix follows the local total velocity, the kinematic helix the
#: free stream, the geometric helix the blade pitch, and a longer/finer helix
#: beats a shorter one.
WAKE_FIDELITY = [
    ("induced", 2.0, 48),
    ("induced", 1.0, 24),
    ("kinematic", 2.0, 48),
    ("kinematic", 1.0, 24),
    ("geometric", 2.0, 48),
    ("geometric", 1.0, 24),
]


def select_fine_configuration(probe: dict, panel_budget: int = 400) -> dict:
    """Derive the highest-fidelity *numerically stable* level combination.

    Rule (fixed before the sweep, only the stability data is empirical):

    1. a wake setting is admissible only if it is stable at every probe J **and
       at every lattice level of the probe** -- the wake setting must not depend
       on the lattice, otherwise the two ablation axes are not separable;
    2. among the admissible wake settings take the highest fidelity in
       :data:`WAKE_FIDELITY` order;
    3. then take the largest lattice that is stable with that wake setting
       within the panel budget.

    The rule is deliberately ordered so that step 3 can be checked against step
    2: it refuses a "highest wake fidelity on the coarsest lattice" answer, which
    would make the mesh axis degenerate.
    """

    lattices = sorted(
        {(entry["n_radial"], entry["n_chord"]) for entry in probe["combinations"]}
    )
    stability = {}
    for model, length, steps in {
        (e["wake_model"], e["wake_length"], e["wake_steps"])
        for e in probe["combinations"]
    }:
        key = (model, length, steps)
        stable_lattices = [
            (entry["n_radial"], entry["n_chord"])
            for entry in probe["combinations"]
            if entry["wake_model"] == model
            and entry["wake_length"] == length
            and entry["wake_steps"] == steps
            and entry["stable_at_all_J"]
        ]
        stability[key] = {
            "stable_lattices": [
                {"n_radial": nr, "n_chord": nc} for nr, nc in stable_lattices
            ],
            "stable_at_every_lattice": bool(len(stable_lattices) == len(lattices)),
        }

    chosen = None
    for model, length, steps in WAKE_FIDELITY:
        key = (model, length, steps)
        if stability.get(key, {}).get("stable_at_every_lattice"):
            chosen = key
            break
    if chosen is None:
        raise RuntimeError(
            "no wake setting is stable at every lattice level of the probe grid"
        )
    candidates = [
        lattice
        for lattice in stability[chosen]["stable_lattices"]
        if lattice["n_radial"] * lattice["n_chord"] * B4_70_CONFIG.blades <= panel_budget
    ]
    if not candidates:
        raise RuntimeError("no lattice within the panel budget is stable")
    best = max(candidates, key=lambda item: item["n_radial"] * item["n_chord"])
    return {
        "wake": {"wake_model": chosen[0], "wake_length": chosen[1], "wake_steps": chosen[2]},
        "mesh": {"n_radial": best["n_radial"], "n_chord": best["n_chord"]},
        "panel_budget": panel_budget,
        "probe_lattices": [{"n_radial": nr, "n_chord": nc} for nr, nc in lattices],
        "wake_fidelity_order": [
            {"wake_model": m, "wake_length": l, "wake_steps": st}
            for m, l, st in WAKE_FIDELITY
        ],
        "wake_stability_by_setting": [
            {
                "wake_model": key[0],
                "wake_length": key[1],
                "wake_steps": key[2],
                **value,
            }
            for key, value in sorted(stability.items())
        ],
        "unstable_combinations": [
            {
                "n_radial": entry["n_radial"],
                "n_chord": entry["n_chord"],
                "wake_model": entry["wake_model"],
                "wake_length": entry["wake_length"],
                "unstable_points": entry["unstable_points"],
            }
            for entry in probe["combinations"]
            if not entry["stable_at_all_J"]
        ],
    }


# ---------------------------------------------------------------------------
# Shapley decomposition (exact, over binary fidelity levels)
# ---------------------------------------------------------------------------


def shapley_binary(values: dict, factors: list) -> dict:
    """Shapley values of a binary factor set from all ``2**k`` coalition values.

    ``values`` is keyed by ``frozenset`` of *improved* factor names.  The
    identity ``sum(phi) == values[all] - values[none]`` holds exactly; it is the
    C3 exactness check.
    """

    n = len(factors)
    phi = {}
    for factor in factors:
        others = [f for f in factors if f != factor]
        total = 0.0
        for k in range(len(others) + 1):
            weight = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            for subset in itertools.combinations(others, k):
                s = frozenset(subset)
                total += weight * (values[s | {factor}] - values[s])
        phi[factor] = total
    return phi


def _deviation(config: Config, advance: float, quantity: str, anchor_value: float) -> float:
    return solve(config, advance)[quantity] - anchor_value


def decompose_b4_70(advance: float, series: B4Series, mesh_fine: dict | None = None,
                    wake_fine: dict | None = None) -> dict:
    """Three-factor Shapley decomposition of the B4-70 deviation at one J.

    Factor levels (predeclared):

    ``mesh``      (8 x 1 lattice) -> (16 x 3 lattice)   [both stable, see stability_probe]
    ``wake``      geometric helix, L = 1.0 D, 24 steps -> induced helix, L = 2.0 D, 48
    ``friction``  inviscid -> ITTC-78 section drag

    The value function is the signed deviation from the anchor, so the Shapley
    values are contributions *to the deviation* in coefficient units, and
    ``v(none)`` is the model-form remainder.
    """

    mesh_fine = dict(MESH_FINE if mesh_fine is None else mesh_fine)
    wake_fine = dict(WAKE_FINE if wake_fine is None else wake_fine)
    factors = ["mesh", "wake", "friction"]
    levels = {
        "mesh": (dict(n_radial=8, n_chord=1), mesh_fine),
        "wake": (dict(WAKE_BASE), wake_fine),
        "friction": (dict(FRICTION_BASE), dict(FRICTION_FINE)),
    }
    out = {"J": advance, "factors": {}}
    for quantity, anchor_value in (
        ("KT", series.kt(advance)),
        ("10KQ", 10.0 * series.kq(advance)),
    ):
        values = {}
        for size in range(len(factors) + 1):
            for subset in itertools.combinations(factors, size):
                s = frozenset(subset)
                overrides = {}
                for factor in factors:
                    overrides.update(levels[factor][1 if factor in s else 0])
                values[s] = _deviation(replace(B4_70_CONFIG, **overrides), advance,
                                       quantity, anchor_value)
        phi = shapley_binary(values, factors)
        total = values[frozenset(factors)] - values[frozenset()]
        out["factors"][quantity] = {
            "anchor_value": float(anchor_value),
            "deviation_fine": float(values[frozenset(factors)]),
            "deviation_pct_fine": float(100.0 * values[frozenset(factors)] / anchor_value),
            "deviation_coarse_inviscid": float(values[frozenset()]),
            "contributions": {k: float(v) for k, v in phi.items()},
            "contributions_pct_of_anchor": {
                k: float(100.0 * v / anchor_value) for k, v in phi.items()
            },
            "sum_contributions": float(sum(phi.values())),
            "sum_minus_total": float(sum(phi.values()) - total),
            "model_form_remainder": float(values[frozenset()]),
            "model_form_remainder_pct_of_anchor": float(
                100.0 * values[frozenset()] / anchor_value
            ),
            "coalition_values": {
                "+".join(sorted(s)) if s else "none": float(v) for s, v in values.items()
            },
        }
    return out


# ---------------------------------------------------------------------------
# KP505 attribution (the historically recorded ~29 % deviation)
# ---------------------------------------------------------------------------


def _kp505_anchor_series() -> dict:
    """KP505 anchor table as plain floats (machine copy of the module table)."""

    return {
        float(j): {"KT": float(v["KT"]), "10KQ": float(v["10KQ"])}
        for j, v in KP505_ANCHOR.items()
    }


def _kp505_j0_from_anchor() -> dict:
    """Zero-thrust advance ratio by linear extrapolation of the last two rows."""

    rows = sorted(_kp505_anchor_series().items())
    (j1, v1), (j2, v2) = rows[-2], rows[-1]
    slope = (v2["KT"] - v1["KT"]) / (j2 - j1)
    return {
        "J0": j1 - v1["KT"] / slope,
        "method": (
            "linear extrapolation of the last two anchor rows of Table 2 "
            f"(J={j1}, KT={v1['KT']}; J={j2}, KT={v2['KT']}); the table has no "
            "measured zero-thrust point"
        ),
        "slope_dKT_dJ": slope,
    }


KP505_CONFIG = Config(
    blades=5,
    expanded_area_ratio=0.80,
    pitch_ratio=0.95,
    hub_ratio=0.18,
    thickness_ratio=0.06,
    root_chord_offset=0.12,
    n_radial=16,
    n_chord=2,
    wake_model="geometric",
    wake_length=1.0,
    wake_steps=24,
    viscous=True,
    rev_per_second=10.0,
    re_match=False,
)


def _bem_zero_thrust(config: Config, j_hi: float = 2.0, tol: float = 2e-3) -> dict:
    """Zero-thrust advance ratio of the lattice kernel (bisection on K_T)."""

    lo, hi = 0.0, j_hi
    evaluations = 0
    kt_lo = solve(config, lo)["KT"]
    evaluations += 1
    if kt_lo <= 0.0:
        return {"J0": 0.0, "evaluations": evaluations, "converged": True}
    kt_hi = solve(config, hi)["KT"]
    evaluations += 1
    if kt_hi > 0.0:
        return {"J0": float("nan"), "evaluations": evaluations, "converged": False}
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        kt_mid = solve(config, mid)["KT"]
        evaluations += 1
        if kt_mid > 0.0:
            lo = mid
        else:
            hi = mid
    return {"J0": 0.5 * (lo + hi), "evaluations": evaluations, "converged": True}


def _calibrate_pitch(config: Config, target_j0: float, lo: float = 0.60,
                     hi: float = 1.40, iterations: int = 8) -> dict:
    """Pitch ratio whose BEM zero-thrust advance ratio matches ``target_j0``.

    The zero-thrust advance ratio is the one point of the open-water curve that
    is (almost) purely geometric: it says where the induced velocity balances
    the axial inflow.  Matching it isolates the *geometric* part of the model
    error from the friction and wake parts, and the target is derived from the
    anchor -- not tuned so that a coefficient looks good.
    """

    history = []
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        j0 = _bem_zero_thrust(replace(config, pitch_ratio=mid))["J0"]
        history.append({"P_D": mid, "J0_bem": j0})
        if not np.isfinite(j0):
            hi = mid
            continue
        if j0 < target_j0:
            lo = mid
        else:
            hi = mid
    pitch = 0.5 * (lo + hi)
    return {
        "pitch_ratio": pitch,
        "target_J0": target_j0,
        "history": history,
        "bracket": [lo, hi],
        "residual_J0": history[-1]["J0_bem"] - target_j0,
    }


def decompose_kp505(advance: float, calibrated_pitch: float) -> dict:
    """Shapley decomposition of the KP505 deviation into geometry / wake / friction."""

    factors = ["geometry", "wake", "friction"]
    levels = {
        "geometry": (
            dict(pitch_ratio=KP505_CONFIG.pitch_ratio),
            dict(pitch_ratio=calibrated_pitch),
        ),
        "wake": (dict(WAKE_BASE), dict(WAKE_FINE)),
        "friction": (dict(FRICTION_BASE), dict(FRICTION_FINE)),
    }
    anchor = _kp505_anchor_series()[advance]
    out = {"J": advance, "factors": {}}
    for quantity, anchor_key in (("KT", "KT"), ("10KQ", "10KQ")):
        anchor_value = anchor[anchor_key]
        values = {}
        for size in range(len(factors) + 1):
            for subset in itertools.combinations(factors, size):
                s = frozenset(subset)
                overrides = {}
                for factor in factors:
                    overrides.update(levels[factor][1 if factor in s else 0])
                values[s] = _deviation(replace(KP505_CONFIG, **overrides), advance,
                                       quantity, anchor_value)
        phi = shapley_binary(values, factors)
        total = values[frozenset(factors)] - values[frozenset()]
        out["factors"][quantity] = {
            "anchor_value": float(anchor_value),
            "deviation_best": float(values[frozenset(factors)]),
            "deviation_worst": float(values[frozenset()]),
            "deviation_pct_best": float(100.0 * values[frozenset(factors)] / anchor_value),
            "deviation_pct_worst": float(100.0 * values[frozenset()] / anchor_value),
            "contributions": {k: float(v) for k, v in phi.items()},
            "contributions_pct_of_anchor": {
                k: float(100.0 * v / anchor_value) for k, v in phi.items()
            },
            "sum_contributions": float(sum(phi.values())),
            "sum_minus_total": float(sum(phi.values()) - total),
            "model_form_remainder": float(values[frozenset()]),
            "model_form_remainder_pct_of_anchor": float(
                100.0 * values[frozenset()] / anchor_value
            ),
            "coalition_values": {
                "+".join(sorted(s)) if s else "none": float(v) for s, v in values.items()
            },
        }
    return out


def pitch_dof_identifiability(pitches, advance: float = 0.8) -> dict:
    """Can one pitch value fix the geometry?  (Two-equation identifiability test.)

    Two anchor-derived targets are used: the thrust deviation at ``advance`` and
    the zero-thrust advance ratio.  If a single geometric degree of freedom
    (constant pitch) can satisfy both, there exists a pitch where both residuals
    vanish; if not, the geometry error is *structural* (radial pitch and camber
    distribution, i.e. the shape of the curve), not a scalar offset.
    """

    anchor = _kp505_anchor_series()[advance]
    j0 = _kp505_j0_from_anchor()["J0"]
    rows = []
    for pitch in pitches:
        config = replace(KP505_CONFIG, pitch_ratio=pitch)
        point = solve(config, advance)
        zero = _bem_zero_thrust(config)
        rows.append(
            {
                "P_D": float(pitch),
                "KT": point["KT"],
                "dKT_pct": 100.0 * (point["KT"] - anchor["KT"]) / anchor["KT"],
                "J0_bem": zero["J0"],
                "dJ0_pct": 100.0 * (zero["J0"] - j0) / j0,
            }
        )
    # best single-dof compromise: minimise the largest of the two absolute residuals
    best = min(rows, key=lambda r: max(abs(r["dKT_pct"]), abs(r["dJ0_pct"])))
    return {
        "advance": advance,
        "targets": {
            "KT_anchor": anchor["KT"],
            "J0_anchor": j0,
            "J0_method": _kp505_j0_from_anchor()["method"],
        },
        "rows": rows,
        "best_single_dof": {
            "P_D": best["P_D"],
            "dKT_pct": best["dKT_pct"],
            "dJ0_pct": best["dJ0_pct"],
            "worst_abs_pct": max(abs(best["dKT_pct"]), abs(best["dJ0_pct"])),
        },
        "single_dof_sufficient_2pct": bool(
            max(abs(best["dKT_pct"]), abs(best["dJ0_pct"])) < 2.0
        ),
    }


def historical_deviation_check(advance: float = 0.8) -> dict:
    """Re-derive the deviation logged by the repository's earlier mission.

    The number is *machine-extracted* from the earlier report (regex over the
    file, sha256 recorded) rather than retyped, and then recomputed from the
    base configuration of this study.
    """

    repo_root = Path(__file__).resolve().parents[3]
    report = repo_root / "papers" / "reports" / "w20_propeller_openwater_bem5.md"
    anchor = _kp505_anchor_series()[advance]
    point = solve(KP505_CONFIG, advance)
    measured = 100.0 * (point["KT"] - anchor["KT"]) / anchor["KT"]
    entry = {
        "advance": advance,
        "measured_pct": measured,
        "measured_KT": point["KT"],
        "anchor_KT": anchor["KT"],
        "base_config": asdict(KP505_CONFIG),
    }
    if report.exists():
        text = report.read_text()
        matches = re.findall(r"-?\d+\.\d+\s?%", text)
        needle = re.findall(r"-29\.\d+\s?%", text)
        entry.update(
            {
                "report_path": str(report.relative_to(repo_root)),
                "report_sha256": _sha256(report),
                "reported_tokens_matching_29_pct": sorted(set(needle))[:5],
                "n_percent_tokens_in_report": len(matches),
            }
        )
        if needle:
            reported = float(needle[0].replace("%", "").strip())
            entry["reported_pct"] = reported
            entry["delta_pct_points"] = measured - reported
    return entry


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - provenance nicety only
        return "unknown"


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------


def _criteria(rows, config_label: str) -> list:
    in_range = [row for row in rows if row["in_criterion_range"]]
    usable = [
        row for row in in_range
        if row["anchor_KT_positive"] and row["anchor_KQ_positive"]
    ]
    excluded = [row["J"] for row in in_range if row["KT_anchor"] <= 0.0 or row["KQ_anchor"] <= 0.0]

    def worst(key):
        scored = [row for row in usable if np.isfinite(row[key])]
        return max(scored, key=lambda row: abs(row[key]))

    kt, kq, eta = worst("dKT_pct"), worst("dKQ_pct"), worst("deta0_pct")
    return [
        {
            "id": "C1",
            "statement": (
                f"every point with J <= {CRITERION_J_MAX} has |dKT| < {KT_GATE_PCT}% "
                f"and |dKQ| < {KQ_GATE_PCT}% ({config_label})"
            ),
            "threshold_pct": KT_GATE_PCT,
            "n_points_scored": len(usable),
            "scored_J": [row["J"] for row in usable],
            "excluded_points_outside_anchor_positive_range": excluded,
            "worst_dKT": {"J": kt["J"], "pct": kt["dKT_pct"], "abs_pct": abs(kt["dKT_pct"])},
            "worst_dKQ": {"J": kq["J"], "pct": kq["dKQ_pct"], "abs_pct": abs(kq["dKQ_pct"])},
            "passed": bool(
                abs(kt["dKT_pct"]) < KT_GATE_PCT and abs(kq["dKQ_pct"]) < KQ_GATE_PCT
            ),
        },
        {
            "id": "C2",
            "statement": f"every scored point has |d eta_0| < {ETA_GATE_PCT}%",
            "threshold_pct": ETA_GATE_PCT,
            "n_points_scored": len(usable),
            "worst_deta0": {
                "J": eta["J"], "pct": eta["deta0_pct"], "abs_pct": abs(eta["deta0_pct"])
            },
            "passed": bool(abs(eta["deta0_pct"]) < ETA_GATE_PCT),
        },
    ]


def _wake_family_summary(wake_rows) -> list:
    """Per-J comparison of the wake models at the base lattice.

    Rows where a non-geometric helix broke down numerically are kept with
    ``numerically_stable = False`` so the reader sees the failure instead of a
    silently missing family member.
    """

    grouped = {}
    for row in wake_rows:
        grouped.setdefault(row["J"], []).append(
            {
                "wake_model": row["wake_model"],
                "wake_length": row["wake_length"],
                "wake_steps": row["wake_steps"],
                "KT": row["KT"],
                "10KQ": row["10KQ"],
                "dKT_pct": row["dKT_pct"],
                "d10KQ_pct": row["d10KQ_pct"],
                "numerically_stable": row["numerically_stable"],
            }
        )
    out = []
    for advance in sorted(grouped):
        family = sorted(grouped[advance], key=lambda r: (r["wake_model"], r["wake_length"]))
        stable_family = [r for r in family if r["numerically_stable"]]
        spread = (
            max(r["dKT_pct"] for r in stable_family)
            - min(r["dKT_pct"] for r in stable_family)
            if len(stable_family) > 1
            else float("nan")
        )
        out.append(
            {
                "J": advance,
                "family": family,
                "n_stable": len(stable_family),
                "dKT_pct_spread_across_stable_models": spread,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Artifact assembly
# ---------------------------------------------------------------------------


def build_artifact(outdir: Path | None = None, quick: bool = False) -> dict:
    series = b4_70(1.0)
    anchor_data = anchor_payload()

    grid = PRIMARY_GRID[:3] + PRIMARY_GRID[-1:] if quick else PRIMARY_GRID

    # Pre-sweep stability envelope: it decides which lattice levels are admissible
    # and is recorded in the artifact with its numbers.
    probe = stability_probe()
    selection = select_fine_configuration(probe)
    fine_config = replace(
        B4_70_CONFIG, **selection["mesh"], **selection["wake"], **FRICTION_FINE
    )
    selection["fine_config"] = asdict(fine_config)
    selection["matches_declared_constants"] = bool(
        selection["mesh"] == MESH_FINE and selection["wake"] == WAKE_FINE
    )

    rows_base = sweep(B4_70_BASE, grid, series)
    rows_fine = sweep(fine_config, grid, series)
    rows_round = [] if quick else sweep(B4_70_BASE, ROUND_GRID, series)

    mesh_rows = []
    for level in MESH_LADDER:
        config = replace(B4_70_CONFIG, **level)
        for row in sweep(config, grid, series):
            row.update({"axis": "mesh", "level": dict(level)})
            mesh_rows.append(row)

    wake_rows = []
    for level in WAKE_LADDER:
        config = replace(B4_70_CONFIG, **level)
        for row in sweep(config, grid, series):
            row.update(
                {
                    "axis": "wake",
                    "level": dict(level),
                    "numerically_stable": bool(
                        np.isfinite(row["KT"])
                        and np.isfinite(row["KQ"])
                        and abs(row["KT"]) <= KT_PLAUSIBLE_MAX
                        and abs(row["KQ"]) <= KQ_PLAUSIBLE_MAX
                    ),
                }
            )
            wake_rows.append(row)

    friction_rows = []
    for level in (FRICTION_BASE, FRICTION_FINE):
        config = replace(B4_70_CONFIG, **level)
        for row in sweep(config, grid, series):
            row.update({"axis": "friction", "level": dict(level)})
            friction_rows.append(row)

    j0_anchor = series.j0
    j0_bem_base = _bem_zero_thrust(replace(B4_70_CONFIG, **MESH_BASE))
    j0_bem_fine = _bem_zero_thrust(
        replace(B4_70_CONFIG, **selection["mesh"], **selection["wake"]), tol=1e-3
    )

    decompositions = [
        decompose_b4_70(j, series, mesh_fine=selection["mesh"], wake_fine=selection["wake"])
        for j in grid
    ]
    exactness = []
    for entry in decompositions:
        for quantity, block in entry["factors"].items():
            scale = max(
                abs(block["deviation_fine"]),
                abs(block["model_form_remainder"]),
                1e-12,
            )
            exactness.append(
                {
                    "J": entry["J"],
                    "quantity": quantity,
                    "sum_minus_total": block["sum_minus_total"],
                    "scale": scale,
                    "relative_exactness": abs(block["sum_minus_total"]) / scale,
                }
            )
    c3_passed = all(check["relative_exactness"] < 1e-9 for check in exactness)

    # ---- KP505 ----------------------------------------------------------
    kp_anchor = _kp505_anchor_series()
    kp_j0_anchor = _kp505_j0_from_anchor()
    kp_calibration = _calibrate_pitch(KP505_CONFIG, kp_j0_anchor["J0"])
    kp_j0_bem = _bem_zero_thrust(KP505_CONFIG)
    kp_attr_08 = decompose_kp505(0.8, kp_calibration["pitch_ratio"])
    kp_attr_design = decompose_kp505(KP505_DESIGN_ADVANCE, kp_calibration["pitch_ratio"])

    kp_band = []
    for name, overrides in {
        "baseline": {},
        "root_chord_offset_0.06": {"root_chord_offset": 0.06},
        "root_chord_offset_0.20": {"root_chord_offset": 0.20},
        "t_over_c_0.04": {"thickness_ratio": 0.04},
        "t_over_c_0.08": {"thickness_ratio": 0.08},
        "pitch_0.90": {"pitch_ratio": 0.90},
        "pitch_1.00": {"pitch_ratio": 1.00},
        "hub_0.20": {"hub_ratio": 0.20},
    }.items():
        row = solve(replace(KP505_CONFIG, **overrides), 0.8)
        anchor = kp_anchor[0.8]
        kp_band.append(
            {
                "variant": name,
                "overrides": overrides,
                "KT": row["KT"],
                "dKT_pct": 100.0 * (row["KT"] - anchor["KT"]) / anchor["KT"],
                "10KQ": row["10KQ"],
                "d10KQ_pct": 100.0 * (row["10KQ"] - anchor["10KQ"]) / anchor["10KQ"],
            }
        )

    criteria = _criteria(rows_fine, "highest-fidelity configuration") + [
        {
            "id": "C3",
            "statement": (
                "ablation axes quantified and additive: sum(phi) == v(all) - v(none) "
                "to 1e-9 relative, with the model-form remainder reported"
            ),
            "max_relative_exactness_error": max(c["relative_exactness"] for c in exactness),
            "n_checks": len(exactness),
            "passed": bool(c3_passed),
        }
    ]

    repo_root = Path(__file__).resolve().parents[3]
    anchor_file = repo_root / "benchmarks" / "anchors" / "wageningen_b_series.json"
    kp_source = (
        repo_root / "benchmarks" / "anchors" / "sources" / "kp505_marine2023_p10_table2.txt"
    )

    artifact = {
        "artifact_id": "w25-b4-70-error-surface",
        "mission": "paper-exp-propeller-openwater-bem-7",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "case": {
            "propeller": "Wageningen B4-70 (Z=4, Ae/Ao=0.70, P/D=1.00)",
            "geometry": B4_70_GEOMETRY_DECLARED,
            "configurations": {
                "base": asdict(B4_70_BASE),
                "fine": asdict(fine_config),
            },
            "advance_grid": {
                "primary": grid,
                "primary_spec": "10 evenly spaced points on J = 0.1 .. 1.2",
                "supplementary_round": ROUND_GRID,
                "criterion_range_J_max": CRITERION_J_MAX,
            },
            "reynolds_matching": {
                "target": ANCHOR_RE,
                "rule": "n per point from Re = W_nd n c_nd D^2 / nu at r = 0.75 R",
                "achieved": [
                    {
                        "J": row["J"],
                        "rev_per_second": row["rev_per_second"],
                        "reynolds_075R": row["reynolds_075R"],
                    }
                    for row in rows_fine
                ],
            },
        },
        "anchor": {
            "id": anchor_data["anchor_id"],
            "citation": anchor_data["citation"],
            "provenance": anchor_data["provenance"],
            "fingerprints": anchor_data["fingerprints"],
            "artifact_path": str(anchor_file.relative_to(repo_root)),
            "artifact_sha256": _sha256(anchor_file),
            "integrity": verify_tables(),
            "zero_thrust_advance_ratio": j0_anchor,
            "scoring_range": series.scoring_range(),
        },
        "criteria": criteria,
        "error_surface": {
            "base_configuration": rows_base,
            "fine_configuration": rows_fine,
            "round_grid_base": rows_round,
        },
        "ablation": {"mesh": mesh_rows, "wake": wake_rows, "friction": friction_rows},
        "stability_envelope": {
            "probe": probe,
            "selection": selection,
            "fine_config_all_points_physical": bool(
                all(row["physical"] for row in rows_fine)
            ),
            "wake_model_family_at_base_lattice": _wake_family_summary(wake_rows),
        },
        "zero_thrust": {
            "anchor": j0_anchor,
            "bem_base": j0_bem_base,
            "bem_fine": j0_bem_fine,
            "dJ0_pct_base": 100.0 * (j0_bem_base["J0"] - j0_anchor) / j0_anchor,
            "dJ0_pct_fine": 100.0 * (j0_bem_fine["J0"] - j0_anchor) / j0_anchor,
        },
        "shapley_decomposition_b4_70": {
            "factors": {
                "mesh": {
                    "coarse": {"n_radial": 8, "n_chord": 1},
                    "fine": selection["mesh"],
                },
                "wake": {"coarse": WAKE_BASE, "fine": selection["wake"]},
                "friction": {"coarse": FRICTION_BASE, "fine": FRICTION_FINE},
            },
            "exactness_checks": exactness,
            "rows": decompositions,
        },
        "kp505": {
            "anchor": {
                "source": KP505_ANCHOR_SOURCE,
                "geometry": KP505_GEOMETRY_SOURCE,
                "design_advance_ratio": KP505_DESIGN_ADVANCE,
                "table": {str(j): v for j, v in sorted(kp_anchor.items())},
                "source_file": str(kp_source.relative_to(repo_root)),
                "source_file_sha256": _sha256(kp_source),
                "zero_thrust_extrapolation": kp_j0_anchor,
            },
            "config": asdict(KP505_CONFIG),
            "bem_zero_thrust": kp_j0_bem,
            "geometry_calibration": kp_calibration,
            "decomposition_J_0.8": kp_attr_08,
            "decomposition_design_J": kp_attr_design,
            "geometry_sensitivity_band_J_0.8": kp_band,
            "pitch_dof_identifiability": pitch_dof_identifiability(
                [0.90, 0.95, 1.00, 1.05, round(kp_calibration["pitch_ratio"], 4)]
            ),
            "historical_deviation_check": historical_deviation_check(0.8),
            "design_advance_note": (
                "the design advance of the KP505 anchor module is the argmax of the "
                f"anchor efficiency column, which is J = {KP505_DESIGN_ADVANCE}; it "
                "coincides with the advance of the historically logged deviation, so "
                "'decomposition_design_J' and 'decomposition_J_0.8' are the same point"
                if abs(KP505_DESIGN_ADVANCE - 0.8) < 1e-12
                else "design advance differs from J = 0.8"
            ),
        },
        "provenance": {
            "command": " ".join([sys.executable, "-m", "tensorbem.benchmarks.b4_70_error_surface"]),
            "repo_root": str(repo_root),
            "git_commit": _git_commit(repo_root),
            "git_branch": subprocess.run(
                ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True,
            ).stdout.strip(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "inputs": {
                "wageningen_anchor": {
                    "path": str(anchor_file.relative_to(repo_root)),
                    "sha256": _sha256(anchor_file),
                },
                "kp505_anchor_table": {
                    "path": str(kp_source.relative_to(repo_root)),
                    "sha256": _sha256(kp_source),
                },
                "kernel_module": "src/tensorbem/propeller.py",
                "kernel_sha256": _sha256(repo_root / "src" / "tensorbem" / "propeller.py"),
                "driver_sha256": _sha256(Path(__file__)),
            },
        },
    }
    return artifact


# ---------------------------------------------------------------------------
# CSV export + CLI
# ---------------------------------------------------------------------------

SURFACE_FIELDS = [
    "J", "KT", "KQ", "10KQ", "eta0", "KT_anchor", "KQ_anchor", "10KQ_anchor",
    "eta0_anchor", "dKT_pct", "dKQ_pct", "d10KQ_pct", "deta0_pct",
    "dKT_norm_by_KT0", "dKQ_norm_by_KQ0", "n_radial", "n_chord", "wake_model",
    "wake_length", "wake_steps", "viscous", "rev_per_second", "reynolds_075R",
    "physical", "in_criterion_range",
]


def write_csv(path: Path, rows, fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="W25 B4-70 open-water error surface")
    parser.add_argument(
        "--outdir",
        default="papers/artifacts/w25/propeller-openwater-bem",
        help="artifact directory (relative to the repository root)",
    )
    parser.add_argument("--quick", action="store_true", help="smoke run on 4 of 10 J points")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[3]
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = repo_root / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    artifact = build_artifact(outdir=outdir, quick=args.quick)
    json_path = outdir / "b4_70_error_surface.json"
    json_path.write_text(json.dumps(artifact, indent=2) + "\n")

    write_csv(
        outdir / "b4_70_error_surface_points.csv",
        artifact["error_surface"]["fine_configuration"],
        SURFACE_FIELDS,
    )
    write_csv(
        outdir / "b4_70_error_surface_base_points.csv",
        artifact["error_surface"]["base_configuration"],
        SURFACE_FIELDS,
    )
    ablation_fields = SURFACE_FIELDS + ["axis", "level"]
    write_csv(outdir / "b4_70_ablation_mesh.csv", artifact["ablation"]["mesh"], ablation_fields)
    write_csv(outdir / "b4_70_ablation_wake.csv", artifact["ablation"]["wake"], ablation_fields)
    write_csv(
        outdir / "b4_70_ablation_friction.csv",
        artifact["ablation"]["friction"],
        ablation_fields,
    )

    def _show(criteria):
        for c in criteria:
            print(f"  {c['id']}: {'PASS' if c['passed'] else 'FAIL'}  {c['statement']}")

    print(f"wrote {json_path}")
    print("criteria (B4-70, highest-fidelity configuration):")
    _show(artifact["criteria"])
    print(
        "zero thrust: anchor "
        f"{artifact['zero_thrust']['anchor']:.4f}, BEM base "
        f"{artifact['zero_thrust']['bem_base']['J0']:.4f} "
        f"({artifact['zero_thrust']['dJ0_pct_base']:+.2f}%), BEM fine "
        f"{artifact['zero_thrust']['bem_fine']['J0']:.4f} "
        f"({artifact['zero_thrust']['dJ0_pct_fine']:+.2f}%)"
    )
    kp = artifact["kp505"]
    kt = kp["decomposition_J_0.8"]["factors"]["KT"]
    print(
        f"KP505 J=0.8 KT: worst config {kt['deviation_pct_worst']:+.2f}%, "
        f"best {kt['deviation_pct_best']:+.2f}%, contributions "
        + ", ".join(f"{k}={v:+.2f}pt" for k, v in kt["contributions_pct_of_anchor"].items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())