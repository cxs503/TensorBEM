"""Executable gates for every number published in ``docs/finite_depth3d.md``.

Why this module exists
----------------------
The same accident has now happened four times in this repository: a document
quotes a number, no test reads it, and a later refactor of the kernel silently
invalidates the document (``7.7e-3`` -> ``8.5e-3`` for the Ursell added mass,
``1.9e-4`` -> ``7.6e-4`` for the MFS cross-check, the Wigley displacement
``2LBT/3`` -> ``4LBT/9``, and the 3-D declared cells of
``docs/three_dimensional_bem.md``).  Guards now exist for
``docs/marine_bem_benchmarks.md`` (:mod:`tensorbem.benchmarks.doc_number_ledger`),
``README.md`` (:mod:`tensorbem.benchmarks.doc_pins`),
``docs/three_dimensional_bem.md``
(:mod:`tensorbem.benchmarks.three_dimensional_doc_pins`) and the Ursell table
(:mod:`tensorbem.doc_claims`) — but ``docs/finite_depth3d.md``, the entire
verification face of the 3-D finite-depth John-eigenseries kernel
(:mod:`tensorbem.finite_depth3d`), was still **declared but unguarded**: its six
tables and its §5 verdict are hand-free numbers written by
``docs/m3015/finite_depth3d_report.py``, and nothing failed if the kernel moved.

What is pinned
--------------
Every table cell and every verdict number of the document is re-measured here
from the library and judged against its *printed* precision (half-ulp of the
last written digit, rounded up by :data:`PRINT_SAFETY`):

* ``value``    — the document states a computed value; it must reproduce.
* ``bound``    — the document states an upper bound (a residual, a deviation);
  reproducing the same order of magnitude is what the claim means, so the
  measured value must stay below the printed bound times :data:`BOUND_SLACK`.
* ``zero``     — the column is exactly zero by construction (the bottom
  condition annihilates every vertical mode at ``z=-h``); measured must stay at
  roundoff.

Four claims are additionally checked against **references that share no code
path with the quantity under test**, so a self-consistent bug cannot hide:

* the propagating wavenumber ``k0`` against an independent bisection solve of
  ``k tanh(kh) = nu`` (the module's own Newton polish is not reused);
* the group velocity ``c_g`` against a central-difference derivative of
  ``omega(k) = sqrt(g k tanh(kh))`` (the document's closed form is not reused);
* the deep-water collapse against the independent Kelvin kernel
  :func:`tensorbem.free_surface3d.green_function_3d`;
* the far-field phase slope against the analytic value 1 and the singularity
  strength against the analytic value 1 (``4 pi r G -> 1``).

Usage::

    PYTHONPATH=src python3 -m tensorbem.benchmarks.finite_depth3d_pins --check
    PYTHONPATH=src python3 -m tensorbem.benchmarks.finite_depth3d_pins --write
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from tensorbem.finite_depth import finite_depth_wavenumber
from tensorbem.finite_depth3d import (
    depth_modes,
    dispersion_residual,
    green_function_3d,
    green_gradient_z_3d,
)
from tensorbem.free_surface3d import green_function_3d as kelvin_green_3d

REPO = Path(__file__).resolve().parents[3]
DOC = REPO / "docs" / "finite_depth3d.md"
PINS_FILE = REPO / "docs" / "finite_depth3d_pins.json"
STUDY_FILE = REPO / "docs" / "m3015" / "outputs" / "finite_depth3d" / "study.json"

NU = 0.6
G = 9.80665
OMEGA = math.sqrt(NU * G)

DEPTHS: tuple[float, ...] = (0.05, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 20.0, 100.0)
BC_DEPTHS: tuple[float, ...] = (2.0, 4.0, 10.0, 40.0)
DEEP_DEPTHS: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
TRUNC_MODES: tuple[int, ...] = (40, 80, 160, 320, 640)
SING_RADII = (4e-2, 2e-2, 1e-2, 5e-3, 2.5e-3)

#: Extra factor on the half-ulp of the last printed digit.
PRINT_SAFETY = 3.0
#: How far a stated upper bound (a roundoff-level residual) may grow.
BOUND_SLACK = 5.0
#: Absolute floor for the ``zero`` class (the columns that are exactly zero).
ZERO_FLOOR = 1e-25

NUMBER_RE = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|\d+(?:[eE][-+]?\d+)?")


# ---------------------------------------------------------------------------
# document parsing
# ---------------------------------------------------------------------------
def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_rule(row: str) -> bool:
    return set(row.replace("|", "").replace(" ", "")) <= {"-"}


def parse_tables(text: str) -> dict[str, tuple[list[str], list[list[str]]]]:
    """Return ``{name: (header, rows)}`` for the six tables of the document."""
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("|"):
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)

    signatures = {
        "dispersion": "k_0h",
        "boundary_conditions": "bottom residual",
        "deep_water_limit": "max relative deviation",
        "singularity": "pi r",
        "truncation": "max relative change",
    }
    tables: dict[str, tuple[list[str], list[list[str]]]] = {}
    for block in blocks:
        if len(block) < 2 or not _is_rule(block[1]):
            continue
        header = _cells(block[0])
        joined = " ".join(header)
        name = None
        for candidate, needle in signatures.items():
            if needle in joined:
                # ``k_0h`` appears in the dispersion table only once as a column
                if candidate == "dispersion" and "$k_0$ [1/m]" not in joined:
                    continue
                name = candidate
                break
        if name is None:
            continue
        rows = [_cells(row) for row in block[2:] if not _is_rule(row)]
        tables[name] = (header, rows)
    return tables


def parse_prose(text: str) -> dict[str, Any]:
    """Pull the numbers that the document states in running text."""
    prose: dict[str, Any] = {}

    def grab(pattern: str, cast=float, group: int = 1):
        match = re.search(pattern, text)
        return cast(match.group(group)) if match else None

    prose["spreading"] = grab(r"mean\} = ([0-9.]+e[-+]\d+)")
    prose["phase_slope"] = grab(r"\(k_0R\) = ([0-9.]+)")
    prose["evanescent"] = [
        (float(ratio), float(radius.rstrip(".")))
        for ratio, radius in re.findall(r"([0-9.]+e[-+]\d+) at \$k_0R\$=([0-9.]+)", text)
    ]
    prose["verdict_fs_residual"] = grab(r"free-surface condition to\s+([0-9.]+e[-+]\d+)")
    prose["verdict_deep_error"] = grab(r"max deviation ([0-9.]+e[-+]\d+)")
    prose["verdict_deep_k0h"] = grab(r"\$k_0h=([0-9.]+)\$", cast=float)
    prose["verdict_spreading"] = grab(r"spreading flatness ([0-9.]+e[-+]\d+)")
    prose["verdict_phase_slope"] = grab(r"phase slope\s+([0-9.]+)\.")
    prose["verdict_singularity_offset"] = grab(r"extrapolates to 1 within\s+([0-9.]+e[-+]\d+)")
    return prose


def printed_tolerance(cell: str, factor: float = PRINT_SAFETY) -> float:
    """Relative tolerance implied by the number of digits actually printed."""
    mantissa, exponent = re.split(r"[eE]", cell.strip(), maxsplit=1) + [] if re.search(
        r"[eE]", cell
    ) else (cell.strip(), "0")
    digits = len(mantissa.split(".")[1]) if "." in mantissa else 0
    magnitude = abs(float(mantissa))
    if magnitude == 0.0:
        return 0.0
    ulp = 10.0 ** (-digits)
    return factor * 0.5 * ulp / magnitude


# ---------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------
def independent_wavenumber(nu: float, depth: float, *, iterations: int = 200) -> float:
    """Bisection solve of ``k tanh(k h) = nu`` (no Newton polish, no library code)."""
    lo, hi = 1e-12, 1.0
    while hi * math.tanh(hi * depth) < nu:
        hi *= 2.0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if mid * math.tanh(mid * depth) < nu:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def omega_of_k(k: float, depth: float) -> float:
    """Dispersion ``omega(k)`` of the strip, ``k tanh(k h) = omega^2 / g``."""
    return math.sqrt(G * k * math.tanh(k * depth))


def measure() -> dict[str, Any]:
    """Recompute every pinned quantity from the library."""
    out: dict[str, Any] = {"dispersion": {}, "boundary_conditions": {}, "deep_water_limit": {},
                           "far_field": {}, "singularity": {}, "truncation": {}, "independent": {}}

    # --- §3.1 dispersion, phase and group velocities -----------------------
    for depth in DEPTHS:
        k0 = finite_depth_wavenumber(NU, depth)
        c = OMEGA / k0
        cg = 0.5 * (OMEGA / k0) * (1.0 + 2.0 * k0 * depth / math.sinh(2.0 * k0 * depth))
        out["dispersion"][depth] = {
            "k0": k0,
            "k0_h": k0 * depth,
            "c": c,
            "cg": cg,
            "residual": dispersion_residual(k0, NU, depth),
        }

    # --- §3.2 free-surface and bottom conditions ---------------------------
    targets_fs = np.stack([np.linspace(0.3, 9.0, 14), np.zeros(14), np.zeros(14)], axis=-1)
    for depth in BC_DEPTHS:
        source = np.array([0.0, 0.0, -0.5 * depth])
        value = green_function_3d(targets_fs, source, NU, depth)
        grad = green_gradient_z_3d(targets_fs, source, NU, depth)
        fs = float(np.max(np.abs(grad - NU * value)) / np.max(np.abs(NU * value)))
        bottom_points = np.stack(
            [np.linspace(0.4, 6.0, 10), np.zeros(10), -depth * np.ones(10)], axis=-1
        )
        bottom = float(
            np.max(np.abs(green_gradient_z_3d(bottom_points, source, NU, depth)))
            / (np.max(np.abs(green_function_3d(bottom_points, source, NU, depth))) / depth)
        )
        out["boundary_conditions"][depth] = {
            "free_surface_residual": fs,
            "bottom_residual": bottom,
        }

    # --- §3.3 deep-water collapse against the Kelvin kernel ----------------
    points = np.array(
        [[0.5, 0.0, -1.0], [2.0, 0.0, 0.0], [3.0, 1.0, -1.2], [6.0, 0.0, -0.5],
         [9.0, 0.0, -1.0]]
    )
    source = np.array([0.0, 0.0, -2.0])
    reference = kelvin_green_3d(points, source, NU)
    for depth in DEEP_DEPTHS:
        finite = green_function_3d(points, source, NU, depth)
        out["deep_water_limit"][depth] = {
            "k0_h": finite_depth_wavenumber(NU, depth) * depth,
            "max_rel_error": float(
                np.max(np.abs(finite - reference)) / np.max(np.abs(reference))
            ),
        }

    # --- §3.4 far field ---------------------------------------------------
    depth_ff = 4.0
    k0_ff = finite_depth_wavenumber(NU, depth_ff)
    radii = np.linspace(8.0, 90.0, 60)
    targets = np.stack([radii, np.zeros_like(radii), np.zeros_like(radii)], axis=-1)
    full = green_function_3d(targets, source, NU, depth_ff)
    evanescent = green_function_3d(targets, source, NU, depth_ff, include_wave=False)
    window = radii > 25.0
    amplitude = np.abs(full) * np.sqrt(k0_ff * radii)
    phase = np.unwrap(np.angle(full))
    out["far_field"] = {
        "k0": k0_ff,
        "spreading": float(np.ptp(amplitude[window]) / np.mean(amplitude[window])),
        "phase_slope": float(np.polyfit(k0_ff * radii[window], phase[window], 1)[0]),
        "evanescent": {
            float(k0_ff * radius): float(ratio)
            for radius, ratio in zip(radii, np.abs(evanescent) / np.abs(full))
        },
    }

    # --- §3.5 source singularity and truncation ---------------------------
    depth_s = 5.0
    radii_s = np.array(SING_RADII)
    targets_s = np.stack([radii_s, np.zeros_like(radii_s), -2.0 * np.ones_like(radii_s)], axis=-1)
    values_s = green_function_3d(targets_s, np.array([0.0, 0.0, -2.0]), NU, depth_s)
    for radius, value in zip(SING_RADII, values_s):
        out["singularity"][radius] = {"four_pi_r_G": float(4.0 * math.pi * radius * np.real(value))}

    probe = np.array([[0.7, 0.0, -0.4], [1.5, 1.0, -2.0]])
    reference_value = green_function_3d(probe, np.array([0.0, 0.0, -1.0]), NU, 4.0, modes=1200)
    for modes in TRUNC_MODES:
        approx = green_function_3d(probe, np.array([0.0, 0.0, -1.0]), NU, 4.0, modes=modes)
        out["truncation"][modes] = {
            "max_rel_change": float(
                np.max(np.abs(approx - reference_value)) / np.max(np.abs(reference_value))
            )
        }

    # --- independent references ------------------------------------------
    indep: dict[str, Any] = {"dispersion": {}}
    for depth in DEPTHS:
        row = out["dispersion"][depth]
        bisected = independent_wavenumber(NU, depth)
        cg_numeric = _numeric_cg(bisected, depth)
        indep["dispersion"][depth] = {
            "k0_bisected": bisected,
            "cg_derivative": cg_numeric,
            "k0_rel_gap": abs(bisected - row["k0"]) / row["k0"],
            "cg_rel_gap": abs(cg_numeric - row["cg"]) / row["cg"],
            "c_rel_gap": abs(OMEGA / row["k0"] - row["c"]) / row["c"],
        }
    indep["phase_slope_offset"] = abs(out["far_field"]["phase_slope"] - 1.0)
    indep["singularity_offset"] = abs(out["singularity"][SING_RADII[-1]]["four_pi_r_G"] - 1.0)
    indep["deep_water_monotone"] = all(
        out["deep_water_limit"][hi]["max_rel_error"]
        < out["deep_water_limit"][lo]["max_rel_error"]
        for lo, hi in zip(DEEP_DEPTHS, DEEP_DEPTHS[1:])
    )
    out["independent"] = indep
    return out


def _numeric_cg(k0: float, depth: float, *, step: float = 1e-6) -> float:
    """``d omega / d k`` by central difference; shares nothing with the doc formula."""
    return (omega_of_k(k0 * (1.0 + step), depth) - omega_of_k(k0 * (1.0 - step), depth)) / (
        2.0 * step * k0
    )


# ---------------------------------------------------------------------------
# claims
# ---------------------------------------------------------------------------
@dataclass
class Claim:
    """One published number and the recomputed value it must agree with."""

    claim_id: str
    section: str
    declared: float
    measured: float
    kind: str
    tolerance: float
    token: str
    note: str = ""

    @property
    def deviation(self) -> float:
        return abs(self.measured - self.declared)

    @property
    def relative(self) -> float:
        if self.declared == 0.0:
            return self.deviation
        return self.deviation / abs(self.declared)

    @property
    def passed(self) -> bool:
        if self.kind == "value":
            return self.relative <= self.tolerance
        if self.kind == "bound":
            return self.measured <= self.declared * BOUND_SLACK
        if self.kind == "zero":
            return abs(self.measured) <= ZERO_FLOOR
        raise ValueError(f"unknown claim kind {self.kind!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.claim_id,
            "section": self.section,
            "kind": self.kind,
            "declared": self.declared,
            "measured": self.measured,
            "relative": self.relative,
            "tolerance": self.tolerance,
            "verdict": "PASS" if self.passed else "DRIFT",
            "note": self.note,
        }


def _float(cell: str) -> Optional[float]:
    text = cell.replace("$", "").replace("*", "").strip()
    try:
        return float(text)
    except ValueError:
        match = NUMBER_RE.search(text)
        return float(match.group(0)) if match else None


def build_claims(
    text: str, measured: Mapping[str, Any]
) -> tuple[list[Claim], dict[str, Any]]:
    """Turn the parsed document into claims against the fresh measurements."""
    tables = parse_tables(text)
    prose = parse_prose(text)
    claims: list[Claim] = []

    def add(claim: Claim) -> None:
        claims.append(claim)

    # --- §3.1 dispersion --------------------------------------------------
    header, rows = tables["dispersion"]
    columns = {
        "k0": ("k_0$ [1/m]", "value"),
        "k0_h": ("k_0h$", "value"),
        "c": ("$c$ [m/s]", "value"),
        "cg": ("$c_g$ [m/s]", "value"),
        "residual": ("nu|/\\nu$", "bound"),
    }
    for row in rows:
        depth = _float(row[0])
        fresh = measured["dispersion"][depth]
        for column, value in fresh.items():
            cell = row[list(columns).index(column) + 1]
            declared = _float(cell)
            if declared is None:
                continue
            kind = "bound" if column == "residual" else "value"
            add(
                Claim(
                    claim_id=f"dispersion/h={depth:g}/{column}",
                    section="3.1",
                    declared=declared,
                    measured=float(value),
                    kind=kind,
                    tolerance=printed_tolerance(cell),
                    token=cell,
                )
            )

    # --- §3.2 boundary conditions ----------------------------------------
    _, rows = tables["boundary_conditions"]
    for row in rows:
        depth = _float(row[0])
        fresh = measured["boundary_conditions"][depth]
        for column, cell in (
            ("free_surface_residual", row[1]),
            ("bottom_residual", row[2]),
        ):
            declared = _float(cell)
            add(
                Claim(
                    claim_id=f"boundary_conditions/h={depth:g}/{column}",
                    section="3.2",
                    declared=declared if declared is not None else 0.0,
                    measured=float(fresh[column]),
                    kind="zero" if declared == 0.0 else "bound",
                    tolerance=printed_tolerance(cell),
                    token=cell,
                    note="bottom condition annihilates every vertical mode at z=-h"
                    if column == "bottom_residual"
                    else "",
                )
            )

    # --- §3.3 deep-water limit -------------------------------------------
    _, rows = tables["deep_water_limit"]
    for row in rows:
        depth = _float(row[0])
        fresh = measured["deep_water_limit"][depth]
        add(
            Claim(
                claim_id=f"deep_water_limit/h={depth:g}/k0_h",
                section="3.3",
                declared=_float(row[1]),
                measured=fresh["k0_h"],
                kind="value",
                tolerance=printed_tolerance(row[1]),
                token=row[1],
            )
        )
        add(
            Claim(
                claim_id=f"deep_water_limit/h={depth:g}/deviation",
                section="3.3",
                declared=_float(row[2]),
                measured=fresh["max_rel_error"],
                kind="bound",
                tolerance=printed_tolerance(row[2]),
                token=row[2],
                note="K0/hankel series vs the independent Kelvin kernel",
            )
        )

    # --- §3.4 far field prose --------------------------------------------
    add(
        Claim(
            claim_id="far_field/spreading",
            section="3.4",
            declared=prose["spreading"],
            measured=measured["far_field"]["spreading"],
            kind="bound",
            tolerance=0.0,
            token=f"{prose['spreading']:.3e}",
            note="flatness of |G| sqrt(k0 R) for k0 R > 25",
        )
    )
    add(
        Claim(
            claim_id="far_field/phase_slope",
            section="3.4",
            declared=prose["phase_slope"],
            measured=measured["far_field"]["phase_slope"],
            kind="value",
            tolerance=printed_tolerance(f"{prose['phase_slope']:.6f}"),
            token=f"{prose['phase_slope']:.6f}",
            note="d arg G / d(k0 R); analytic value 1",
        )
    )
    for ratio, radius in prose["evanescent"]:
        key = min(measured["far_field"]["evanescent"], key=lambda k: abs(k - radius))
        add(
            Claim(
                claim_id=f"far_field/evanescent/k0R={radius:g}",
                section="3.4",
                declared=ratio,
                measured=measured["far_field"]["evanescent"][key],
                kind="value",
                tolerance=0.02,
                token=f"{ratio:.1e}",
                note=f"printed to two significant digits; k0 R = {radius:g}",
            )
        )

    # --- §3.5 singularity and truncation ----------------------------------
    _, rows = tables["singularity"]
    for row in rows:
        radius = _float(row[0])
        add(
            Claim(
                claim_id=f"singularity/r={radius:g}",
                section="3.5",
                declared=_float(row[1]),
                measured=measured["singularity"][radius]["four_pi_r_G"],
                kind="value",
                tolerance=printed_tolerance(row[1]),
                token=row[1],
            )
        )
    _, rows = tables["truncation"]
    for row in rows:
        modes = int(_float(row[0]))
        declared = _float(row[1])
        add(
            Claim(
                claim_id=f"truncation/modes={modes}",
                section="3.5",
                declared=declared,
                measured=measured["truncation"][modes]["max_rel_change"],
                kind="zero" if declared == 0.0 else "bound",
                tolerance=printed_tolerance(row[1]),
                token=row[1],
                note="self-convergence against modes=1200",
            )
        )

    return claims, prose


def verdict_claims(prose: Mapping[str, Any], text: str) -> list[Claim]:
    """§5 must quote the extremes of §3 and §3.4 — the document against itself."""
    tables = parse_tables(text)
    _, bc_rows = tables["boundary_conditions"]
    _, deep_rows = tables["deep_water_limit"]
    _, sing_rows = tables["singularity"]
    fs_max = max(_float(row[1]) for row in bc_rows)
    deep_last = _float(deep_rows[-1][2])
    deep_last_k0h = _float(deep_rows[-1][1])
    sing_last = _float(sing_rows[-1][1])

    return [
        Claim("verdict/fs_residual", "5", fs_max, prose["verdict_fs_residual"], "value",
              1e-12, f"{prose['verdict_fs_residual']:.3e}", "§5 quotes max(§3.2)"),
        Claim("verdict/deep_error", "5", deep_last, prose["verdict_deep_error"], "value",
              1e-12, f"{prose['verdict_deep_error']:.3e}", "§5 quotes last(§3.3)"),
        Claim("verdict/deep_k0h", "5", deep_last_k0h, prose["verdict_deep_k0h"], "value",
              0.01, f"{prose['verdict_deep_k0h']:.0f}", "§5 quotes the k0h of the last row, "
              "printed with %g-like integer rounding"),
        Claim("verdict/singularity_offset", "5", abs(sing_last - 1.0),
              prose["verdict_singularity_offset"], "value", 1e-12,
              f"{prose['verdict_singularity_offset']:.3e}", "|§3.5 last - 1|"),
        Claim("verdict/spreading", "5", prose["spreading"], prose["verdict_spreading"],
              "value", 1e-12, f"{prose['verdict_spreading']:.3e}", "§5 quotes §3.4"),
    ]


def consistency_claims(text: str, measured: Mapping[str, Any]) -> list[Claim]:
    """Compare the document against references that share no code with the tested quantity."""
    tables = parse_tables(text)
    _, rows = tables["dispersion"]
    claims: list[Claim] = []
    for row in rows:
        depth = _float(row[0])
        indep = measured["independent"]["dispersion"][depth]
        claims.append(
            Claim(
                f"independent/dispersion/h={depth:g}/k0_vs_bisection",
                "3.1",
                _float(row[1]),
                indep["k0_bisected"],
                "value",
                max(printed_tolerance(row[1]), 1e-11),
                row[1],
                f"200-step bisection of k tanh(kh)=nu; gap {indep['k0_rel_gap']:.2e}",
            )
        )
        claims.append(
            Claim(
                f"independent/dispersion/h={depth:g}/cg_vs_derivative",
                "3.1",
                _float(row[4]),
                indep["cg_derivative"],
                "value",
                max(printed_tolerance(row[4]), 1e-9),
                row[4],
                "central difference d omega/dk on omega=sqrt(g k tanh(kh)); "
                f"gap {indep['cg_rel_gap']:.2e}",
            )
        )
    claims.append(
        Claim(
            "independent/far_field/phase_slope_vs_one",
            "3.4",
            1.0,
            measured["far_field"]["phase_slope"],
            "value",
            1e-3,
            "",
            "analytic phase slope of the outgoing H0(1) wave is 1",
        )
    )
    claims.append(
        Claim(
            "independent/singularity/last_vs_one",
            "3.5",
            1.0,
            measured["singularity"][SING_RADII[-1]]["four_pi_r_G"],
            "value",
            2e-3,
            "",
            "analytic strength 4 pi r G -> 1",
        )
    )
    claims.append(
        Claim(
            "independent/deep_water_limit/monotone",
            "3.3",
            1.0,
            float(measured["independent"]["deep_water_monotone"]),
            "value",
            0.0,
            "",
            "deviation must decrease monotonically as k0 h grows",
        )
    )
    return claims


def coverage(text: str, claims: Sequence[Claim]) -> dict[str, Any]:
    """How much of the document's numeric content the registry actually reads."""
    tokens = NUMBER_RE.findall(text)
    protected = 0
    for claim in claims:
        if claim.token:
            protected += text.count(claim.token)
    return {
        "numeric_tokens_total": len(tokens),
        "protected_occurrences": protected,
        "protected_fraction": (protected / len(tokens)) if tokens else 0.0,
        "claims": len(claims),
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def evaluate(doc: Path = DOC) -> dict[str, Any]:
    text = doc.read_text()
    measured = measure()
    claims, prose = build_claims(text, measured)
    claims.extend(verdict_claims(prose, text))
    claims.extend(consistency_claims(text, measured))
    cov = coverage(text, claims)
    drift = [claim.as_dict() for claim in claims if not claim.passed]
    return {
        "document": str(doc.relative_to(REPO)) if doc.is_relative_to(REPO) else str(doc),
        "generated_by": "tensorbem.benchmarks.finite_depth3d_pins",
        "parameters": {"nu": NU, "gravity": G, "omega": OMEGA, "depths": list(DEPTHS),
                       "bc_depths": list(BC_DEPTHS), "deep_depths": list(DEEP_DEPTHS),
                       "truncation_modes": list(TRUNC_MODES), "singularity_radii": list(SING_RADII)},
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "classes": {"PRINT_SAFETY": PRINT_SAFETY, "BOUND_SLACK": BOUND_SLACK,
                    "ZERO_FLOOR": ZERO_FLOOR},
        "claims": [claim.as_dict() for claim in claims],
        "drift": drift,
        "coverage": cov,
        "verdict": "PASS" if not drift else "DRIFT",
    }


def write(path: Path = PINS_FILE, ledger: Optional[Mapping[str, Any]] = None) -> Path:
    payload = dict(ledger or evaluate())
    payload.pop("drift", None)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


def summary(ledger: Mapping[str, Any]) -> str:
    lines = []
    for claim in ledger["claims"]:
        if not claim.get("measured") == claim.get("measured"):  # nan guard
            continue
        lines.append(
            f"{claim['verdict']:5s} {claim['id']:52s} declared {claim['declared']:.6e} "
            f"measured {claim['measured']:.6e} rel {claim['relative']:.2e}"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the pin ledger JSON")
    parser.add_argument("--check", action="store_true", help="print the verdict table")
    parser.add_argument("--document", type=Path, default=DOC)
    parser.add_argument("--pins", type=Path, default=PINS_FILE)
    args = parser.parse_args(argv)

    ledger = evaluate(args.document)
    if args.check or not args.write:
        print(summary(ledger))
    print(
        f"\n{ledger['verdict']}: {len(ledger['claims'])} claims, "
        f"{len(ledger['drift'])} drift; coverage "
        f"{ledger['coverage']['protected_fraction']:.3%} of "
        f"{ledger['coverage']['numeric_tokens_total']} numeric tokens"
    )
    if args.write:
        print("wrote", write(args.pins, ledger))
    return 0 if ledger["verdict"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())