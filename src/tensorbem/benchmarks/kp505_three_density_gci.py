"""Case ``CASE-BEM-KP505-GCI`` -- KP505 three-density GCI convergence (TensorBEM).

Case definition (``tensor_case_pool``, ``source='feat-casepool-100'``)
--------------------------------------------------------------------
``case_code``  ``BEM-KP505-GCI``
``name``       KP505 三密度 GCI 收敛
``ref_source`` Roache GCI（r=2 三密度阶梯）
``tol_policy`` ``<3%``
``tier``       core

So the reference is *Roache's grid-convergence-index procedure* applied to a
three-density ladder with refinement ratio ``r = 2`` (Roache 1994, *J. Fluids
Eng.* 116:405-413; Roache 1998, *Verification and Validation in Computational
Science and Engineering*, Hermosa; the factor of safety ``Fs = 1.25`` for a
three-grid study and the ``GCI_medium/(r^p GCI_fine) -> 1`` asymptotic-range
test are the two numbers the procedure contributes, not any measured value).
The case metric is therefore the GCI itself and the acceptance line is
``GCI_fine <= 3 %``.

What this driver adds to the existing ladder studies
----------------------------------------------------
``benchmarks/kp505_w20_gci.py`` and ``benchmarks/kp505_gci_ladder.py`` already
sweep single axes.  Neither contains a *case-pool-shaped* artefact: one
uniform three-density ladder, `r = 2` on **all** panel axes simultaneously, at
three advance ratios, with (i) the GCI implementation itself verified against
closed-form exact sequences, (ii) the Roache asymptotic-range admission test
reported next to the plain ``< 3 %`` line, and (iii) an axis-isolation
diagnostic that says which panel axis the numbers actually come from.

Predeclared design (fixed by the case definition before any level was run)
--------------------------------------------------------------------------
============  ==========  =========  ===========  ==================================
level         n_radial    n_chord    wake_steps   bound panels per blade
============  ==========  =========  ===========  ==================================
``D1``              8          2           24        16
``D2``             16          4           48        64
``D3``             32          8           96       256
============  ==========  =========  ===========  ==================================

Every axis is doubled between levels, so ``r = 2`` holds for the lattice as a
whole.  Held constant across levels: ``wake_model='geometric'``,
``wake_length=1.0``, ``rev_per_second=10``, ``thickness_ratio=0.06``,
``radial_spacing='cosine'``, ``root_chord_offset=0.12``,
``core_ratio=0.05``, ``viscous=True`` -- i.e. the same physical model at three
lattices, which is what the GCI procedure requires.

Advance ratios: ``J = 0.5, 0.7, 0.9``; ``J = 0.70`` is the design point
(``DESIGN_TOL_J``).  Quantities: ``K_T``, ``10 K_Q``, ``eta_0``.

Predeclared criteria
--------------------
G1  coverage: 3 advance ratios x 3 densities x 3 quantities finite.
G2  physical admissibility at every level: ``K_T > 0``, ``K_Q > 0``.
G3  every ``K_T``/``10 K_Q`` triplet is monotone (Roache cannot form ``p`` from
    an odd-even triplet; an oscillatory triplet must be reported, not smeared).
G4  observed order ``p >= 1.0`` in every cell (a negative or sub-first-order
    ``p`` means the triplet carries no usable discretisation information).
G5  case tolerance: ``GCI_fine <= 3 %`` for ``K_T``, ``10 K_Q`` and ``eta_0``
    at the design advance ratio ``J = 0.70``.
G6  Roache asymptotic-range admission: ``|GCI_medium / (r^p GCI_fine) - 1| <=
    0.10`` in every cell.  A GCI that is numerically small but out of the
    asymptotic range is *not* an error bound; the case verdict must say so.
G7  reference-procedure verification (performed independently of the solver):
    (a) a manufactured power-law sequence ``f(h) = A + C h^pp`` with ``pp =
    1.7`` must return ``p`` to 1e-10 and ``f_ext -> A`` to 1e-10 relative;
    (b) ``GCI_fine / Fs`` must equal the *true* fine-grid relative error of
    that sequence to 1e-10; (c) the closed-form formula implemented inline in
    this module must reproduce :func:`tensorbem.uq.grid_convergence_index` to
    1e-12 relative.  This is the "reference" half of the ref-vs-got table: the
    procedure is verified before it is used.
G8  reference anchor: ``K_T``, ``10 K_Q`` and ``eta_0`` of the transcribed
    KP505 open-water table (``Table 2, page 10`` of the committed PDF, see
    :mod:`tensorbem.benchmarks.kp505_openwater_bem`) are tabulated against the
    finest lattice and the Richardson extrapolate with per-item errors.  G8 is
    a *reporting* criterion (the case tolerance G5 is about discretisation, not
    about model error); the deviation is reported, never absolutioned.

Run with ``python -m tensorbem.benchmarks.kp505_three_density_gci [outdir]``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..propeller import BladeGeometry, open_water_point
from ..uq import DEFAULT_SAFETY_FACTOR, grid_convergence_index
from .kp505_openwater_bem import (
    ANCHOR,
    ANCHOR_SOURCE,
    PRIMARY,
    TABLE_2_ROWS_VERBATIM,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTDIR = REPO_ROOT / "benchmarks" / "verified" / "bem_kp505_gci"
SCHEMA = "tensorbem.case_bem_kp505_gci/1"
CASE_CODE = "BEM-KP505-GCI"

#: Advance ratios of the case (design point first-class: 0.70).
ADVANCES = (0.5, 0.7, 0.9)
DESIGN_TOL_J = 0.7
QUANTITIES = ("KT", "10KQ", "eta0")

#: Three-density ladder, r = 2 on every panel axis (predeclared above).
LEVELS = (
    {"level": 1, "label": "D1", "n_radial": 8, "n_chord": 2, "wake_steps": 24},
    {"level": 2, "label": "D2", "n_radial": 16, "n_chord": 4, "wake_steps": 48},
    {"level": 3, "label": "D3", "n_radial": 32, "n_chord": 8, "wake_steps": 96},
)
REFINEMENT_RATIO = 2.0
CORE_RATIO = 0.05
HELD_FIXED = {
    "wake_model": PRIMARY["wake_model"],
    "wake_length": PRIMARY["wake_length"],
    "rev_per_second": PRIMARY["rev_per_second"],
    "thickness_ratio": PRIMARY["thickness_ratio"],
    "radial_spacing": PRIMARY["radial_spacing"],
    "root_chord_offset": PRIMARY["root_chord_offset"],
    "viscous": PRIMARY["viscous"],
    "core_ratio": CORE_RATIO,
}

#: Thresholds -- the 3 % is the case pool's ``tol_policy``, the 0.10 is
#: Roache's asymptotic-range admission for a three-grid study.
CASE_TOL_PCT = 3.0
STRONG_TOL_PCT = 1.0
ASYMPTOTIC_ADMISSION = 0.10
THEORY_ORDER = 2.0
MIN_OBSERVED_ORDER = 1.0
MANUFACTURED_ORDER = 1.7

#: Axis-isolation diagnostic at the design point: spanwise only.
SPAN_ONLY = {"n_chord": 4, "wake_steps": 48, "n_radial": (8, 16, 32)}

#: Fail-mode localisation: a plain density sweep (no geometric ladder) at the
#: design point, chord/wake frozen.  Answers "is the non-monotonicity a property
#: of one refinement step or of the whole span direction?".
SPAN_DENSITY_SWEEP = (8, 12, 16, 20, 24, 32)


# ---------------------------------------------------------------------------
# reference procedure (implemented inline, independently of tensorbem.uq)
# ---------------------------------------------------------------------------
def roache_gci_closed_form(fine: float, medium: float, coarse: float,
                           r: float = REFINEMENT_RATIO,
                           safety_factor: float = DEFAULT_SAFETY_FACTOR) -> dict:
    """Roache's three-grid GCI, written out from the published procedure.

    ``p = ln |(f_c - f_m)/(f_m - f_f)| / ln r``
    ``f_ext = f_f + (f_f - f_m)/(r^p - 1)``
    ``GCI_f = Fs |(f_m - f_f)/f_f| / |r^p - 1|``
    ``GCI_m = Fs |(f_c - f_m)/f_m| / |r^p - 1|``
    """

    d_fine = medium - fine
    d_coarse = coarse - medium
    if d_fine == 0.0 or d_coarse == 0.0 or fine == 0.0 or medium == 0.0:
        raise ValueError("degenerate triplet for the closed form")
    p = math.log(abs(d_coarse / d_fine)) / math.log(r)
    denom = r ** p - 1.0
    return {
        "observed_order": p,
        "extrapolated": fine + (fine - medium) / denom,
        "gci_fine": safety_factor * abs(d_fine / fine) / abs(denom),
        "gci_medium": safety_factor * abs(d_coarse / medium) / abs(denom),
        "asymptotic_ratio": (
            (safety_factor * abs(d_coarse / medium) / abs(denom))
            / (r ** p * (safety_factor * abs(d_fine / fine) / abs(denom)))
        ),
        "oscillatory": (d_fine * d_coarse) < 0.0,
    }


def _rel(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / scale


def verify_reference_procedure() -> dict:
    """G7: verify the GCI implementation before it is used on the case."""

    checks = []

    # (a)+(b) manufactured sequence with exact order and exact limit.
    A, C, pp = 0.5, 3.0, MANUFACTURED_ORDER
    h = {"coarse": 0.04, "medium": 0.02, "fine": 0.01}  # r = 2
    f = {k: A + C * v ** pp for k, v in h.items()}
    closed = roache_gci_closed_form(f["fine"], f["medium"], f["coarse"])
    truth_rel_error = abs(f["fine"] - A) / abs(f["fine"])
    checks.append({
        "id": "G7a-observed-order",
        "target": pp,
        "got": closed["observed_order"],
        "error": _rel(closed["observed_order"], pp),
        "pass": _rel(closed["observed_order"], pp) <= 1e-10,
    })
    checks.append({
        "id": "G7b-extrapolated-limit",
        "target": A,
        "got": closed["extrapolated"],
        "error": _rel(closed["extrapolated"], A),
        "pass": _rel(closed["extrapolated"], A) <= 1e-10,
    })
    gci_as_error = closed["gci_fine"] / DEFAULT_SAFETY_FACTOR
    checks.append({
        "id": "G7c-gci-equals-true-error-over-fs",
        "target": truth_rel_error,
        "got": gci_as_error,
        "error": _rel(gci_as_error, truth_rel_error),
        "pass": _rel(gci_as_error, truth_rel_error) <= 1e-10,
    })

    # (c) the shipped library must agree with the inline closed form.
    lib = grid_convergence_index(f["fine"], f["medium"], f["coarse"],
                                 refinement_ratio=REFINEMENT_RATIO)
    for key, got in (("observed_order", lib.observed_order),
                     ("extrapolated", lib.extrapolated),
                     ("gci_fine", lib.gci_fine),
                     ("gci_medium", lib.gci_medium)):
        checks.append({
            "id": f"G7d-library-vs-closed-form:{key}",
            "target": closed[key],
            "got": got,
            "error": _rel(got, closed[key]),
            "pass": _rel(got, closed[key]) <= 1e-12,
        })

    # (d) an odd-even sequence must be flagged, not silently linearised.
    osc = grid_convergence_index(1.0, 1.1, 1.05, refinement_ratio=REFINEMENT_RATIO)
    checks.append({
        "id": "G7e-oscillatory-flagged",
        "target": True,
        "got": bool(osc.oscillatory and not osc.converged),
        "error": None,
        "pass": bool(osc.oscillatory and not osc.converged),
    })

    return {
        "safety_factor": DEFAULT_SAFETY_FACTOR,
        "refinement_ratio": REFINEMENT_RATIO,
        "manufactured": {"A": A, "C": C, "order": pp,
                         "h": h, "f": f},
        "checks": checks,
        "all_pass": all(c["pass"] for c in checks),
    }


# ---------------------------------------------------------------------------
# lattice evaluation
# ---------------------------------------------------------------------------
def _geometry(n_radial: int, n_chord: int) -> BladeGeometry:
    return BladeGeometry(
        n_radial=int(n_radial),
        n_chord=int(n_chord),
        thickness_ratio=HELD_FIXED["thickness_ratio"],
        radial_spacing=HELD_FIXED["radial_spacing"],
        root_chord_offset=HELD_FIXED["root_chord_offset"],
    )


CACHE_DIRNAME = "cache"


def _cache_file(outdir: Path, advance: float, n_radial: int, n_chord: int,
                wake_steps: int) -> Path:
    return outdir / CACHE_DIRNAME / f"J{advance:.2f}_nr{n_radial}_nc{n_chord}_ws{wake_steps}.json"


def evaluate(advance: float, *, n_radial: int, n_chord: int, wake_steps: int,
             cache_dir: Path | None = None) -> dict:
    """One lattice solve; returns the three open-water quantities.

    Results are cached per (J, lattice) so that the driver can be run one
    advance ratio at a time under a wall-clock limited harness without ever
    recomputing a level (the D3 solve is ~1 min and 6.4 GB).
    """

    cached = None
    if cache_dir is not None:
        path = _cache_file(cache_dir, advance, n_radial, n_chord, wake_steps)
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
    t0 = time.perf_counter()
    result = open_water_point(
        _geometry(n_radial, n_chord),
        advance,
        wake_model=HELD_FIXED["wake_model"],
        wake_length=HELD_FIXED["wake_length"],
        wake_steps=int(wake_steps),
        rev_per_second=HELD_FIXED["rev_per_second"],
        viscous=HELD_FIXED["viscous"],
        core_ratio=HELD_FIXED["core_ratio"],
    )
    wall = time.perf_counter() - t0
    # `open_water_point` returns K_Q with the sign of the driving torque; the
    # resisting (physical) torque -- hence 10 K_Q and eta_0 -- is positive for
    # an absorbing propeller, as in the existing ladder studies.
    kq = -float(result.torque)
    kt = float(result.thrust)
    cell = {
        "J": float(advance),
        "n_radial": int(n_radial),
        "n_chord": int(n_chord),
        "wake_steps": int(wake_steps),
        "n_panels_per_blade": int(result.n_panels),
        "KT": kt,
        "10KQ": 10.0 * kq,
        "eta0": advance * kt / (2.0 * math.pi * kq) if kq else float("nan"),
        "KT_inviscid": float(result.thrust_inviscid),
        "viscous_drag_nd": float(result.viscous_drag),
        "wall_seconds": wall,
        "cached": bool(cached),
    }
    if cache_dir is not None:
        path = _cache_file(cache_dir, advance, n_radial, n_chord, wake_steps)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cell, indent=2), encoding="utf-8")
    return cell


def _gci(fine: float, medium: float, coarse: float, quantity: str) -> dict:
    lib = grid_convergence_index(fine, medium, coarse,
                                 refinement_ratio=REFINEMENT_RATIO,
                                 quantity=quantity)
    closed = roache_gci_closed_form(fine, medium, coarse)
    row = {
        "quantity": quantity,
        "levels": [fine, medium, coarse],
        "observed_order": lib.observed_order,
        "order_error_vs_theory_pct": (
            100.0 * (lib.observed_order - THEORY_ORDER) / THEORY_ORDER
            if math.isfinite(lib.observed_order) else None
        ),
        "extrapolated": lib.extrapolated,
        "gci_fine_pct": 100.0 * lib.gci_fine,
        "gci_medium_pct": 100.0 * lib.gci_medium,
        "estimated_fine_error_pct": 100.0 * lib.gci_fine / lib.safety_factor,
        "asymptotic_ratio": lib.asymptotic_ratio,
        "monotone": lib.monotone,
        "oscillatory": lib.oscillatory,
        "converged": lib.converged,
        "under_case_tol_3pct": bool(math.isfinite(lib.gci_fine)
                                   and 100.0 * lib.gci_fine <= CASE_TOL_PCT),
        "under_strong_tol_1pct": bool(math.isfinite(lib.gci_fine)
                                     and 100.0 * lib.gci_fine <= STRONG_TOL_PCT),
        # Roache's ratio test presupposes a monotone sequence: on an odd-even
        # triplet p < 0 and the ratio can land near 1 by accident (seen at
        # J = 0.5 on 10 K_Q here), so admission is gated on monotonicity.
        "asymptotic_admitted": bool(
            lib.monotone
            and math.isfinite(lib.observed_order)
            and lib.observed_order > 0.0
            and math.isfinite(lib.asymptotic_ratio)
            and abs(lib.asymptotic_ratio - 1.0) <= ASYMPTOTIC_ADMISSION
        ),
        "library_vs_closed_form_max_rel_error": max(
            _rel(lib.observed_order, closed["observed_order"]),
            _rel(lib.extrapolated, closed["extrapolated"]),
            _rel(lib.gci_fine, closed["gci_fine"]),
        ),
    }
    return row


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def _anchor_row(advance: float) -> dict:
    entry = ANCHOR[advance]
    kq = entry["10KQ"] / 10.0
    return {
        "J": advance,
        "KT": entry["KT"],
        "10KQ": entry["10KQ"],
        "eta0": advance * entry["KT"] / (2.0 * math.pi * kq),
    }


def _git(args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=False).stdout.strip()
    except OSError:
        return ""


def _figures(cells, gci_rows, outdir: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    made = []

    # Figure 1: the three-density ladder itself.
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    for ax, q in zip(axes, ("KT", "10KQ", "eta0")):
        for advance in ADVANCES:
            pts = [c for c in cells if c["J"] == advance]
            pts.sort(key=lambda c: c["n_panels_per_blade"])
            x = [c["n_panels_per_blade"] for c in pts]
            y = [c[q] for c in pts]
            ax.plot(x, y, "o-", label=f"J = {advance:.2f}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("bound panels per blade (r = 2 ladder)")
        ax.set_ylabel(q)
        ax.set_title(f"{q} vs panel density")
        ax.grid(alpha=0.3)
        if q in ("KT", "10KQ"):
            anchor = _anchor_row(DESIGN_TOL_J)[q]
            ax.axhline(anchor, color="k", ls="--", lw=1.0,
                       label=f"KP505 table anchor @ J=0.70 = {anchor:.3f}")
        ax.legend(fontsize=8)
    fig.suptitle("CASE-BEM-KP505-GCI: three-density ladder (r = 2)")
    fig.tight_layout()
    path = figdir / "kp505_gci_ladder.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(str(path.relative_to(outdir)))

    # Figure 2: GCI per quantity and advance ratio against the 3 % line.
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    width = 0.8 / len(QUANTITIES)
    for ax, metric, ylabel in (
        (axes[0], "gci_fine_pct", "GCI_fine [%]"),
        (axes[1], "asymptotic_ratio", "GCI_medium / (r^p GCI_fine)"),
    ):
        for k, q in enumerate(QUANTITIES):
            xs, ys = [], []
            for i, advance in enumerate(ADVANCES):
                row = next(r for r in gci_rows
                           if r["J"] == advance and r["quantity"] == q)
                xs.append(i + k * width - 0.4 + width / 2)
                ys.append(row[metric])
            ax.bar(xs, ys, width=width, label=q)
        ax.set_xticks(range(len(ADVANCES)))
        ax.set_xticklabels([f"J = {a:.2f}" for a in ADVANCES])
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    axes[0].axhline(CASE_TOL_PCT, color="r", ls="--", lw=1.2,
                    label=f"case tolerance {CASE_TOL_PCT:.0f} %")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[1].axhline(1.0, color="r", ls="--", lw=1.2, label="asymptotic (ratio = 1)")
    axes[1].axhspan(1.0 - ASYMPTOTIC_ADMISSION, 1.0 + ASYMPTOTIC_ADMISSION,
                    color="r", alpha=0.10, label=f"admission +-{ASYMPTOTIC_ADMISSION:.0%}")
    axes[1].legend(fontsize=8)
    fig.suptitle("CASE-BEM-KP505-GCI: GCI and Roache asymptotic-range test")
    fig.tight_layout()
    path = figdir / "kp505_gci_indices.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(str(path.relative_to(outdir)))
    return made


def precompute_ladder(outdir: Path, advance: float) -> None:
    """Cache the three densities of one advance ratio (wall-clock slicing)."""
    outdir.mkdir(parents=True, exist_ok=True)
    for level in LEVELS:
        cell = evaluate(advance, **{k: level[k] for k in ("n_radial", "n_chord", "wake_steps")},
                        cache_dir=outdir)
        print(f"  J={advance:.2f} {level['label']} ({level['n_radial']}x{level['n_chord']}x"
              f"{level['wake_steps']}) KT={cell['KT']:.6f} 10KQ={cell['10KQ']:.6f} "
              f"eta0={cell['eta0']:.5f} [{cell['wall_seconds']:.1f} s]", flush=True)


def precompute_span(outdir: Path) -> None:
    """Cache the spanwise-only isolation ladder at the design point."""
    outdir.mkdir(parents=True, exist_ok=True)
    for n_radial in SPAN_ONLY["n_radial"]:
        cell = evaluate(DESIGN_TOL_J, n_radial=n_radial,
                        n_chord=SPAN_ONLY["n_chord"], wake_steps=SPAN_ONLY["wake_steps"],
                        cache_dir=outdir)
        print(f"  span-only n_radial={n_radial} KT={cell['KT']:.6f} "
              f"10KQ={cell['10KQ']:.6f} [{cell['wall_seconds']:.1f} s]", flush=True)


def precompute_sweep(outdir: Path) -> None:
    """Cache the plain spanwise density sweep at the design point."""
    outdir.mkdir(parents=True, exist_ok=True)
    for n_radial in SPAN_DENSITY_SWEEP:
        cell = evaluate(DESIGN_TOL_J, n_radial=n_radial,
                        n_chord=SPAN_ONLY["n_chord"], wake_steps=SPAN_ONLY["wake_steps"],
                        cache_dir=outdir)
        print(f"  sweep n_radial={n_radial} KT={cell['KT']:.6f} "
              f"10KQ={cell['10KQ']:.6f} [{cell['wall_seconds']:.1f} s]", flush=True)


def assemble(outdir: Path, *, advances=ADVANCES, span_only: bool = True) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    verification = verify_reference_procedure()

    cells = []
    for advance in advances:
        for level in LEVELS:
            cell = evaluate(advance, **{k: level[k] for k in ("n_radial", "n_chord", "wake_steps")},
                            cache_dir=outdir)
            cell["level"] = level["level"]
            cell["label"] = level["label"]
            print(f"  J={advance:.2f} {level['label']} "
                  f"({level['n_radial']}x{level['n_chord']}x{level['wake_steps']}) "
                  f"KT={cell['KT']:.6f} 10KQ={cell['10KQ']:.6f} "
                  f"eta0={cell['eta0']:.5f} [{cell['wall_seconds']:.1f} s]", flush=True)
            cells.append(cell)

    gci_rows = []
    for advance in ADVANCES:
        for q in QUANTITIES:
            triple = sorted([c for c in cells if c["J"] == advance],
                            key=lambda c: c["n_panels_per_blade"], reverse=True)
            row = _gci(triple[0][q], triple[1][q], triple[2][q], q)
            row["J"] = advance
            gci_rows.append(row)

    # Axis isolation at the design point: spanwise only, chord/wake frozen.
    span_cells = []
    for n_radial in (SPAN_ONLY["n_radial"] if span_only else ()):
        cell = evaluate(DESIGN_TOL_J, n_radial=n_radial,
                        n_chord=SPAN_ONLY["n_chord"], wake_steps=SPAN_ONLY["wake_steps"],
                        cache_dir=outdir)
        span_cells.append(cell)
        print(f"  span-only n_radial={n_radial} KT={cell['KT']:.6f} "
              f"10KQ={cell['10KQ']:.6f} [{cell['wall_seconds']:.1f} s]", flush=True)
    span_gci_kt = _gci(span_cells[2]["KT"], span_cells[1]["KT"], span_cells[0]["KT"], "KT")
    span_gci_kq = _gci(span_cells[2]["10KQ"], span_cells[1]["10KQ"], span_cells[0]["10KQ"], "10KQ")
    span_gci_kt["J"] = DESIGN_TOL_J
    span_gci_kq["J"] = DESIGN_TOL_J

    # Reference anchor at the design point (G8) -- verbatim transcription.
    anchor = _anchor_row(DESIGN_TOL_J)
    finest = sorted([c for c in cells if c["J"] == DESIGN_TOL_J],
                    key=lambda c: c["n_panels_per_blade"], reverse=True)[0]
    anchor_rows = []
    for q in QUANTITIES:
        row = next(r for r in gci_rows if r["J"] == DESIGN_TOL_J and r["quantity"] == q)
        got_fine = finest[q]
        got_ext = row["extrapolated"]
        anchor_rows.append({
            "quantity": q,
            "ref_anchor": anchor[q],
            "got_finest": got_fine,
            "error_finest_pct": 100.0 * (got_fine - anchor[q]) / anchor[q],
            "got_extrapolated": got_ext,
            "error_extrapolated_pct": 100.0 * (got_ext - anchor[q]) / anchor[q],
            "under_case_tol_3pct": abs(100.0 * (got_fine - anchor[q]) / anchor[q]) <= CASE_TOL_PCT,
        })

    def cell_of(advance, q):
        return next(r for r in gci_rows if r["J"] == advance and r["quantity"] == q)

    design_cells = [cell_of(DESIGN_TOL_J, q) for q in QUANTITIES]
    criteria = [
        {"id": "G1-coverage", "pass": all(
            math.isfinite(c[q]) for c in cells for q in QUANTITIES)
            and len(cells) == len(ADVANCES) * len(LEVELS)},
        {"id": "G2-physical", "pass": all(
            c["KT"] > 0.0 and c["10KQ"] > 0.0 for c in cells)},
        {"id": "G3-monotone", "pass": all(
            r["monotone"] for r in gci_rows if r["quantity"] in ("KT", "10KQ"))},
        {"id": "G4-observed-order>=1", "pass": all(
            math.isfinite(r["observed_order"]) and r["observed_order"] >= MIN_OBSERVED_ORDER
            for r in gci_rows)},
        {"id": "G5-case-tolerance-3pct", "pass": all(
            r["under_case_tol_3pct"] for r in design_cells)},
        {"id": "G6-asymptotic-admission", "pass": all(
            r["asymptotic_admitted"] for r in gci_rows)},
        {"id": "G7-reference-procedure-verified", "pass": verification["all_pass"]},
        {"id": "G8-anchor-reported", "pass": len(anchor_rows) == len(QUANTITIES)},
    ]

    verdict = {
        "case": CASE_CODE,
        "tolerance_pct": CASE_TOL_PCT,
        "pass_by_id": {c["id"]: bool(c["pass"]) for c in criteria},
        "all_criteria_pass": all(c["pass"] for c in criteria),
        "case_tolerance_answer": {
            q: {
                "GCI_fine_pct": next(r for r in design_cells if r["quantity"] == q)["gci_fine_pct"],
                "under_3pct": next(r for r in design_cells if r["quantity"] == q)["under_case_tol_3pct"],
                "asymptotic_ratio": next(r for r in design_cells if r["quantity"] == q)["asymptotic_ratio"],
                "certified": bool(next(r for r in design_cells if r["quantity"] == q)["under_case_tol_3pct"]
                                  and next(r for r in design_cells if r["quantity"] == q)["asymptotic_admitted"]),
            }
            for q in QUANTITIES
        },
        "fail_stage": None,
        "failure_reason": None,
    }
    failed = [c["id"] for c in criteria if not c["pass"]]
    if failed:
        verdict["fail_stage"] = failed[0]
        verdict["failure_reason"] = (
            "predeclared criterion(s) not met: " + ", ".join(failed)
            + "; the GCI value itself is reported regardless "
            "(never smeared, never re-thresholded)"
        )

    # Fail-mode localisation: plain spanwise density sweep (cached).
    sweep_cells = []
    for n_radial in SPAN_DENSITY_SWEEP:
        path = _cache_file(outdir, DESIGN_TOL_J, n_radial,
                           SPAN_ONLY["n_chord"], SPAN_ONLY["wake_steps"])
        if path.is_file():
            sweep_cells.append(json.loads(path.read_text(encoding="utf-8")))
    sweep = {
        "config": {"J": DESIGN_TOL_J, "n_chord": SPAN_ONLY["n_chord"],
                   "wake_steps": SPAN_ONLY["wake_steps"],
                   "n_radial": list(SPAN_DENSITY_SWEEP)},
        "cells": sweep_cells,
        "available": len(sweep_cells) == len(SPAN_DENSITY_SWEEP),
    }
    if sweep["available"]:
        kt = [c["KT"] for c in sweep_cells]
        sweep["KT_spread_pct"] = 100.0 * (max(kt) - min(kt)) / (sum(kt) / len(kt))
        sweep["KT_range"] = [min(kt), max(kt)]
        sweep["monotone_in_n_radial"] = all(
            (kt[i + 1] - kt[i]) * (kt[1] - kt[0]) > 0.0 for i in range(len(kt) - 1))

    # Cross-check against the independently produced W20 artefact (if present).
    cross_check = {"status": "NOT_AVAILABLE",
                   "note": "benchmarks/verified/bem_kp505_openwater_w20/gci.json "
                           "not present in this worktree"}
    w20 = REPO_ROOT / "benchmarks" / "verified" / "bem_kp505_openwater_w20" / "gci.json"
    if w20.is_file():
        try:
            ref = json.loads(w20.read_text(encoding="utf-8"))
            ref_design = ref["verdict"]["design_point"]
            here = {q: next(r for r in design_cells if r["quantity"] == q)["gci_fine_pct"]
                    for q in ("KT", "10KQ")}
            cross_check = {
                "status": "COMPARED",
                "reference": str(w20.relative_to(REPO_ROOT)),
                "reference_mission": ref.get("generated_by") or ref.get("topic"),
                "items": [
                    {"quantity": q,
                     "ref_gci_fine_pct": ref_design[f"gci_{q}_pct"],
                     "got_gci_fine_pct": here[q],
                     "abs_difference_pct_points": abs(here[q] - ref_design[f"gci_{q}_pct"]),
                     "pass": abs(here[q] - ref_design[f"gci_{q}_pct"]) <= 1e-6}
                    for q in ("KT", "10KQ")
                ],
            }
            cross_check["all_pass"] = all(i["pass"] for i in cross_check["items"])
        except (KeyError, ValueError) as exc:  # pragma: no cover - artefact drift
            cross_check = {"status": "UNREADABLE", "error": repr(exc)}

    figures = _figures(cells, gci_rows, outdir)

    payload = {
        "schema": SCHEMA,
        "case": {
            "code": CASE_CODE,
            "name": "KP505 三密度 GCI 收敛",
            "repo": "TensorBEM",
            "ref_source": "Roache GCI (r = 2, three-density ladder)",
            "tol_policy": f"<{CASE_TOL_PCT:.0f}%",
            "ref_citations": [
                "P. J. Roache (1994), Perspective: a method for uniform reporting "
                "of grid refinement studies, J. Fluids Eng. 116(3):405-413 "
                "(GCI, Fs = 1.25 for three grids).",
                "P. J. Roache (1998), Verification and Validation in Computational "
                "Science and Engineering, Hermosa (asymptotic-range ratio "
                "GCI_medium/(r^p GCI_fine) -> 1).",
                "ASME V&V 20-2009 / I. B. Celik et al. (2008), J. Fluids Eng. "
                "130(7):078001 (procedure adopted by tensorbem.uq).",
            ],
            "anchor": {
                "citation": ANCHOR_SOURCE["citation"],
                "doi": ANCHOR_SOURCE["doi"],
                "location": ANCHOR_SOURCE["location"],
                "table": ANCHOR_SOURCE["table_caption_verbatim"],
                "column_header_verbatim": ANCHOR_SOURCE["column_header_verbatim"],
                "row_J0.7_verbatim": next(
                    r for r in TABLE_2_ROWS_VERBATIM if r.split()[0] == "0.7"),
                "provenance_status": ANCHOR_SOURCE["provenance_depth"],
                "access": ANCHOR_SOURCE["access"],
            },
        },
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "env": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
            "git_commit": _git(["rev-parse", "HEAD"]),
            "git_dirty": bool(_git(["status", "--porcelain"])),
        },
        "design": {
            "advances": list(ADVANCES),
            "design_advance": DESIGN_TOL_J,
            "levels": [dict(l) for l in LEVELS],
            "refinement_ratio": REFINEMENT_RATIO,
            "held_fixed": HELD_FIXED,
            "quantities": list(QUANTITIES),
            "case_tolerance_pct": CASE_TOL_PCT,
            "asymptotic_admission": ASYMPTOTIC_ADMISSION,
            "theory_order": THEORY_ORDER,
            "solver_entry_point": "tensorbem.propeller.open_water_point",
            "geometry_model": "tensorbem.propeller.BladeGeometry (KP505 particulars "
                              "from GEOMETRY_SOURCE; chord/camber/skew modelled)",
        },
        "reference_verification": verification,
        "cells": cells,
        "gci": gci_rows,
        "axis_isolation_spanwise": {
            "config": SPAN_ONLY,
            "cells": span_cells,
            "gci_KT": span_gci_kt,
            "gci_10KQ": span_gci_kq,
        },
        "anchor_comparison": anchor_rows,
        "fail_mode_localisation": sweep,
        "cross_check_w20": cross_check,
        "criteria": criteria,
        "verdict": verdict,
        "figures": figures,
        "reproduction": {
            "cmd": "PYTHONPATH=src python -m tensorbem.benchmarks.kp505_three_density_gci",
            "wall_seconds_total": sum(c["wall_seconds"] for c in cells)
            + sum(c["wall_seconds"] for c in span_cells),
            "deterministic": True,
            "random_seed": None,
        },
        "method_notes": [
            "Roache's asymptotic-range ratio GCI_medium/(r^p GCI_fine) is an "
            "algebraic identity equal to f_fine/f_medium for a single three-grid "
            "triplet (same Fs, same triplet), so on a *three-density* study it is "
            "almost always ~1 and does not test p at all: here the ratios are "
            "0.91-0.99 (inside any +-10 % band) while every observed order but "
            "one is negative.  G6 is therefore gated on monotone AND p > 0; "
            "otherwise an anti-convergent sequence would be 'admitted' by an "
            "identity.  Testing the asymptotic range needs a fourth level and a "
            "comparison of successive triplets (see "
            "benchmarks/kp505_gci_ladder.py, triplet disagreement ~30x).",
        ],
        "friction": [
            "The platform knowledge search engine (:8021) was offline during the "
            "kick-off sweep (knowledge_base / oss_solutions checks returned "
            "'engine_offline'), so the three-question pre-check could only be done "
            "against the in-repo code and the negative-knowledge API.",
            "The lattice solve is direct-dense in (bound + wake) segments: D3 "
            "(256 bound panels, 96 wake steps) already needs ~6.4 GB peak RSS and "
            "52 s, so a fourth density level (r = 2 again) is out of reach in this "
            "container - the asymptotic range therefore cannot be *tested* by "
            "refinement, only *admitted* by Roache's ratio.",
        ],
    }

    (outdir / "case.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                      encoding="utf-8")

    with (outdir / "levels.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["J", "level", "n_radial", "n_chord", "wake_steps",
                    "panels_per_blade", "KT", "10KQ", "eta0", "wall_seconds"])
        for c in cells:
            w.writerow([c["J"], c["label"], c["n_radial"], c["n_chord"],
                        c["wake_steps"], c["n_panels_per_blade"],
                        f"{c['KT']:.10g}", f"{c['10KQ']:.10g}", f"{c['eta0']:.10g}",
                        f"{c['wall_seconds']:.3f}"])

    with (outdir / "gci.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["J", "quantity", "observed_order", "extrapolated",
                    "gci_fine_pct", "gci_medium_pct", "estimated_fine_error_pct",
                    "asymptotic_ratio", "monotone", "oscillatory", "converged",
                    "under_3pct", "asymptotic_admitted"])
        for r in gci_rows:
            w.writerow([r["J"], r["quantity"], f"{r['observed_order']:.6f}",
                        f"{r['extrapolated']:.10g}", f"{r['gci_fine_pct']:.6f}",
                        f"{r['gci_medium_pct']:.6f}",
                        f"{r['estimated_fine_error_pct']:.6f}",
                        f"{r['asymptotic_ratio']:.6g}", r["monotone"],
                        r["oscillatory"], r["converged"], r["under_case_tol_3pct"],
                        r["asymptotic_admitted"]])
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("outdir", nargs="?", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--phase", default="assemble",
                        choices=("assemble", "ladder", "span", "sweep"),
                        help="ladder/span cache one slice (wall-clock slicing); "
                             "assemble builds the artefact from the cache")
    parser.add_argument("--advance", type=float, default=DESIGN_TOL_J,
                        help="advance ratio for --phase ladder")
    args = parser.parse_args(argv)
    outdir = Path(args.outdir)
    if args.phase == "ladder":
        precompute_ladder(outdir, args.advance)
        return 0
    if args.phase == "span":
        precompute_span(outdir)
        return 0
    if args.phase == "sweep":
        precompute_sweep(outdir)
        return 0
    payload = assemble(outdir)
    v = payload["verdict"]
    print("\n=== CASE-BEM-KP505-GCI ===")
    for k, ok in v["pass_by_id"].items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}")
    print(f"  case tolerance {v['tolerance_pct']:.0f} % -> "
          f"all_criteria_pass = {v['all_criteria_pass']}")
    if v["fail_stage"]:
        print(f"  honest FAIL at {v['fail_stage']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())