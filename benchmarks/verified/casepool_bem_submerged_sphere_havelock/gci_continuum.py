#!/usr/bin/env python3
"""Systematic grid-convergence study for ``CASE-BEM-SUBMERGED-SPHERE-HAVELOCK``.

Why this second driver exists
-----------------------------
``run_case.py`` answers the pool entry with a *comparison* verdict: the 1280-panel
BEM result lies within 5 % of the multipole reference.  That answers "does the
solver agree with the reference" but not "who is wrong, and by how much": a 1.4 %
residual is compatible with (a) pure discretisation error of the BEM, (b) error in
the reference series, or (c) both.  This driver separates the two with the
standard V&V apparatus (Roache 1994/1998; ASME V&V 20-2009; Celik et al. 2008):

    1. run the *same* problem on a three-grid ladder (80 / 320 / 1280 panels);
    2. fit the observed order of accuracy p from the two successive differences
       (trilinear refinement ratio r = 2 between consecutive rungs);
    3. Richardson-extrapolate to the continuum limit f_exact^num;
    4. report the grid-convergence index GCI_fine = Fs |(f_f - f_c)/f_f| / (r^p - 1)
       (Fs = 1.25, 95 % confidence), which is the numerical uncertainty U_num;
    5. compare f_fine with the *independent* multipole reference and apply the
       ASME V&V 20 validation criterion  E <= U_V = sqrt(U_num^2 + U_ref^2).

The point of step 5 is the interesting one.  If the reference series were wrong by
its own size of the residual, refining the mesh would not drive E to zero; if the
residual is the mesh's, it must fall as h^p and the extrapolated value must land on
the reference.  This study therefore *validates the reference* rather than merely
comparing two numbers -- the missing half of the case-pool evidence.

Predeclared gates (fixed in this file before any number was produced)
---------------------------------------------------------------------
G1 (order consistency)
    For every (ka, coefficient) cell whose successive differences are resolvable
    (|f2-f1| > 100 |f3-f2| is *not* required; we require |f3-f2| > 1e-12 |f3|),
    1.5 <= p <= 3.0.  Constant-strength panels on a curved body are expected to be
    O(h^2) for the dipole/integral operators; a p outside [1.5, 3.0] means the
    ladder is not in the asymptotic range and the extrapolation is not admissible.
    Cells that cannot pin an order (differences at round-off) are reported as
    ``NOT_RESOLVED`` and are *excluded from G1* but never silently dropped: their
    values are still compared against the reference.
G2 (continuum limit vs reference, full band)
    max over the *full declared band* ka in {0.1, 0.5, 1, 2, 3, 4, 5} of
    |f_exact^num - ref| / |ref|  <  1.0 %.  This is the reference-validation
    statement: the discrete solution's own error must be removed before the
    reference can be judged.
G3 (ASME V&V 20 validation at the case mesh)
    E = |f_1280 - ref| / |ref|  <=  U_V = 1.25 GCI_1280  for every cell of the
    band.  (U_ref is the multipole truncation uncertainty, measured < 1e-9
    relative by the 20 -> 40 term test, hence negligible.)
G4 (the economy-tier FAIL is a discretisation effect, not a solver defect)
    The observed 320-panel relative error must be within a factor 2 of the
    second-order prediction r^p * e_1280 (r = 2, p from G1).  If the economy-tier
    error were a formulation bias it would not scale like a grid error.
Stopping rule
    One ladder, one band, no threshold adjusted after the numbers are seen; the
    band is the *full* pool band (7 wavenumbers), extending the 5-point band that
    ``run_case.py`` gated on the case mesh and closing that documented gap.

Reproduce
    PYTHONPATH=src python3 \\
        benchmarks/verified/casepool_bem_submerged_sphere_havelock/gci_continuum.py \\
        --outdir docs/case_bem_submerged_sphere_havelock/gci --figure

Runtime note
    The 1280-panel rungs dominate the cost (~5 min/point single core on loongarch64).
    ``--subdivisions 1 2`` gives the cheap 3-point variant used by the regression
    test; the committed artefact was produced with the full ladder.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorbem.benchmarks import submerged_sphere_havelock as library

#: Pool band (the full band of ``run_case.py``'s protocol, all seven wavenumbers).
BAND_KA = (0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
#: Grid ladder: 4*4**s constant panels -> 80 / 320 / 1280.
SUBDIVISIONS = (1, 2, 3)
#: Linear refinement ratio of the ladder is 2 (panel count ratio 4).
REFINEMENT_RATIO = 2.0
#: Roache safety factor for the 95 % confidence band.
SAFETY_FACTOR = 1.25

RADIUS = 1.0
SUBMERGENCE = 2.0
DENSITY = 1025.0
GRAVITY = 9.80665

QUANTITIES = ("a33", "b33", "a11", "b11")

GATE_ORDER = (1.5, 3.0)
GATE_CONTINUUM_VS_REF = 0.01
GATE_ECONOMY_PREDICTION_FACTOR = 2.0


# --------------------------------------------------------------------------- #
# V&V kernels (pure functions; unit-tested in tests/test_casepool_havelock_gci.py)
# --------------------------------------------------------------------------- #
def observed_order(coarse: float, medium: float, fine: float, r: float) -> float:
    """Order of accuracy from three successive grids (Roache eq. 5).

    ``coarse``/``medium``/``fine`` are the values of the same functional on the
    three grids of linear refinement ratio ``r``.  ``nan`` is returned when the
    differences are too close to resolve an order.
    """
    d1 = medium - coarse
    d2 = fine - medium
    scale = max(abs(coarse), abs(medium), abs(fine), 1e-300)
    if abs(d2) <= 1e-13 * scale or abs(d1) <= 1e-13 * scale:
        return float("nan")
    ratio = d1 / d2
    if ratio <= 0.0:
        # Non-monotone ladder: no admissible order.
        return float("nan")
    return math.log(abs(ratio)) / math.log(r)


def richardson(f_coarse: float, f_fine: float, p: float, r: float) -> float:
    """Richardson extrapolation to zero grid spacing."""
    if not math.isfinite(p):
        return float("nan")
    return f_fine + (f_fine - f_coarse) / (r**p - 1.0)


def gci_fine(f_coarse: float, f_fine: float, p: float, r: float,
             safety: float = SAFETY_FACTOR) -> float:
    """Grid-convergence index on the fine grid (95 % band, Roache)."""
    if not math.isfinite(p):
        return float("nan")
    scale = abs(f_fine) if abs(f_fine) > 0.0 else 1.0
    return safety * abs((f_fine - f_coarse) / scale) / (r**p - 1.0)


def cell_study(values: Sequence[float], ref: float,
               r: float = REFINEMENT_RATIO) -> dict[str, Any]:
    """Full per-cell V&V verdict for one (ka, coefficient) triple."""
    coarse, medium, fine = (float(v) for v in values)
    p = observed_order(coarse, medium, fine, r)
    extrapolated = richardson(medium, fine, p, r)
    gci = gci_fine(medium, fine, p, r)
    e_fine = abs(fine - ref) / abs(ref) if ref != 0.0 else abs(fine - ref)
    e_coarse = abs(medium - ref) / abs(ref) if ref != 0.0 else abs(medium - ref)
    e_cont = (abs(extrapolated - ref) / abs(ref)
              if ref != 0.0 and math.isfinite(extrapolated) else float("nan"))
    return {
        "coarse": coarse,
        "medium": medium,
        "fine": fine,
        "reference": ref,
        "observed_order": p,
        "order_resolved": bool(math.isfinite(p)),
        "richardson_continuum": extrapolated,
        "gci_fine": gci,
        "err_fine_vs_ref": e_fine,
        "err_coarse_vs_ref": e_coarse,
        "err_continuum_vs_ref": e_cont,
        "gate_order": (bool(math.isfinite(p)) and GATE_ORDER[0] <= p <= GATE_ORDER[1]),
        "gate_validation": bool(math.isfinite(gci) and e_fine <= gci),
        "gate_continuum": bool(math.isfinite(e_cont) and e_cont < GATE_CONTINUUM_VS_REF),
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _env() -> dict[str, Any]:
    def _run(*args: str) -> str:
        try:
            return subprocess.run(args, cwd=Path(__file__).resolve().parents[3],
                                  capture_output=True, text=True, check=False).stdout.strip()
        except Exception:  # pragma: no cover - provenance is best effort
            return ""

    status = _run("git", "status", "--porcelain")
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "host": platform.node(),
        "git_head": _run("git", "rev-parse", "HEAD"),
        "git_branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(status),
    }


def run_ladder(ka_values: Sequence[float], subdivisions: Sequence[int],
               log: list[str]) -> dict[float, dict[int, dict[str, Any]]]:
    """Evaluate every (ka, subdivision) cell; returns the nested raw table."""
    table: dict[float, dict[int, dict[str, Any]]] = {}
    for ka in ka_values:
        table[ka] = {}
        for sub in subdivisions:
            started = time.perf_counter()
            rows = library.run_frequency_sweep(
                ka_values=(float(ka),), radius=RADIUS, submergence=SUBMERGENCE,
                subdivision=int(sub), density=DENSITY, gravity=GRAVITY,
            )
            seconds = time.perf_counter() - started
            row = dict(rows[0])
            row["seconds"] = seconds
            table[float(ka)][int(sub)] = row
            log.append(
                f"ka={ka:g} subdiv={sub} ({row['n_panels']} panels): {seconds:.2f} s"
            )
    return table


def build_result(ka_values: Sequence[float] = BAND_KA,
                 subdivisions: Sequence[int] = SUBDIVISIONS) -> dict[str, Any]:
    log: list[str] = []
    started = time.perf_counter()
    table = run_ladder(ka_values, subdivisions, log)

    cells: list[dict[str, Any]] = []
    for ka in ka_values:
        for quantity in QUANTITIES:
            values = [table[float(ka)][int(s)][f"{quantity}_bem"] for s in subdivisions]
            ref = table[float(ka)][int(subdivisions[0])][f"{quantity}_exact"]
            study = cell_study(values, ref)
            cells.append({
                "ka": float(ka),
                "quantity": quantity,
                "panels": [table[float(ka)][int(s)]["n_panels"] for s in subdivisions],
                "levels": [{"subdivision": int(s),
                            "n_panels": table[float(ka)][int(s)]["n_panels"],
                            "value": table[float(ka)][int(s)][f"{quantity}_bem"]}
                           for s in subdivisions],
                **study,
            })

    resolved = [c for c in cells if c["order_resolved"]]
    orders = [c["observed_order"] for c in resolved]
    cont_err = [c["err_continuum_vs_ref"] for c in cells
                if math.isfinite(c["err_continuum_vs_ref"])]

    # G4: does the coarse-grid error scale like a second-order grid effect?
    economy_cells = [c for c in cells if c["order_resolved"] and c["err_fine_vs_ref"] > 0.0]
    prediction_ratios = []
    for c in economy_cells:
        predicted = (REFINEMENT_RATIO ** c["observed_order"]) * c["err_fine_vs_ref"]
        if predicted > 0.0:
            prediction_ratios.append(c["err_coarse_vs_ref"] / predicted)

    gates = {
        "G1_order_in_range": {
            "gate": list(GATE_ORDER),
            "n_resolved": len(resolved),
            "n_cells": len(cells),
            "n_failures": sum(1 for c in resolved if not c["gate_order"]),
            "order_min": min(orders) if orders else None,
            "order_max": max(orders) if orders else None,
            "order_median": float(np.median(orders)) if orders else None,
            "unresolved_cells": [f"ka={c['ka']:g} {c['quantity']}"
                                 for c in cells if not c["order_resolved"]],
            "verdict": "PASS" if resolved and all(c["gate_order"] for c in resolved) else "FAIL",
        },
        "G2_continuum_vs_reference": {
            "gate": GATE_CONTINUUM_VS_REF,
            "max_rel_error": max(cont_err) if cont_err else None,
            "worst_cell": max(
                (c for c in cells if math.isfinite(c["err_continuum_vs_ref"])),
                key=lambda c: c["err_continuum_vs_ref"], default=None,
            ) and None,
            "verdict": "PASS" if cont_err and max(cont_err) < GATE_CONTINUUM_VS_REF else "FAIL",
        },
        "G3_asme_validation_at_1280": {
            "criterion": "E = |f_1280 - ref|/|ref| <= U_V = 1.25*GCI_1280",
            "n_cells": len(cells),
            "n_failures": sum(1 for c in cells if not c["gate_validation"]),
            "max_err_fine_vs_ref": max(c["err_fine_vs_ref"] for c in cells),
            "max_gci_fine": max((c["gci_fine"] for c in cells
                                 if math.isfinite(c["gci_fine"])), default=None),
            "verdict": "PASS" if all(c["gate_validation"] for c in cells) else "FAIL",
        },
        "G4_economy_error_is_grid_scaling": {
            "gate_factor": GATE_ECONOMY_PREDICTION_FACTOR,
            "ratio_min": min(prediction_ratios) if prediction_ratios else None,
            "ratio_max": max(prediction_ratios) if prediction_ratios else None,
            "n_cells": len(prediction_ratios),
            "verdict": "PASS" if prediction_ratios and all(
                1.0 / GATE_ECONOMY_PREDICTION_FACTOR <= q <= GATE_ECONOMY_PREDICTION_FACTOR
                for q in prediction_ratios) else "FAIL",
        },
    }
    if gates["G2_continuum_vs_reference"]["max_rel_error"] is not None:
        worst = max((c for c in cells if math.isfinite(c["err_continuum_vs_ref"])),
                    key=lambda c: c["err_continuum_vs_ref"])
        gates["G2_continuum_vs_reference"]["worst_cell"] = {
            "ka": worst["ka"], "quantity": worst["quantity"],
            "err": worst["err_continuum_vs_ref"],
        }

    overall = "PASS" if all(g["verdict"] == "PASS" for g in gates.values()) else "FAIL"
    return {
        "case": "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
        "study": "grid convergence (Richardson / GCI / ASME V&V 20)",
        "protocol": {
            "ka_band": list(ka_values),
            "subdivisions": list(subdivisions),
            "panels": [4 * 4 ** int(s) for s in subdivisions],
            "refinement_ratio_linear": REFINEMENT_RATIO,
            "safety_factor": SAFETY_FACTOR,
            "radius": RADIUS,
            "submergence": SUBMERGENCE,
            "density": DENSITY,
            "gravity": GRAVITY,
            "reference": "tensorbem.analytic.submerged_sphere_wave_multipole "
                         "(Thorne/Wang/Linton multipole series, n_terms=20)",
            "gates": {
                "G1_order_in_range": list(GATE_ORDER),
                "G2_continuum_vs_reference": GATE_CONTINUUM_VS_REF,
                "G3_asme_validation": "E <= 1.25*GCI_1280",
                "G4_economy_error_is_grid_scaling": GATE_ECONOMY_PREDICTION_FACTOR,
            },
        },
        "environment": _env(),
        "cells": cells,
        "gates": gates,
        "verdict": overall,
        "runtime_seconds": time.perf_counter() - started,
        "log": log,
    }


CSV_FIELDS = (
    "ka", "quantity", "panels", "coarse", "medium", "fine", "reference",
    "observed_order", "richardson_continuum", "gci_fine", "err_fine_vs_ref",
    "err_coarse_vs_ref", "err_continuum_vs_ref", "gate_order", "gate_validation",
    "gate_continuum",
)


def write_csv(path: Path, cells: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for cell in cells:
            row = dict(cell)
            row["panels"] = "/".join(str(p) for p in cell["panels"])
            writer.writerow(row)


def render_markdown(result: dict[str, Any]) -> str:
    gates = result["gates"]
    proto = result["protocol"]
    lines = [
        "# Grid-convergence & reference-validation study",
        "",
        f"**Case**: `{result['case']}` · **verdict**: **{result['verdict']}** "
        f"· runtime {result['runtime_seconds']:.1f} s",
        "",
        "Ladder = " + " / ".join(str(p) for p in proto["panels"]) + " constant panels, "
        f"linear refinement ratio r = {proto['refinement_ratio_linear']:g}, "
        f"Fs = {proto['safety_factor']}, band ka = {proto['ka_band']}.",
        "",
        "## Gate summary",
        "",
        "| gate | criterion | result | verdict |",
        "|---|---|---|---|",
        f"| G1 | observed order 1.5 <= p <= 3.0 | "
        f"resolved {gates['G1_order_in_range']['n_resolved']}/"
        f"{gates['G1_order_in_range']['n_cells']} cells, p in "
        f"[{_fmt(gates['G1_order_in_range']['order_min'])}, "
        f"{_fmt(gates['G1_order_in_range']['order_max'])}], median "
        f"{_fmt(gates['G1_order_in_range']['order_median'])}, "
        f"{gates['G1_order_in_range']['n_failures']} failures | "
        f"**{gates['G1_order_in_range']['verdict']}** |",
        f"| G2 | max |f_cont - ref|/|ref| < {GATE_CONTINUUM_VS_REF:.1%} | "
        f"{_pct(gates['G2_continuum_vs_reference']['max_rel_error'])} | "
        f"**{gates['G2_continuum_vs_reference']['verdict']}** |",
        f"| G3 | E = |f_1280-ref|/|ref| <= 1.25 GCI | "
        f"{gates['G3_asme_validation_at_1280']['n_failures']} failures, max E = "
        f"{_pct(gates['G3_asme_validation_at_1280']['max_err_fine_vs_ref'])}, "
        f"max GCI = {_pct(gates['G3_asme_validation_at_1280']['max_gci_fine'])} | "
        f"**{gates['G3_asme_validation_at_1280']['verdict']}** |",
        f"| G4 | economy error within x/÷{GATE_ECONOMY_PREDICTION_FACTOR:g} of "
        f"r^p e_1280 | ratio in [{_fmt(gates['G4_economy_error_is_grid_scaling']['ratio_min'], 2)}, "
        f"{_fmt(gates['G4_economy_error_is_grid_scaling']['ratio_max'], 2)}] | "
        f"**{gates['G4_economy_error_is_grid_scaling']['verdict']}** |",
        "",
        "## Per-cell table (all numbers programmatically emitted)",
        "",
        "| ka | coeff | 80 | 320 | 1280 | ref | p | f_cont | E(1280) | GCI(1280) | "
        "E(cont) | gates |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cell in result["cells"]:
        ok = "".join("✓" if cell[k] else "✗" for k in
                     ("gate_order", "gate_validation", "gate_continuum"))
        level = {lvl["subdivision"]: lvl["value"] for lvl in cell["levels"]}
        cells_txt = " | ".join(_sci(level.get(s)) for s in proto["subdivisions"])
        lines.append(
            f"| {cell['ka']:g} | {cell['quantity']} | {cells_txt} | "
            f"{_sci(cell['reference'])} | {_fmt(cell['observed_order'], 3)} | "
            f"{_sci(cell['richardson_continuum'])} | {_pct(cell['err_fine_vs_ref'])} | "
            f"{_pct(cell['gci_fine'])} | {_pct(cell['err_continuum_vs_ref'])} | {ok} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "`f_cont` is the Richardson continuum limit of the BEM ladder, i.e. the solver "
        "answer with its own discretisation error removed. `E(cont)` is therefore the "
        "*reference* error, not the solver error: it measures the multipole series "
        "against an independent continuum solution rather than against itself.",
        "",
        "`E(1280)` compared with `GCI(1280)` is the ASME V&V 20 validation test: when "
        "E <= U_V the discrete result is consistent with the reference *within its own "
        "numerical uncertainty band*, which is the strongest statement a single "
        "discretisation level can support.",
        "",
    ]
    return "\n".join(lines)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(val):
        return "n/a"
    return f"{val:.{digits}f}"


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    val = float(value)
    if not math.isfinite(val):
        return "n/a"
    return f"{100.0 * val:+.3f}%"


def _sci(value: Any) -> str:
    if value is None:
        return "n/a"
    val = float(value)
    if val != 0.0 and (abs(val) < 1e-3 or abs(val) >= 1e4):
        return f"{val:.6e}"
    return f"{val:.6f}"


def make_figure(result: dict[str, Any], path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = result["cells"]
    panels = result["protocol"]["panels"]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))

    ax = axes[0]
    for quantity, marker in zip(QUANTITIES, ("o", "s", "^", "v")):
        ka_ok = sorted({c["ka"] for c in cells if c["quantity"] == quantity
                        and c["order_resolved"]})
        err = []
        for ka in ka_ok:
            cell = next(c for c in cells if c["quantity"] == quantity and c["ka"] == ka)
            err.append(max(cell["err_fine_vs_ref"], 1e-12))
        ax.semilogy(ka_ok, err, marker + "-", label=quantity)
    ax.axhline(0.05, color="crimson", ls="--", lw=1.0, label="5 % pool gate")
    ax.set_xlabel("ka = $\\omega^2 a / g$")
    ax.set_ylabel("|BEM(1280) - ref| / |ref|")
    ax.set_title("discrete error vs band")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    for quantity, marker in zip(QUANTITIES, ("o", "s", "^", "v")):
        pts = [(c["ka"], c["observed_order"]) for c in cells
               if c["quantity"] == quantity and c["order_resolved"]]
        pts.sort()
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker + "-", label=quantity)
    ax.axhspan(*GATE_ORDER, color="green", alpha=0.12, label="G1 admissible band")
    ax.set_xlabel("ka")
    ax.set_ylabel("observed order p")
    ax.set_title("grid ladder observed order")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ref_cells = [c for c in cells if c["quantity"] == "a33"]
    for cell in sorted(ref_cells, key=lambda c: c["ka"]):
        vals = [lvl["value"] for lvl in cell["levels"]]
        if len(vals) != len(panels):
            continue
        ax.semilogx([1.0 / math.sqrt(p) for p in panels], vals, "o-", lw=0.9,
                    label=f"ka={cell['ka']:g}")
        if math.isfinite(cell["richardson_continuum"]):
            ax.axhline(cell["richardson_continuum"], color="gray", lw=0.5, alpha=0.5)
    ax.set_xlabel("1/sqrt(panels)  ~ h")
    ax.set_ylabel("$A'_{33}$")
    ax.set_title("A33 ladder + Richardson limits (grey)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path,
                        default=Path("docs/case_bem_submerged_sphere_havelock/gci"))
    parser.add_argument("--ka", type=float, nargs="+", default=list(BAND_KA))
    parser.add_argument("--subdivisions", type=int, nargs="+", default=list(SUBDIVISIONS))
    parser.add_argument("--figure", action="store_true")
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args(argv)

    result = build_result(tuple(args.ka), tuple(args.subdivisions))
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "gci.json").write_text(json.dumps(result, indent=2, default=str))
    write_csv(outdir / "table_gci.csv", result["cells"])
    if not args.json_only:
        (outdir / "GCI.md").write_text(render_markdown(result))
    if args.figure:
        make_figure(result, outdir / "fig5_gci.png")

    gates = result["gates"]
    for name, gate in gates.items():
        print(f"{name}: {gate['verdict']}")
    print(f"overall: {result['verdict']}  ({result['runtime_seconds']:.1f} s)")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())