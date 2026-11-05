"""Machine-checked pins for every number printed in ``docs/finite_depth3d.md``.

Why this module exists
----------------------
``docs/finite_depth3d.md`` is the specification/verification note of the 3-D
finite-depth Green function (John eigenseries).  It quotes 112 numbers: a
dispersion ladder, the free-surface and bottom residuals of the kernel, the
deep-water limit against the independent Kelvin kernel, the far-field
structure, and the singularity/truncation ladders.  Those numbers were written
by ``docs/m3015/finite_depth3d_report.py`` at the time of the change, and
**nothing read them afterwards** -- a coverage audit
(:mod:`tensorbem.benchmarks.doc_claim_audit`) classified the document as
``UNGUARDED``.  That is precisely the failure mode that produced three earlier
incidents in this package (see ``docs/ursell_quadrature_provenance.md``): a
kernel revision moves the numbers, the document keeps the old ones, and nobody
notices until the next audit.

This module closes the hole from the *document end*: it parses the values out
of the markdown tables and recomputes each one from
:mod:`tensorbem.finite_depth3d` / :mod:`tensorbem.free_surface3d`.  A code
change that moves a number, or a hand edit of the document, becomes a
``DRIFT`` row.

Pinned tables
-------------
====================================  ========================================
§3.1 dispersion ladder                ``k0``, ``c``, ``cg`` per depth
§3.2 kernel boundary conditions       free-surface and bottom residuals
§3.3 deep-water limit                 max deviation from the Kelvin kernel
§3.4 far field                        spreading flatness, phase slope
§3.5 singularity + truncation         ``4 pi r Re G`` ladder, mode ladder
====================================  ========================================

Tolerances come from the printed precision (three or more significant digits),
not from physics, with two exceptions that cannot be gated relatively:

``exact zero`` (``0.000e+00``)
    the bottom residual, judged against :data:`ZERO_FLOOR` (5e-16).
``roundoff zero`` (``0 < |declared| <= NOISE_FLOOR``)
    the free-surface residual, ``1e-15``-level cancellation noise of an ``O(1)``
    kernel.  A 17 % *relative* change there is 2 ulp, not a regression, so the
    row is judged absolutely against :data:`NOISE_FLOOR` (1e-12) -- still six
    orders of magnitude below any physical residual.  This is not a loophole:
    the mathematically neutral rewrite of the propagating mode (``cosh`` ->
    log-scaled exponentials, m3409) moved those three residuals by 1-3 ulp and
    would otherwise have turned the gate permanently red.

Usage::

    python -m tensorbem.benchmarks.finite_depth3d_pins \
        --json docs/finite_depth3d_pins/check.json

``--strict`` exits non-zero on any DRIFT; ``tests/test_finite_depth3d_pins.py``
runs the same check in CI.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tensorbem.finite_depth import finite_depth_wavenumber
from tensorbem.finite_depth3d import (

    dispersion_residual,
    green_function_3d,
    green_gradient_z_3d,
)
from tensorbem.free_surface3d import green_function_3d as deep_green_3d

REPO_ROOT = Path(__file__).resolve().parents[3]
DOC_PATH = REPO_ROOT / "docs" / "finite_depth3d.md"

SCHEMA = "tensorbem.finite_depth3d_pins/1"
NU = 0.6
G = 9.80665
#: Absolute floor for rows whose printed value is ``0.000e+00``.
ZERO_FLOOR = 5.0e-16
#: Absolute floor for *residual* rows whose printed value is roundoff noise.
#: A residual is a difference that should be zero, so at ``1e-15`` it is
#: cancellation noise of an ``O(1)`` kernel and carries no digit a relative
#: comparison could protect; a residual regression that actually broke the
#: boundary condition is orders of magnitude above this floor and still drifts.
#: Applied to ``*residual*`` quantities only: a small *physical* value (e.g. the
#: evanescent share, ``4.6e-20``) keeps its relative tolerance.
NOISE_FLOOR = 1.0e-12


@dataclass(frozen=True)
class Row:
    table: str
    key: str
    quantity: str
    declared: float
    recomputed: float
    rtol: float
    verdict: str

    @property
    def is_roundoff_zero(self) -> bool:
        """True when the printed value is cancellation noise, not a measurement."""
        if abs(self.declared) > NOISE_FLOOR:
            return False
        if abs(self.declared) < 1e-300:
            return False  # printed exact zero: ZERO_FLOOR applies
        return "residual" in self.quantity

    @property
    def abs_deviation(self) -> float:
        return abs(self.recomputed - self.declared)

    @property
    def rel_deviation(self) -> float:
        scale = abs(self.declared)
        if scale < 1e-300:
            return abs(self.recomputed) if abs(self.recomputed) <= ZERO_FLOOR else math.inf
        if self.is_roundoff_zero:
            # relative difference of two roundoff values is meaningless
            return 0.0 if abs(self.recomputed) <= NOISE_FLOOR else math.inf
        return abs(self.recomputed - self.declared) / scale


# ---------------------------------------------------------------------------
# markdown parsing
# ---------------------------------------------------------------------------
_HEADING_RE = re.compile(r"^#+\s*(?P<title>.*)$")
_NUMBER_RE = re.compile(r"[-\u2212]?\d+(?:\.\d+)?(?:[eE][-\u2212]?\d+)?")


def parse_number(token: str) -> float:
    return float(token.replace("\u2212", "-"))


def split_sections(text: str) -> dict[str, list[str]]:
    """Map heading title -> the lines that follow it until the next heading."""
    sections: dict[str, list[str]] = {}
    current = ""
    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            current = match.group("title").strip()
            sections[current] = []
        else:
            sections.setdefault(current, []).append(line)
    return sections


def table_rows(lines: list[str]) -> list[list[str]]:
    """Rows of the first markdown table found in ``lines`` (header + body)."""
    rows: list[list[str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            if rows:
                break
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue
        rows.append(cells)
    return rows


def _numbers(cell: str) -> list[float]:
    return [parse_number(t) for t in _NUMBER_RE.findall(cell)]


# ---------------------------------------------------------------------------
# recomputation (mirrors docs/m3015/finite_depth3d_report.py, same settings)
# ---------------------------------------------------------------------------
def dispersion_ladder(depths=(0.05, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 20.0, 100.0)):
    out = {}
    for depth in depths:
        k0 = finite_depth_wavenumber(NU, depth)
        omega = math.sqrt(NU * G)
        c = omega / k0
        cg = 0.5 * c * (1.0 + 2.0 * k0 * depth / math.sinh(2.0 * k0 * depth))
        out[depth] = {"k0": k0, "c": c, "cg": cg, "residual": dispersion_residual(k0, NU, depth)}
    return out


def boundary_condition_rows(depths=(2.0, 4.0, 10.0, 40.0)):
    targets_fs = np.stack([np.linspace(0.3, 9.0, 14), np.zeros(14), np.zeros(14)], axis=-1)
    out = {}
    for depth in depths:
        source = np.array([0.0, 0.0, -0.5 * depth])
        value = green_function_3d(targets_fs, source, NU, depth)
        grad = green_gradient_z_3d(targets_fs, source, NU, depth)
        fs = float(np.max(np.abs(grad - NU * value)) / np.max(np.abs(NU * value)))
        tb = np.stack(
            [np.linspace(0.4, 6.0, 10), np.zeros(10), -depth * np.ones(10)], axis=-1
        )
        bottom = float(
            np.max(np.abs(green_gradient_z_3d(tb, source, NU, depth)))
            / (np.max(np.abs(green_function_3d(tb, source, NU, depth))) / depth)
        )
        out[depth] = {"free_surface": fs, "bottom": bottom}
    return out


def deep_water_ladder(depths=(2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)):
    points = np.array(
        [[0.5, 0.0, -1.0], [2.0, 0.0, 0.0], [3.0, 1.0, -1.2], [6.0, 0.0, -0.5], [9.0, 0.0, -1.0]]
    )
    source = np.array([0.0, 0.0, -2.0])
    reference = deep_green_3d(points, source, NU)
    out = {}
    for depth in depths:
        finite = green_function_3d(points, source, NU, depth)
        out[depth] = {
            "k0_h": finite_depth_wavenumber(NU, depth) * depth,
            "max_rel_error": float(
                np.max(np.abs(finite - reference)) / np.max(np.abs(reference))
            ),
        }
    return out


def far_field_values(depth=4.0):
    k0 = finite_depth_wavenumber(NU, depth)
    radii = np.linspace(8.0, 90.0, 60)
    targets = np.stack([radii, np.zeros_like(radii), np.zeros_like(radii)], axis=-1)
    source = np.array([0.0, 0.0, -2.0])
    full = green_function_3d(targets, source, NU, depth)
    evanescent = green_function_3d(targets, source, NU, depth, include_wave=False)
    ratio = np.abs(evanescent) / np.abs(full)
    window = radii > 25.0
    amplitude = np.abs(full) * np.sqrt(k0 * radii)
    phase = np.unwrap(np.angle(full))
    slope = float(np.polyfit(k0 * radii[window], phase[window], 1)[0])
    probes = {4.9: 0.0, 15.0: 1.0, 25.2: 2.0, 35.4: 3.0, 45.5: 4.0}
    evanescent_share = {}
    k0_r = k0 * radii
    for label, idx in probes.items():
        nearest = int(np.argmin(np.abs(k0_r - label)))
        evanescent_share[label] = float(ratio[nearest])
    return {
        "k0": k0,
        "spreading_ptp_over_mean": float(np.ptp(amplitude[window]) / np.mean(amplitude[window])),
        "phase_slope": slope,
        "evanescent_share": evanescent_share,
    }


def singularity_ladder(radii=(0.04, 0.02, 0.01, 0.005, 0.0025), depth=5.0):
    """``4 pi r Re G`` at a horizontal offset ``r`` from a source at ``(0,0,-2)``."""
    source = np.array([0.0, 0.0, -2.0])
    out = {}
    for radius in radii:
        target = np.array([[radius, 0.0, -2.0]])
        value = green_function_3d(target, source, NU, depth)
        out[radius] = float(4.0 * math.pi * radius * float(np.real(value[0])))
    return out


def truncation_ladder(modes=(40, 80, 160, 320, 640)):
    """Max relative change of ``G`` against the 1200-mode reference."""
    probe = np.array([[0.7, 0.0, -0.4], [1.5, 1.0, -2.0]])
    source = np.array([0.0, 0.0, -1.0])
    reference = green_function_3d(probe, source, NU, 4.0, modes=1200)
    out = {}
    for count in modes:
        approx = green_function_3d(probe, source, NU, 4.0, modes=count)
        out[count] = float(
            np.max(np.abs(approx - reference)) / np.max(np.abs(reference))
        )
    return out


# ---------------------------------------------------------------------------
# the pinned tables
# ---------------------------------------------------------------------------
def _judge(
    declared: float, recomputed: float, rtol: float, quantity: str = ""
) -> tuple[str, float]:
    if abs(declared) < 1e-300:
        ok = abs(recomputed) <= ZERO_FLOOR
        return ("PASS" if ok else "DRIFT"), abs(recomputed)
    if "residual" in quantity and abs(declared) <= NOISE_FLOOR:
        # roundoff residual: only its absolute size is meaningful
        ok = abs(recomputed) <= NOISE_FLOOR
        return ("PASS" if ok else "DRIFT"), abs(recomputed - declared)
    dev = abs(recomputed - declared) / abs(declared)
    return ("PASS" if dev <= rtol else "DRIFT"), dev


def collect(doc_path: Path = DOC_PATH) -> list[Row]:
    text = doc_path.read_text(encoding="utf-8")
    sections = split_sections(text)
    rows: list[Row] = []

    def add(table: str, key: str, quantity: str, declared: float, recomputed: float, rtol: float):
        verdict, _ = _judge(declared, recomputed, rtol, quantity)
        rows.append(Row(table, key, quantity, declared, recomputed, rtol, verdict))

    # --- 3.1 dispersion ladder ------------------------------------------------
    section = next((v for k, v in sections.items() if k.startswith("3.1")), [])
    ladder = dispersion_ladder()
    for cells in table_rows(section):
        if not cells or cells[0].lower().startswith("$h"):
            continue
        numbers = _numbers(cells[0])
        if not numbers:
            continue
        depth = numbers[0]
        if depth not in ladder:
            continue
        values = _numbers(" ".join(cells[1:]))
        if len(values) < 5:
            continue
        for quantity, declared, rtol in (
            ("k0", values[0], 1e-6),
            ("c", values[2], 5e-4),
            ("cg", values[3], 5e-4),
        ):
            add("3.1", f"h={depth:g}", quantity, declared, ladder[depth][quantity], rtol)

    # --- 3.2 boundary conditions ---------------------------------------------
    section = next((v for k, v in sections.items() if k.startswith("3.2")), [])
    bc = boundary_condition_rows()
    for cells in table_rows(section):
        numbers = _numbers(cells[0]) if cells else []
        if not numbers:
            continue
        depth = numbers[0]
        if depth not in bc:
            continue
        values = _numbers(" ".join(cells[1:]))
        if len(values) < 2:
            continue
        add("3.2", f"h={depth:g}", "free_surface_residual", values[0], bc[depth]["free_surface"], 5e-3)
        add("3.2", f"h={depth:g}", "bottom_residual", values[1], bc[depth]["bottom"], 5e-3)

    # --- 3.3 deep-water limit ------------------------------------------------
    section = next((v for k, v in sections.items() if k.startswith("3.3")), [])
    deep = deep_water_ladder()
    for cells in table_rows(section):
        numbers = _numbers(cells[0]) if cells else []
        if not numbers:
            continue
        depth = numbers[0]
        if depth not in deep:
            continue
        values = _numbers(" ".join(cells[1:]))
        if len(values) < 2:
            continue
        add("3.3", f"h={depth:g}", "k0_h", values[0], deep[depth]["k0_h"], 5e-3)
        add(
            "3.3",
            f"h={depth:g}",
            "max_rel_error",
            values[1],
            deep[depth]["max_rel_error"],
            5e-3,
        )

    # --- 3.4 far field -------------------------------------------------------
    far = far_field_values()
    match = re.search(
        r"spreading:\s*\$\\mathrm\{ptp\}\(\|G\|\\sqrt\{k_0R\}\)/\\mathrm\{mean\}\s*=\s*"
        r"([-\d.eE\u2212]+)\$",
        text,
    )
    if match:
        add(
            "3.4",
            "far_field",
            "spreading_ptp_over_mean",
            parse_number(match.group(1)),
            far["spreading_ptp_over_mean"],
            5e-3,
        )
    match = re.search(
        r"phase slope\s*\$\\mathrm\{d\\arg G\}/\\mathrm\{d\}\(k_0R\)\s*=\s*"
        r"([-\d.eE\u2212]+)\$",
        text,
    )
    if match:
        add("3.4", "far_field", "phase_slope", parse_number(match.group(1)), far["phase_slope"], 5e-4)

    # evanescent share bullet: "4.3e-03 at k0R=4.9, 2.5e-07 at k0R=15.0, ..."
    bullet = re.search(r"evanescent share[^\n]*\|G\|[^\n]*?\$?:\s*([^\n]*)", text)
    if bullet:
        for value, probe in re.findall(
            r"([-\d.eE\u2212]+)\s*at \$k_0R\$=(\d+(?:\.\d+)?)", bullet.group(1)
        ):
            key = float(probe)
            nearest = min(far["evanescent_share"], key=lambda k: abs(k - key))
            if abs(nearest - key) < 0.6:
                add(
                    "3.4",
                    f"k0R={key:g}",
                    "evanescent_share",
                    parse_number(value),
                    far["evanescent_share"][nearest],
                    5e-2,
                )

    # --- 3.5 singularity + truncation ---------------------------------------
    section = next((v for k, v in sections.items() if k.startswith("3.5")), [])
    tables: list[list[list[str]]] = []
    bucket: list[list[str]] = []
    for line in section:
        if line.strip().startswith("|"):
            bucket.append(line)
        elif bucket:
            tables.append(table_rows(bucket))
            bucket = []
    if bucket:
        tables.append(table_rows(bucket))

    singular = singularity_ladder()
    if tables:
        for cells in tables[0]:
            numbers = _numbers(" ".join(cells))
            if len(numbers) < 2:
                continue
            radius = numbers[0]
            nearest = min(singular, key=lambda r: abs(r - radius))
            if abs(nearest - radius) < 1e-9:
                add("3.5", f"r={radius:g}", "four_pi_r_ReG", numbers[1], singular[nearest], 5e-4)

    truncation = truncation_ladder()
    if len(tables) > 1:
        for cells in tables[1]:
            numbers = _numbers(" ".join(cells))
            if len(numbers) < 2:
                continue
            count = int(numbers[0])
            if count in truncation:
                add("3.5", f"modes={count}", "max_rel_change", numbers[1], truncation[count], 5e-2)
    return rows


def summary(rows: list[Row]) -> dict:
    drift = [r for r in rows if r.verdict == "DRIFT"]
    worst = max(
        (r for r in rows if abs(r.declared) > NOISE_FLOOR or not r.is_roundoff_zero),
        key=lambda r: r.rel_deviation,
        default=None,
    )
    return {
        "schema": SCHEMA,
        "document": str(DOC_PATH.relative_to(REPO_ROOT)),
        "rows": len(rows),
        "pass": len(rows) - len(drift),
        "drift": len(drift),
        "worst": (
            {
                "table": worst.table,
                "key": worst.key,
                "quantity": worst.quantity,
                "declared": worst.declared,
                "recomputed": worst.recomputed,
                "rel_deviation": worst.rel_deviation,
            }
            if worst is not None
            else None
        ),
        "max_rel_deviation": max((r.rel_deviation for r in rows), default=0.0),
        "roundoff_zero_rows": sum(r.is_roundoff_zero for r in rows),
        "noise_floor": NOISE_FLOOR,
        "tables_covered": sorted({r.table for r in rows}),
    }


def check(doc_path: Path = DOC_PATH) -> tuple[dict, list[Row]]:
    rows = collect(doc_path)
    return summary(rows), rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", type=Path, default=DOC_PATH)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report, rows = check(args.doc)
    for row in rows:
        if row.verdict == "DRIFT":
            if row.is_roundoff_zero:
                scale = (
                    f"(|abs {row.abs_deviation:.2e}| with recomputed "
                    f"{abs(row.recomputed):.2e} > NOISE_FLOOR {NOISE_FLOOR:.1e})"
                )
            else:
                scale = f"(rel {row.rel_deviation:.2e} > {row.rtol:.1e})"
            print(
                f"DRIFT {row.table} {row.key} {row.quantity}: "
                f"declared={row.declared:.6e} recomputed={row.recomputed:.6e} {scale}"
            )
    print(
        f"{report['rows']} pinned values, {report['pass']} PASS, {report['drift']} DRIFT, "
        f"max relative deviation {report['max_rel_deviation']:.2e} "
        f"tables {report['tables_covered']}"
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": report, "rows": [asdict(r) for r in rows]}
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 1 if (args.strict and report["drift"]) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
