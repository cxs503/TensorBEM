#!/usr/bin/env python3
"""Does an exact surface measure remove the coarse-mesh bias of the flat-panel BEM?

Anchor: ``CASE-BEM-SUBMERGED-SPHERE-HAVELOCK`` (pool entry
``BEM-SUBMERGED-SPHERE-HAVELOCK``, ref Havelock 1955 multipole, tolerance < 5 %).

Motivation (measured before this study, in
``docs/case_bem_submerged_sphere_havelock/numbers.json``): the 320-panel economy
mesh -- the production mesh predeclared by mission m17242 -- misses the 5 % band
on the *damping* coefficients (band-max 5.59 % on B33'), while the 1280-panel
case mesh passes at 1.43 %.  The committed ``area_deficit_diagnostic`` shows the
error tracks the polyhedral area deficit of the inscribed icosphere
(err_a33 / deficit ~ 0.96, err_b33 / deficit ~ 2.86), i.e. the flat-panel
discretisation integrates over *chords* of the sphere and under-counts the
wetted surface.

Predeclared protocol (fixed before the numbers below were produced)
-------------------------------------------------------------------
Decision
    Is the coarse-mesh damping error a *surface-measure* error that the exact
    spherical-patch measure removes, and does the fix survive a change of mesh,
    submergence ratio and frequency band?
Intervention (single, minimal)
    ``panel_areas = spherical_patch_areas(...)`` threaded through
    ``assemble_free_surface_operators_3d`` / ``solve_radiation_3d``.  Panels,
    panel normals, collocation points and every quadrature rule are unchanged:
    only the integration measure over the body surface changes.
Controls (what a mere renormalisation would NOT explain)
    C1  uniform rescaling of the flat measure, ``panel_areas`` = flat * k with
        ``k = sum(patch) / sum(flat)``.  If the mechanism were a global area
        normalisation, C1 would reproduce the gain of the exact measure.
    C2  the same mesh unchanged (flat measure) -- the reproduction control.
Tests (predeclared, all on meshes/submergences NOT used to derive the fix)
    T1  economy mesh (subdivision 2, 320 panels), band ka = 0.1 ... 5.0
    T2  coarsest mesh (subdivision 1, 80 panels), band ka = 0.1 ... 3.0
    T3  case mesh (subdivision 3, 1280 panels), ka = 0.1, 1.0, 3.0
    T4  submergence ratio f/a = 1.5 and 3.0, economy mesh, ka = 1.0
    T5  low-frequency limit: B33' -> 0 as ka -> 0, economy mesh, exact measure
Threshold
    5 % per item (the pool tolerance), reported for every tier; a tier that
    misses it is reported as missing it.  Success of the *decision* is
    (i) T1 every item < 5 % with the exact measure, (ii) the same tier > 5 %
    without it, (iii) no tier regressing relative to its flat control.
Stopping rule
    One pass over T1..T5 plus the two controls; no mesh, frequency or
    submergence added or dropped after the errors are visible, no threshold
    relaxed.

Reproduce
    PYTHONPATH=src python3 \
        benchmarks/verified/casepool_bem_submerged_sphere_havelock/measure_correction_study.py \
        --outdir docs/case_bem_submerged_sphere_havelock/measure_correction
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorbem.analytic import submerged_sphere_wave_multipole
from tensorbem.free_surface_solver3d import solve_radiation_3d
from tensorbem.geometry3d import spherical_patch_areas, submerged_sphere_mesh

GATE = 0.05
RADIUS = 1.0
DENSITY = 1025.0
GRAVITY = 9.80665
COEFFICIENTS = ("a33", "b33", "a11", "b11")
TIER_BAND = (0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
COARSE_BAND = (0.1, 0.5, 1.0, 2.0, 3.0)
CASE_BAND = (0.1, 1.0, 3.0)
LOW_KA = (0.01, 0.02, 0.05, 0.1, 0.2)


class Tier:
    """One (subdivision, submergence, ka) cell evaluated under several measures."""

    def __init__(self, subdivision: int, submergence: float, radius: float = RADIUS):
        self.subdivision = subdivision
        self.submergence = submergence
        self.radius = radius
        self.mesh = submerged_sphere_mesh(
            subdivision, radius=radius, submergence=submergence
        )
        self.center = np.array([0.0, 0.0, -submergence])
        self.patch = spherical_patch_areas(
            self.mesh.vertices, self.mesh.faces, center=self.center, radius=radius
        )
        k = self.patch.sum() / self.mesh.surface_area
        self.uniform = self.mesh.areas * k
        self._cache: dict[float, dict[str, dict[str, Any]]] = {}

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "subdivision": self.subdivision,
            "n_panels": self.mesh.panel_count,
            "submergence_ratio": self.submergence / self.radius,
            "h_mean": self.mesh.h_mean,
            "flat_surface_area": self.mesh.surface_area,
            "patch_surface_area": float(self.patch.sum()),
            "exact_surface_area": 4.0 * math.pi * self.radius ** 2,
            "area_deficit": 1.0 - self.mesh.surface_area / self.patch.sum(),
            "uniform_factor": float(self.patch.sum() / self.mesh.surface_area),
        }

    def at(self, ka: float) -> dict[str, dict[str, Any]]:
        if ka in self._cache:
            return self._cache[ka]
        nu = ka / self.radius
        ref = submerged_sphere_wave_multipole(
            nu, radius=self.radius, submergence=self.submergence,
            density=DENSITY, gravity=GRAVITY,
        )
        vol = ref.displaced_volume
        omega = ref.omega
        measures = {
            "flat": None,
            "patch": self.patch,
            "uniform": self.uniform,
        }
        out: dict[str, dict[str, Any]] = {}
        for name, measure in measures.items():
            rad = solve_radiation_3d(
                self.mesh, nu, center=self.center.tolist(),
                density=DENSITY, gravity=GRAVITY, panel_areas=measure,
            )
            got = {
                "a33": float(rad.added_mass[2, 2] / (DENSITY * vol)),
                "b33": float(rad.damping[2, 2] / (DENSITY * omega * vol)),
                "a11": float(rad.added_mass[0, 0] / (DENSITY * vol)),
                "b11": float(rad.damping[0, 0] / (DENSITY * omega * vol)),
            }
            refv = {
                "a33": ref.heave_added_mass_dimless,
                "b33": ref.heave_damping_dimless,
                "a11": ref.surge_added_mass_dimless,
                "b11": ref.surge_damping_dimless,
            }
            out[name] = {
                "measure": name,
                "ka": ka,
                "got": got,
                "ref": refv,
                "rel_error": {
                    c: abs(got[c] - refv[c]) / abs(refv[c]) for c in COEFFICIENTS
                },
                "condition_number": rad.condition_number,
                "relative_residual": rad.relative_residual,
            }
        self._cache[ka] = out
        return out


def _band(tier: Tier, band: Sequence[float]) -> list[dict[str, Any]]:
    rows = []
    for ka in band:
        cells = tier.at(ka)
        for measure in ("flat", "patch", "uniform"):
            cell = cells[measure]
            rows.append(
                {
                    "subdivision": tier.subdivision,
                    "n_panels": tier.mesh.panel_count,
                    "submergence_ratio": tier.submergence / tier.radius,
                    "ka": ka,
                    "measure": measure,
                    **{f"{c}": cell["got"][c] for c in COEFFICIENTS},
                    **{f"{c}_ref": cell["ref"][c] for c in COEFFICIENTS},
                    **{f"{c}_rel_error": cell["rel_error"][c] for c in COEFFICIENTS},
                    "band_max_rel_error": max(cell["rel_error"].values()),
                }
            )
    return rows


def _verdict(rows: Sequence[dict[str, Any]], tier_label: str) -> dict[str, Any]:
    out = {}
    for measure in ("flat", "patch", "uniform"):
        sel = [r for r in rows if r["measure"] == measure]
        if not sel:
            continue
        worst = max(sel, key=lambda r: r["band_max_rel_error"])
        n_bad = sum(
            1 for r in sel for c in COEFFICIENTS if r[f"{c}_rel_error"] >= GATE
        )
        out[measure] = {
            "tier": tier_label,
            "n_items": 4 * len(sel),
            "n_items_over_gate": n_bad,
            "band_max_rel_error": worst["band_max_rel_error"],
            "band_max_at": {"ka": worst["ka"], "quantity": max(
                COEFFICIENTS, key=lambda c: worst[f"{c}_rel_error"]
            )},
            "verdict": "PASS" if n_bad == 0 else "FAIL",
        }
    return out


def build_study() -> dict[str, Any]:
    started = time.perf_counter()
    log: list[str] = []
    study: dict[str, Any] = {
        "case_code": "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
        "pool_code": "BEM-SUBMERGED-SPHERE-HAVELOCK",
        "intervention": (
            "panel_areas = tensorbem.geometry3d.spherical_patch_areas(...) threaded "
            "through assemble_free_surface_operators_3d / solve_radiation_3d; panels, "
            "normals, collocation points and quadrature rules unchanged"
        ),
        "gate": GATE,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "tiers": {},
        "controls": {},
        "verdict": {},
    }

    def _record(label: str, tier: Tier, band: Sequence[float], section: str) -> None:
        t0 = time.perf_counter()
        rows = _band(tier, band)
        key = f"{section}:{label}"
        study["tiers"][key] = {"summary": tier.summary, "rows": rows}
        study["verdict"][key] = _verdict(rows, key)
        log.append(f"{key}: {time.perf_counter() - t0:.1f} s")
        print(key, json.dumps(study["verdict"][key], default=float), flush=True)

    _record("economy_s2_band", Tier(2, 2.0), TIER_BAND, "T1")
    _record("coarsest_s1_band", Tier(1, 2.0), COARSE_BAND, "T2")
    _record("case_s3_band", Tier(3, 2.0), CASE_BAND, "T3")
    _record("economy_s2_fa1.5", Tier(2, 1.5), (1.0,), "T4")
    _record("economy_s2_fa3.0", Tier(2, 3.0), (1.0,), "T4")
    low = Tier(2, 2.0)
    _record("economy_s2_low_ka", low, LOW_KA, "T5")

    # T5 also asks whether B33' itself vanishes as ka -> 0 (it is O(ka^? ) in the
    # wave-free limit); report the raw numbers so the trend is auditable.
    study["low_frequency_limit"] = [
        {"ka": ka, "b33": low.at(ka)["patch"]["got"]["b33"],
         "b33_ref": low.at(ka)["patch"]["ref"]["b33"]}
        for ka in LOW_KA
    ]

    study["runtime_seconds"] = time.perf_counter() - started
    study["log"] = log
    return study


def write_artifacts(study: dict[str, Any], outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    numbers = outdir / "numbers.json"
    numbers.write_text(json.dumps(study, indent=2, default=float) + "\n")
    written.append(numbers)

    fields = (
        "subdivision", "n_panels", "submergence_ratio", "ka", "measure",
        "a33", "a33_ref", "a33_rel_error", "b33", "b33_ref", "b33_rel_error",
        "a11", "a11_ref", "a11_rel_error", "b11", "b11_ref", "b11_rel_error",
        "band_max_rel_error",
    )
    for key, payload in study["tiers"].items():
        path = outdir / f"table_{key.replace(':', '_').replace('.', 'p')}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in payload["rows"]:
                writer.writerow({k: row[k] for k in fields})
        written.append(path)

    verdict = outdir / "verdict.json"
    verdict.write_text(json.dumps(study["verdict"], indent=2) + "\n")
    written.append(verdict)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", type=Path,
                        default=Path("docs/case_bem_submerged_sphere_havelock/measure_correction"))
    args = parser.parse_args(argv)
    study = build_study()
    written = write_artifacts(study, args.outdir)
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())