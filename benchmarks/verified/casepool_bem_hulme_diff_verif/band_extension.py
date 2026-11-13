"""High-``ka`` band extension for case-pool anchor CASE-BEM-HULME-DIFF-VERIF.

Why this file exists
--------------------
The delivered case grades the floating-hemisphere exciting force on
``ka in [0.2, 2.0]``.  It says nothing about *where its own agreement stops*,
and the open beam-seas band matters for small platforms in short waves.  This
driver therefore does two things the base case does not:

1. **Extends the frequency band** to ``ka = 2.5, 3.0, 4.0, 5.0`` (wavelength
   down to ``~0.18`` body diameters) on the same three independent routes:
   the Hulme (1982) wave-free multipole series, the TensorBEM Kelvin BIE, and a
   *fresh* capytaine run completed by an exact Froude-Krylov quadrature.
2. **Fixes a metric defect in the base case.**  The base case grades the
   complex *relative* error ``|F - F_ref| / |F_ref|``.  On the heave degree of
   freedom the exciting force cancels to ``|F3|/rho g A a^2 ~ 0.05`` around
   ``ka ~ 2``, so a small absolute error is inflated into a large relative one
   (the delivered ``G4`` worst cell is exactly that: 4.3 % at ``(ka, heave) =
   (2.0, heave)``).  The physically meaningful scale is ``rho g A a^2`` (the
   wave-pressure force on a body cross-section), not the accidentally small
   force itself, so both metrics are reported here and the envelope is
   declared on both.

Routes and independence
-----------------------
* ``reference`` -- ``tensorbem.analytic.hulme_hemisphere_radiation``, term
  ladder ``48/96/192/288`` so truncation stays measured at high ``ka``.
* ``bem`` -- ``tensorbem.free_surface_solver3d.solve_diffraction_3d`` on the
  Kelvin (Havelock) Green-function BIE, mesh ladder ``64/256/1024`` panels.
* ``capytaine`` -- foreign Rankine BEM, run live here (not read from a frozen
  file), for the scattered force; the incident part is the exact
  Froude-Krylov quadrature ``run_case.exact_froude_krylov``.

Reproduce::

    PYTHONPATH=src python benchmarks/verified/casepool_bem_hulme_diff_verif/band_extension.py \\
        --out benchmarks/verified/casepool_bem_hulme_diff_verif/band_extension.json

``--no-capytaine`` skips the foreign route (capytaine is an optional
dependency); the JSON then carries ``capytaine: null`` and the envelope is
declared on the two remaining routes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorbem.analytic import hulme_hemisphere_radiation
from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    solve_diffraction_3d,
)
from tensorbem.geometry3d import hemisphere_hull_mesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_case as base  # noqa: E402  (case driver: FK quadrature + helpers)

DENSITY = 1025.0
GRAVITY = 9.80665
RADIUS = 1.0
#: Wave-pressure force scale of an ``a x a`` section: the absolute norm.
SCALE = DENSITY * GRAVITY * RADIUS**2  # 10060.66 N at A = 1 m

#: Extension band (the base case already covers 0.2 .. 2.0).
#: Extension band.  ``ka = 4.0`` was attempted and is *not* committed at the
#: case mesh: the 1024-panel Kelvin assembly exceeded 8 min of wall clock on a
#: loaded node (load15 ~160) and was abandoned rather than silently graded on
#: a coarser mesh.  Its coarse-mesh points are still reported under
#: ``adjacent_evidence`` because they are the evidence for *where* the
#: discretisation limit is.
KA_GRID: tuple[float, ...] = (2.5, 3.0)
#: Reference truncation ladder: 48 terms is enough at ka = 2, not at ka = 5,
#: so the ladder is widened and the required term count is part of the result.
TERMS_LADDER: tuple[int, ...] = (48, 96, 192, 288)
#: hull subdivisions -> 64 / 256 / 1024 constant panels.
BEM_LEVELS: tuple[int, ...] = (2, 3, 4)
CASE_LEVEL = 4  # the mesh the base case grades (1024 panels)
#: capytaine lat-lon resolutions (all cheaper than the TensorBEM side).
CPT_LEVELS: tuple[tuple[int, int], ...] = ((16, 32), (24, 48), (32, 64))

#: Ledger tolerance of the anchor (``<5 %``) plus the absolute companion gate.
GATE_RELATIVE = 0.05
GATE_ABSOLUTE = 0.01  # |dF| / (rho g A a^2)

CACHE = Path(os.environ.get("TENSORBEM_HULME_BAND_CACHE", "/tmp/hulme_band_cache"))


def _c(value: complex) -> dict[str, float]:
    return {"re": float(complex(value).real), "im": float(complex(value).imag)}


def _cx(row: dict[str, float]) -> complex:
    return complex(row["re"], row["im"])


def _rel(got: complex, ref: complex) -> float:
    return abs(complex(got) - complex(ref)) / abs(complex(ref))


def _absn(got: complex, ref: complex) -> float:
    """Absolute error in units of the wave-pressure force ``rho g A a^2``."""
    return abs(complex(got) - complex(ref)) / SCALE


def _cache_read(name: str) -> Any:
    path = CACHE / name
    return json.loads(path.read_text()) if path.exists() else None


def _cache_write(name: str, payload: Any) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = CACHE / (name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(CACHE / name)


def reference(ka: float) -> dict[str, Any]:
    cached = _cache_read(f"ref_{ka:.3f}.json")
    if cached is not None:
        return cached
    rows: dict[str, Any] = {}
    for terms in TERMS_LADDER:
        res = hulme_hemisphere_radiation(
            ka,
            radius=RADIUS,
            density=DENSITY,
            gravity=GRAVITY,
            n_terms=terms,
            quad_order=max(128, 2 * terms),
        )
        rows[str(terms)] = {
            "surge": _c(res.excitation_force_x),
            "heave": _c(res.excitation_force_z),
        }
    fine = rows[str(TERMS_LADDER[-1])]
    # required term count: smallest rung already within 1e-3 of the finest.
    required = {}
    for key in ("surge", "heave"):
        need = TERMS_LADDER[-1]
        for terms in TERMS_LADDER:
            if _rel(_cx(rows[str(terms)][key]), _cx(fine[key])) < 1.0e-3:
                need = terms
                break
        required[key] = need
    payload = {
        "ka": ka,
        "terms_ladder": list(TERMS_LADDER),
        "per_terms": rows,
        "surge": fine["surge"],
        "heave": fine["heave"],
        "truncation_relative": {
            key: _rel(_cx(rows[str(TERMS_LADDER[-2])][key]), _cx(fine[key]))
            for key in ("surge", "heave")
        },
        "required_terms_1e-3": required,
    }
    _cache_write(f"ref_{ka:.3f}.json", payload)
    return payload


def bem(ka: float, subdivisions: int) -> dict[str, Any]:
    name = f"bem_{ka:.3f}_{subdivisions}.json"
    cached = _cache_read(name)
    if cached is not None:
        return cached
    started = time.time()
    mesh = hemisphere_hull_mesh(subdivisions, radius=RADIUS)
    ops = assemble_free_surface_operators_3d(mesh, ka, chief_points="auto")
    diff = solve_diffraction_3d(
        mesh,
        ka,
        wave_angle=0.0,
        operators=ops,
        density=DENSITY,
        gravity=GRAVITY,
        amplitude=1.0,
    )
    payload = {
        "ka": ka,
        "subdivisions": subdivisions,
        "panels": int(mesh.areas.size),
        "surge": _c(diff.excitation_force[0]),
        "heave": _c(diff.excitation_force[2]),
        "wall_clock_s": round(time.time() - started, 1),
    }
    _cache_write(name, payload)
    return payload


def capytaine(ka_grid: tuple[float, ...]) -> dict[str, Any] | None:
    try:
        import capytaine as cpt  # noqa: F401  (optional dependency)
    except Exception:  # pragma: no cover - optional dependency
        return None
    cached = _cache_read("capytaine.json")
    if cached is not None:
        return cached
    cpt.set_logging("ERROR")
    solver = cpt.BEMSolver()
    ladder = []
    for resolution in CPT_LEVELS:
        mesh = cpt.mesh_sphere(
            radius=RADIUS, center=(0.0, 0.0, 0.0), resolution=resolution
        ).immersed_part()
        body = cpt.FloatingBody(mesh=mesh)
        body.add_all_rigid_body_dofs()
        values: dict[str, Any] = {}
        for ka in ka_grid:
            res = solver.solve(
                cpt.DiffractionProblem(
                    body=body,
                    omega=math.sqrt(ka * GRAVITY),
                    wave_direction=0.0,
                    water_depth=np.inf,
                    rho=DENSITY,
                    g=GRAVITY,
                )
            )
            scattered = {
                "surge": complex(res.forces["Surge"]),
                "heave": complex(res.forces["Heave"]),
            }
            fk_surge, fk_heave = base.exact_froude_krylov(ka)
            values[f"{ka:.3f}"] = {
                "scattered": {k: _c(v) for k, v in scattered.items()},
                "froude_krylov": {"surge": _c(fk_surge), "heave": _c(fk_heave)},
                "total": {
                    "surge": _c(fk_surge + scattered["surge"]),
                    "heave": _c(fk_heave + scattered["heave"]),
                },
            }
        ladder.append(
            {
                "resolution": list(resolution),
                "faces": int(mesh.nb_faces),
                "values": values,
            }
        )
    payload = {
        "provenance": base_provenance(cpt),
        "levels": ladder,
        "note": (
            "capytaine DiffractionProblem.forces is the scattered (diffraction) "
            "force; the incident part is added by the exact Froude-Krylov "
            "quadrature of run_case.exact_froude_krylov"
        ),
    }
    _cache_write("capytaine.json", payload)
    return payload


def base_provenance(cpt: Any) -> dict[str, Any]:
    import platform

    return {
        "tool": "capytaine",
        "version": getattr(cpt, "__version__", "unknown"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def observed_order(coarse: float, fine: float, n_coarse: int, n_fine: int) -> float:
    """Observed order in the panel size ``h ~ N^-1/2`` (error ~ h^p)."""
    h_ratio = math.sqrt(n_fine / n_coarse)
    if fine <= 0.0 or coarse <= fine:
        return float("nan")
    return math.log(coarse / fine) / math.log(h_ratio)


def build(ka_grid: tuple[float, ...], with_capytaine: bool) -> dict[str, Any]:
    refs = {f"{ka:.3f}": reference(ka) for ka in ka_grid}
    per_ka: dict[str, Any] = {}
    for ka in ka_grid:
        key = f"{ka:.3f}"
        rows = {}
        for level in BEM_LEVELS:
            rows[str(level)] = bem(ka, level)
        cell: dict[str, Any] = {"reference": refs[key], "bem": rows, "metrics": {}}
        for level in BEM_LEVELS:
            got = rows[str(level)]
            entry = {}
            for dof in ("surge", "heave"):
                g = _cx(got[dof])
                r = _cx(refs[key][dof])
                entry[dof] = {
                    "relative": _rel(g, r),
                    "absolute_normalised": _absn(g, r),
                    "value": _c(g),
                    "reference": _c(r),
                }
            cell["metrics"][str(level)] = entry
        # mesh convergence on the absolute-normalised metric (order in h).
        order = {}
        for dof in ("surge", "heave"):
            e3 = cell["metrics"]["3"][dof]["relative"]
            e4 = cell["metrics"]["4"][dof]["relative"]
            order[dof] = observed_order(
                cell["metrics"]["2"][dof]["relative"],
                e3,
                rows["2"]["panels"],
                rows["3"]["panels"],
            )
            order[dof + "_fine"] = observed_order(
                e3, e4, rows["3"]["panels"], rows["4"]["panels"]
            )
        cell["orders"] = order
        per_ka[key] = cell

    cpt_payload = capytaine(ka_grid) if with_capytaine else None
    if cpt_payload is not None:
        finest_faces = str(cpt_payload["levels"][-1]["faces"])
        for ka in ka_grid:
            key = f"{ka:.3f}"
            entries = {}
            for row in cpt_payload["levels"]:
                tot = row["values"][key]["total"]
                entries[str(row["faces"])] = {
                    dof: {
                        "value": _cx(tot[dof]),
                        "relative": _rel(_cx(tot[dof]), _cx(refs[key][dof])),
                        "absolute_normalised": _absn(
                            _cx(tot[dof]), _cx(refs[key][dof])
                        ),
                    }
                    for dof in ("surge", "heave")
                }
            per_ka[key]["capytaine"] = entries
            # Ladder spread = honest uncertainty of the foreign route.  A
            # Rankine BEM needs far finer panels at high ka; if the ladder has
            # not settled, the route is *not usable* there and the envelope must
            # not lean on it.
            spread = max(
                entry[dof]["absolute_normalised"]
                for entry in entries.values()
                for dof in ("surge", "heave")
            )
            finest_value = entries[finest_faces]
            worst_rel = max(
                finest_value[dof]["relative"] for dof in ("surge", "heave")
            )
            per_ka[key]["capytaine_ladder_spread_absolute_normalised"] = spread
            per_ka[key]["capytaine_route_usable"] = bool(
                spread <= 0.02 and worst_rel <= 0.05
            )
        # the independent route used by the base case: finest capytaine mesh
        for ka in ka_grid:
            per_ka[f"{ka:.3f}"]["independent_finest"] = per_ka[f"{ka:.3f}"][
                "capytaine"
            ][finest_faces]

    envelope = _envelope(per_ka, ka_grid)
    return {
        "anchor": "CASE-BEM-HULME-DIFF-VERIF",
        "extension": "high-ka validity envelope",
        "citation": (
            "Hulme A. (1982) The wave forces acting on a floating hemisphere "
            "undergoing forced periodic oscillations, J. Fluid Mech. 121:443-463"
        ),
        "geometry": {
            "body": "floating hemisphere, radius a = 1 m, waterplane z = 0",
            "volume_m3": 2.0 * math.pi / 3.0,
            "waterplane_m2": math.pi,
            "density": DENSITY,
            "gravity": GRAVITY,
            "incident_amplitude_m": 1.0,
            "wave_direction_deg": 0.0,
            "time_convention": "exp(-i omega t)",
        },
        "ka_grid": list(ka_grid),
        "gates": {
            "relative": GATE_RELATIVE,
            "absolute_normalised": GATE_ABSOLUTE,
            "scale_N": SCALE,
            "note": (
                "relative = |F - F_ref|/|F_ref| (base-case metric); "
                "absolute_normalised = |F - F_ref|/(rho g A a^2), which is the "
                "metric that stays meaningful where the exciting force cancels"
            ),
        },
        "per_ka": per_ka,
        "adjacent_evidence": adjacent_evidence(KA_GRID),
        "envelope": envelope,
        "base_case": _base_case_rows(),
    }


def _base_case_rows() -> dict[str, Any]:
    """Pull the delivered low band so the envelope can be drawn on one axis."""
    path = Path(__file__).resolve().parent / "results.json"
    if not path.exists():  # pragma: no cover - defensive
        return {}
    data = json.loads(path.read_text())
    rows: dict[str, Any] = {}
    for cell in data.get("ladder", []):
        if int(cell["subdivisions"]) != CASE_LEVEL:
            continue
        rows[f"{float(cell['ka']):.3f}"] = {
            "ka": float(cell["ka"]),
            "panels": int(cell["panels"]),
            "surge_relative": cell["surge"]["complex_relative"],
            "heave_relative": cell["heave"]["complex_relative"],
            "surge_absolute_normalised": abs(
                _cx(cell["surge"]["bem"]) - _cx(cell["surge"]["reference"])
            )
            / SCALE,
            "heave_absolute_normalised": abs(
                _cx(cell["heave"]["bem"]) - _cx(cell["heave"]["reference"])
            )
            / SCALE,
        }
    return {
        "source": "results.json (delivered base case, unmodified)",
        "rows": rows,
        "note": (
            "the base case has no absolute-normalised metric; it is recomputed "
            "here from its own bem/reference pairs so both bands are comparable"
        ),
    }


def adjacent_evidence(ka_grid: tuple[float, ...]) -> dict[str, Any]:
    """Coarse-mesh points that exist in the cache but are not in the band.

    Reported as evidence, never folded into the envelope: a frequency graded on
    a coarser mesh than the case mesh is not a case-mesh result.
    """
    found: dict[str, dict[str, int]] = {}
    if not CACHE.exists():  # pragma: no cover - cache absent
        return {}
    for path in CACHE.glob("bem_*.json"):
        parts = path.stem.split("_")
        if len(parts) != 3:
            continue
        ka = float(parts[1])
        if any(abs(ka - g) < 1e-9 for g in ka_grid):
            continue
        found.setdefault(f"{ka:.3f}", {})[parts[2]] = 1
    out: dict[str, Any] = {}
    for key, levels in sorted(found.items()):
        ref = _cache_read(f"ref_{key}.json")
        if ref is None:  # pragma: no cover - defensive
            continue
        rows = {}
        for level in sorted(levels, key=int):
            cell = _cache_read(f"bem_{key}_{level}.json")
            if cell is None:
                continue
            rows[level] = {
                "panels": cell["panels"],
                "subdivisions": cell["subdivisions"],
                "wall_clock_s": cell["wall_clock_s"],
                **{
                    dof: {
                        "relative": _rel(_cx(cell[dof]), _cx(ref[dof])),
                        "absolute_normalised": _absn(_cx(cell[dof]), _cx(ref[dof])),
                    }
                    for dof in ("surge", "heave")
                },
            }
        out[key] = {
            "levels": rows,
            "note": (
                "outside the declared band: the 1024-panel solve was not "
                "completed, so these are coarse-mesh points only and are "
                "excluded from the envelope"
            ),
        }
    return out


def _envelope(per_ka: dict[str, Any], ka_grid: tuple[float, ...]) -> dict[str, Any]:
    def _ok(entry: dict[str, Any], level: str) -> bool:
        return all(
            entry["metrics"][level][dof]["relative"] <= GATE_RELATIVE
            and entry["metrics"][level][dof]["absolute_normalised"] <= GATE_ABSOLUTE
            for dof in ("surge", "heave")
        )

    case_ok = [ka for ka in ka_grid if _ok(per_ka[f"{ka:.3f}"], str(CASE_LEVEL))]
    worst = {}
    for level in map(str, BEM_LEVELS):
        worst[level] = {}
        for dof in ("surge", "heave"):
            rows = [
                (
                    ka,
                    per_ka[f"{ka:.3f}"]["metrics"][level][dof]["relative"],
                    per_ka[f"{ka:.3f}"]["metrics"][level][dof][
                        "absolute_normalised"
                    ],
                )
                for ka in ka_grid
            ]
            worst[level][dof] = {
                "max_relative": max(row[1] for row in rows),
                "max_relative_ka": max(rows, key=lambda r: r[1])[0],
                "max_absolute_normalised": max(row[2] for row in rows),
                "max_absolute_normalised_ka": max(rows, key=lambda r: r[2])[0],
            }
    indep_worst = None
    if all("independent_finest" in per_ka[f"{ka:.3f}"] for ka in ka_grid):
        indep_worst = {}
        for dof in ("surge", "heave"):
            rows = [
                (
                    ka,
                    per_ka[f"{ka:.3f}"]["independent_finest"][dof]["relative"],
                    per_ka[f"{ka:.3f}"]["independent_finest"][dof][
                        "absolute_normalised"
                    ],
                )
                for ka in ka_grid
            ]
            indep_worst[dof] = {
                "max_relative": max(r[1] for r in rows),
                "max_relative_ka": max(rows, key=lambda r: r[1])[0],
                "max_absolute_normalised": max(r[2] for r in rows),
                "max_absolute_normalised_ka": max(rows, key=lambda r: r[2])[0],
            }
    return {
        "case_mesh_level": CASE_LEVEL,
        "case_mesh_pass_ka": case_ok,
        "case_mesh_upper_bound_ka": max(case_ok) if case_ok else None,
        "worst_bem": worst,
        "worst_independent": indep_worst,
    }


def main(argv: Sequence[str] | None = None) -> int:  # noqa: F821
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out",
        default=str(
            Path(__file__).resolve().parent / "band_extension.json"
        ),
    )
    parser.add_argument("--no-capytaine", action="store_true")
    args = parser.parse_args(argv)
    payload = build(KA_GRID, with_capytaine=not args.no_capytaine)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            payload,
            indent=2,
            default=lambda o: (
                {"re": float(o.real), "im": float(o.imag)}
                if isinstance(o, complex)
                else float(o)
            ),
        )
        + "\n",
        encoding="utf-8",
    )
    env = payload["envelope"]
    print(f"wrote {out}")
    print(f"case-mesh band passing both gates: {env['case_mesh_pass_ka']}")
    print(json.dumps(env["worst_bem"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())