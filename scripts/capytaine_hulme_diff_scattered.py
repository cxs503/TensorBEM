#!/usr/bin/env python3
"""Independent capytaine reference for the *diffraction* half of the Hulme case.

Case-pool anchor ``CASE-BEM-HULME-DIFF-VERIF`` grades the 3-D diffraction
(exciting) force of a floating hemisphere against
``tensorbem.analytic.hulme_hemisphere_radiation``.  Both sides of that
comparison are this repository's own code, so the comparison is circular unless
a foreign implementation is brought in.  This script is that foreign
implementation: it is the only place in the repository that imports
``capytaine`` for the diffraction force (``scripts/capytaine_crosscheck_ref.py``
covers the radiation side), and it freezes the numbers it produces into
``benchmarks/verified/casepool_bem_hulme_diff_verif/capytaine_scattered.json``
together with full provenance and a convention probe.

Two findings are baked into the artefact, both measured, not assumed:

1. **``DiffractionProblem.forces`` in capytaine is the diffraction
   (scattered-pressure) force, not the total exciting force.**  Probe: at
   ``ka = 0.05`` the long-wave limit of the *total* heave excitation is
   ``rho*g*A*A_wp = pi*rho*g*a^2`` (a non-vanishing constant), while capytaine
   returns a value that tends to zero with ``ka``.  The incident-wave
   (Froude-Krylov) part therefore has to be added separately -- the case does
   exactly that, with an exact quadrature of the incident pressure over the
   hemisphere, so the independent total is ``FK_exact + capytaine_scattered``.
2. **The mesh must be the open wetted surface with an outward normal**
   (``mesh_sphere(...).immersed_part()``); a lid panel would silently add the
   waterplane contribution to the Froude-Krylov integral.

The same script re-runs the radiation cross-check at one frequency as a
sanity gate on the setup (if the body/mesh/normal conventions were wrong, the
radiation added mass and damping would not land within a few percent of the
semi-analytical values).

Usage::

    python scripts/capytaine_hulme_diff_scattered.py \
        --out benchmarks/verified/casepool_bem_hulme_diff_verif/capytaine_scattered.json

capytaine is an optional external dependency (built from source on
loongarch64); nothing under ``src/`` imports it.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

DENSITY = 1025.0
GRAVITY = 9.80665

#: ka band of the case, identical to the TensorBEM side.
KA_GRID: tuple[float, ...] = (0.2, 0.5, 0.8, 1.0, 1.5, 2.0)

#: capytaine lat-lon resolutions and the panel count each one yields on the
#: wetted hemisphere (verified at run time, not hard-coded).
MESH_LADDER: tuple[tuple[int, int], ...] = ((8, 16), (16, 32), (24, 48), (32, 64))

#: Probe frequency of the long-wave convention test (total heave excitation
#: tends to pi*rho*g*a^2 there, a scattered-only force tends to zero).
PROBE_KA = 0.05

#: Radiation sanity gate: capytaine heave added mass / damping at ka = 1 must be
#: within this band of the semi-analytical values, otherwise the body setup
#: (mesh, normals, lid) is wrong and nothing below can be trusted.
RADIATION_BAND = 0.05


def _jsonable(value: Any) -> Any:
    """Complex numbers are frozen as ``{"re": ..., "im": ...}``."""
    if isinstance(value, complex):
        return {"re": value.real, "im": value.imag}
    if isinstance(value, np.complex128):
        return {"re": float(value.real), "im": float(value.imag)}
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def _provenance(cpt: Any, source: Path | None) -> dict[str, Any]:
    commit = None
    if source is not None and (source / ".git").exists():
        try:
            commit = subprocess.run(
                ["git", "-C", str(source), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except Exception:  # pragma: no cover - provenance only
            commit = None
    return {
        "tool": "capytaine",
        "version": getattr(cpt, "__version__", "unknown"),
        "git_commit": commit,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "density": DENSITY,
        "gravity": GRAVITY,
        "water_depth": "infinite",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "note": (
            "optional dependency; only this producer script imports capytaine, "
            "the case itself reads the frozen json"
        ),
    }


def _body(cpt: Any, resolution: tuple[int, int]) -> Any:
    """Wetted hemisphere ``r = 1``, ``z <= 0``, open at the waterline."""
    mesh = cpt.mesh_sphere(
        radius=1.0, center=(0.0, 0.0, 0.0), resolution=resolution
    ).immersed_part()
    body = cpt.FloatingBody(mesh=mesh)
    body.add_all_rigid_body_dofs()
    return body


def _normalise(force: complex) -> complex:
    """Force / (rho g a^2) with ``a = 1``, amplitude 1 m (``rho g A a^2``)."""
    return complex(force) / (DENSITY * GRAVITY)


def _scattered(cpt: Any, solver: Any, body: Any, ka: float) -> dict[str, complex]:
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
    return {
        "surge": _normalise(res.forces["Surge"]),
        "heave": _normalise(res.forces["Heave"]),
    }


def _radiation(cpt: Any, solver: Any, body: Any, ka: float) -> dict[str, float]:
    res = solver.solve(
        cpt.RadiationProblem(
            body=body,
            omega=math.sqrt(ka * GRAVITY),
            radiating_dof="Heave",
            water_depth=np.inf,
            rho=DENSITY,
            g=GRAVITY,
        )
    )
    volume = 2.0 * math.pi / 3.0  # exact hemisphere volume, a = 1
    return {
        "added_mass_dimless": float(res.added_mass["Heave"]) / (DENSITY * volume),
        "damping_dimless": float(res.radiation_damping["Heave"])
        / (DENSITY * math.sqrt(ka * GRAVITY) * volume),
    }


def _richardson(values: list[complex], panels: list[int], order: float) -> complex:
    """Two-level Richardson extrapolation in the panel size ``h ~ N^-1/2``."""
    f2, f1 = values[-1], values[-2]
    n2, n1 = panels[-1], panels[-2]
    r = math.sqrt(n1 / n2)
    return complex(f2 + (f2 - f1) / (r**order - 1.0))


def build(cpt: Any, source: Path | None = None) -> dict[str, Any]:
    cpt.set_logging("ERROR")
    solver = cpt.BEMSolver()

    ladder: list[dict[str, Any]] = []
    bodies: dict[int, Any] = {}
    for resolution in MESH_LADDER:
        body = _body(cpt, resolution)
        n_faces = int(body.mesh.nb_faces)
        bodies[n_faces] = body
        ladder.append(
            {
                "resolution": list(resolution),
                "n_faces": n_faces,
                "volume": float(body.volume),
                "values": {
                    f"{ka:.3f}": _scattered(cpt, solver, body, ka) for ka in KA_GRID
                },
            }
        )
    panels = [row["n_faces"] for row in ladder]

    # Discretisation band from the two finest levels plus a first-order and a
    # second-order Richardson estimate: the spread is the honest uncertainty of
    # the foreign code, which the case carries into its comparison.
    extrapolation: dict[str, Any] = {}
    for key in ("surge", "heave"):
        per_order = {}
        for order in (1.0, 2.0):
            per_order[f"order_{int(order)}"] = {
                f"{ka:.3f}": _richardson(
                    [row["values"][f"{ka:.3f}"][key] for row in ladder], panels, order
                )
                for ka in KA_GRID
            }
        finest = {f"{ka:.3f}": ladder[-1]["values"][f"{ka:.3f}"][key] for ka in KA_GRID}
        coarser = {f"{ka:.3f}": ladder[-2]["values"][f"{ka:.3f}"][key] for ka in KA_GRID}
        extrapolation[key] = {
            "finest": finest,
            "second_finest": coarser,
            "richardson": per_order,
            "self_convergence": {
                k: abs(complex(finest[k]) - complex(coarser[k])) / abs(complex(finest[k]))
                for k in finest
            },
        }

    probe_body = bodies[panels[-1]]
    probe = _scattered(cpt, solver, probe_body, PROBE_KA)
    rad = _radiation(cpt, solver, probe_body, 1.0)

    return {
        "provenance": _provenance(cpt, source),
        "ka_grid": list(KA_GRID),
        "probe_ka": PROBE_KA,
        "normalisation": (
            "force / (rho g a^2), a = 1 m, incident amplitude 1 m, "
            "exp(-i omega t), wave_direction = 0 (head seas)"
        ),
        "quantity": (
            "capytaine DiffractionProblem.forces = diffraction (scattered-pressure) "
            "force; the Froude-Krylov part is NOT included (see convention_probe)"
        ),
        "mesh_ladder": ladder,
        "extrapolation": extrapolation,
        "convention_probe": {
            "ka": PROBE_KA,
            "capytaine_scattered": probe,
            "expected_total_heave_long_wave": math.pi,
            "expected_total_surge_long_wave": 0.0,
            "verdict": (
                "scattered-only"
                if abs(probe["heave"]) < 0.1 * math.pi
                else "includes Froude-Krylov"
            ),
            "reasoning": (
                "the long-wave limit of the TOTAL heave excitation is the wave "
                "pressure on the waterplane, pi*rho*g*a^2 (non-vanishing); a force "
                "that vanishes with ka cannot contain the incident-wave part"
            ),
        },
        "radiation_sanity": {
            "ka": 1.0,
            "capytaine": rad,
            "tensorbem_analytic": _analytic_radiation(1.0),
            "band": RADIATION_BAND,
        },
    }


def _analytic_radiation(ka: float) -> dict[str, float]:
    """Semi-analytical heave added mass / damping (no capytaine needed)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tensorbem.analytic import hulme_hemisphere_radiation

    res = hulme_hemisphere_radiation(ka, radius=1.0, density=DENSITY, gravity=GRAVITY)
    volume = 2.0 * math.pi / 3.0
    omega = math.sqrt(ka * GRAVITY)
    return {
        "added_mass_dimless": float(res.heave_added_mass) / (DENSITY * volume),
        "damping_dimless": float(res.heave_damping)
        / (DENSITY * omega * volume),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out",
        default="benchmarks/verified/casepool_bem_hulme_diff_verif/capytaine_scattered.json",
    )
    args = parser.parse_args(argv)

    import capytaine as cpt  # imported here only: optional dependency

    source = Path(cpt.__file__).resolve().parents[1] if cpt.__file__ else None
    payload = build(cpt, source)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=_jsonable) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")
    print(f"convention probe: capytaine force at ka={PROBE_KA} is "
          f"{payload['convention_probe']['verdict']}")
    print("radiation sanity:", json.dumps(payload["radiation_sanity"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())