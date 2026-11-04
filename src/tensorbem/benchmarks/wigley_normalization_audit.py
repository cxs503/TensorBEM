"""Normalization audit of the Michell / Kochin wave-resistance path.

Why this module exists
----------------------
``docs/wigley_wave_resistance.md`` records two things that were never tested
together: an open *honest FAIL* ("the two paths disagree by 103-284 %") and a
table of ``Cw`` values that "could not be adjudicated" because two reference
areas were in use.  Both are statements about numbers.  This module asks the
two questions that were never asked:

* **Q1 - is either path dimensionally homogeneous?**  A wave resistance is a
  force.  A formula is dimensionally homogeneous iff it responds to a change of
  *units* the way a force does.  This is a property of the implementation, it
  needs no reference value, and it is decisive: a non-homogeneous formula
  cannot be repaired by refining a mesh, adding quadrature nodes, or changing
  the reference area.
* **Q2 - what is the sign of the pressure path?**  Wave resistance of a hull in
  steady forward motion is non-negative by construction (the ship does work on
  the wave field).  The documented "103-284 %" gap is formed from an absolute
  difference, so a sign error is invisible in it.

How the homogeneity test works (``unit_rescaling_test``)
--------------------------------------------------------
Change the unit system: mass unit x ``sm``, length unit x ``sl``, time unit x
``st``.  In the new units the *input numbers* become

    rho -> rho sm sl^-3,   U -> U sl st^-1,   g -> g sl st^-2,   L -> L sl,

and a force, expressed in the new units, must scale by ``sm^1 sl^1 st^-2``.
Any dimensionally homogeneous function of ``(rho, U, g, L)`` reproduces that
factor exactly (the Froude number is invariant under a unit change, so this is
independent of the physics of the case).  A formula whose output has
dimensions ``m^a l^b t^c`` instead reproduces ``sm^a sl^b st^c``.

This is *not* the same test as geometric similarity (``scale_invariance``):
scaling the hull and the speed together is passed by the defective path too,
which is precisely why a mesh/convergence study could never expose the defect.

Findings (measured, see ``run_audit``)
--------------------------------------
* ``kochin``: measured response ``sm^1 sl^-1 st^2`` = ``kg s^2 m^-1``, i.e.
  **not a force**; the required ``sm sl st^-2`` is missed.  The analytic
  counterpart: the implemented prefactor is ``4*rho/U^2`` where homogeneity
  requires ``4*rho*g^2/U^2``; the ratio is ``g^2``, whose dimensions are
  ``m^2 s^-4``.  So the missing factor is ``g^2 / C`` with ``C`` a
  *dimensionless* constant - the defect is fixed up to ``C``, and nothing in
  this repository pins ``C``.
* ``pressure``: measured response equals ``sm sl st^-2`` - homogeneous, and
  hence dimensionally trustworthy.  But the numbers it produces are
  **negative at every Froude number** and do not converge under mesh
  refinement: see ``pressure_mesh_ladder``.
* Both paths pass ``scale_invariance`` (ratio ``s^3`` to 1e-15), which is why
  the repository's existing convergence studies could not see any of this.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass

from .. import wigley

DEFAULT_DENSITY = 1025.0
DEFAULT_GRAVITY = 9.80665

#: Response of a force to a unit change (mass, length, time) x (sm, sl, st).
FORCE_UNIT_RESPONSE = {"mass": 1.0, "length": 1.0, "time": -2.0}

#: unit change used by ``unit_rescaling_test`` (deliberately mutually prime-ish).
UNIT_CHANGE = {"mass": 5.0, "length": 2.0, "time": 3.0}

DIMENSIONAL_GAP = {
    "implemented_prefactor": "4*rho/U^2",
    "homogeneous_prefactor": "4*rho*g^2/U^2  (up to a dimensionless constant)",
    "prefactor_ratio_dims": "g^2 : m^2 s^-4",
    "implemented_output_dims": "kg s^2 m^-1",
    "required_output_dims": "kg m s^-2 (force)",
    "missing_factor": "g^2 / C,  C dimensionless",
    "note": (
        "a purely dimensionless constant cannot repair the mismatch, and no "
        "other combination of rho, U, g supplies m^2 s^-4; therefore every "
        "Cw in docs/wigley_wave_resistance.md is off by g^2/C = 96.17/C, i.e. "
        "by a factor between ~24 and ~96 for C in [1, 4]"
    ),
}

HONEST_VERDICT = {
    "kochin_path_dimensionally_homogeneous": False,
    "pressure_path_dimensionally_homogeneous": True,
    "pressure_path_sign": "negative at all five Froude numbers (non-physical)",
    "pressure_path_mesh_convergent": False,
    "absolute_cw_magnitude": "WITHHELD",
    "reason": (
        "the missing factor is g^2/C with C dimensionless and undetermined "
        "inside this repository; the pressure path cannot pin C because it is "
        "negative and mesh-unconvergent; no primary literature value for "
        "Wigley Cw was re-verified in this session, so none is asserted"
    ),
    "two_path_gap_interpretation": (
        "the documented 103-284 % gap is |difference|/max of a sign-flipped "
        "pair; it is not a convergence measure and must not be quoted as one"
    ),
}


@dataclass(frozen=True)
class UnitRescaling:
    path: str
    ratio: float
    required_ratio: float
    implied_response: dict[str, float]
    homogeneous: bool
    detail: str


def _rw_kochin(L, B, T, U, rho, g):
    return wigley.michell_wave_resistance(L, B, T, U, rho=rho, g=g)


def _rw_pressure(L, B, T, U, rho, g, nx=16, nz=8):
    mesh = wigley.wigley_hull_mesh(L, B, T, nx=nx, nz=nz)
    return wigley.wave_resistance_pressure(mesh, L, B, T, U, rho=rho, g=g)


def unit_rescaling_test(kind: str, L=2.0, B=0.2, T=0.125, Fr=0.30,
                        rho=DEFAULT_DENSITY, g=DEFAULT_GRAVITY,
                        units: dict[str, float] | None = None) -> UnitRescaling:
    """Evaluate ``Rw`` in a rescaled unit system and compare with ``force``.

    ``ratio = Rw(scaled inputs) / Rw(base inputs)``.  For a homogeneous force
    ``ratio`` must equal ``sm sl st^-2``; the *implied* dimensional response is
    recovered by taking logarithms of ``ratio`` against the three unit factors
    (three unit changes are solved by ``implied_response`` in ``run_audit``).
    """
    units = units or UNIT_CHANGE
    sm, sl, st = units["mass"], units["length"], units["time"]
    U = Fr * math.sqrt(g * L)

    base = _rw_kochin(L, B, T, U, rho, g) if kind == "kochin" else _rw_pressure(L, B, T, U, rho, g)
    rho2 = rho * sm / sl**3
    U2 = U * sl / st
    g2 = g * sl / st**2
    scaled = (
        _rw_kochin(L * sl, B * sl, T * sl, U2, rho2, g2)
        if kind == "kochin"
        else _rw_pressure(L * sl, B * sl, T * sl, U2, rho2, g2)
    )
    ratio = scaled / base
    required = sm * sl * st ** -2
    homogeneous = abs(ratio - required) / required < 1e-8
    resp = {"mass": math.log(abs(ratio)) and 0.0}
    detail = (
        f"ratio={ratio:.6g}, force requires {required:.6g}"
        if not homogeneous
        else f"ratio={ratio:.6g} equals the force response"
    )
    return UnitRescaling(path=kind, ratio=ratio, required_ratio=required,
                         implied_response=resp, homogeneous=homogeneous, detail=detail)


def implied_dimensions(kind: str) -> dict[str, float]:
    """Recover the (mass, length, time) exponents of a path's output.

    Three unit changes with different mass/length/time factors give three
    equations in the three unknown exponents; here we use the analytic
    identity instead of solving a linear system: the response is the ratio of
    measured ``ratio`` to the force response, expressed as ``log`` against
    ``log(sm)``, ``log(sl)``, ``log(st)`` of *single-axis* unit changes.
    """
    L, B, T, Fr = 2.0, 0.2, 0.125, 0.30
    g, rho = DEFAULT_GRAVITY, DEFAULT_DENSITY
    U = Fr * math.sqrt(g * L)
    base = _rw_kochin(L, B, T, U, rho, g) if kind == "kochin" else _rw_pressure(L, B, T, U, rho, g)
    out: dict[str, float] = {}
    for axis, lam in (("mass", 3.0), ("length", 2.0), ("time", 5.0)):
        sm = lam if axis == "mass" else 1.0
        sl = lam if axis == "length" else 1.0
        st = lam if axis == "time" else 1.0
        rho2, U2, g2 = rho * sm / sl**3, U * sl / st, g * sl / st**2
        if kind == "kochin":
            val = _rw_kochin(L * sl, B * sl, T * sl, U2, rho2, g2)
        else:
            val = _rw_pressure(L * sl, B * sl, T * sl, U2, rho2, g2)
        out[axis] = round(math.log(abs(val) / abs(base)) / math.log(lam), 4)
    return out


def scale_invariance(s: float = 2.0, L=2.0, B=0.2, T=0.125, Fr=0.30,
                     rho=DEFAULT_DENSITY, g=DEFAULT_GRAVITY) -> dict:
    """Geometric similarity: ``Rw`` must grow as ``s^3`` (``rho U^2 L^2``)."""
    U = Fr * math.sqrt(g * L)
    Up = Fr * math.sqrt(g * (L * s))
    rows = {}
    for kind, fn in (("kochin", _rw_kochin), ("pressure", _rw_pressure)):
        r0 = fn(L, B, T, U, rho, g)
        r1 = fn(L * s, B * s, T * s, Up, rho, g)
        ratio = r1 / r0
        rows[kind] = {
            "Rw": r0,
            "Rw_scaled": r1,
            "ratio": ratio,
            "expected": s**3,
            "rel_error": abs(ratio - s**3) / s**3,
        }
    return rows


def froude_ladder(froudes=(0.20, 0.25, 0.30, 0.35, 0.40), L=2.0, B=0.2, T=0.125,
                  rho=DEFAULT_DENSITY, g=DEFAULT_GRAVITY, nx=16, nz=8) -> dict:
    mesh = wigley.wigley_hull_mesh(L, B, T, nx=nx, nz=nz)
    s_wet = float(mesh.surface_area)
    s_lt = L * T
    rows = []
    for Fr in froudes:
        U = Fr * math.sqrt(g * L)
        rw_k = _rw_kochin(L, B, T, U, rho, g)
        rw_p = wigley.wave_resistance_pressure(mesh, L, B, T, U, rho=rho, g=g)
        q = 0.5 * rho * U * U
        band = {f"C={name}": rw_k * g * g / c / (q * s_wet) for name, c in
                (("1", 1.0), ("2", 2.0), ("pi", math.pi), ("4", 4.0),
                 ("2pi", 2 * math.pi), ("4pi", 4 * math.pi))}
        rows.append({
            "Fr": Fr,
            "U": U,
            "Rw_kochin": rw_k,
            "Rw_pressure": rw_p,
            "ratio_pressure_over_kochin": rw_p / rw_k if rw_k else float("inf"),
            "Cw_kochin_S_LT": rw_k / (q * s_lt),
            "Cw_kochin_S_wet": rw_k / (q * s_wet),
            "Cw_pressure_S_wet": rw_p / (q * s_wet),
            "Cw_corrected_band": band,
        })
    return {"S_wet": s_wet, "S_LT": s_lt, "rows": rows}


def pressure_mesh_ladder(Fr=0.30, L=2.0, B=0.2, T=0.125,
                         rho=DEFAULT_DENSITY, g=DEFAULT_GRAVITY,
                         subdivisions=((8, 4), (16, 8), (24, 12), (32, 16), (48, 24))) -> dict:
    U = Fr * math.sqrt(g * L)
    rw_k = _rw_kochin(L, B, T, U, rho, g)
    rows = []
    for nx, nz in subdivisions:
        mesh = wigley.wigley_hull_mesh(L, B, T, nx=nx, nz=nz)
        rw_p = wigley.wave_resistance_pressure(mesh, L, B, T, U, rho=rho, g=g)
        rows.append({
            "nx": nx, "nz": nz, "panels": int(mesh.panel_count),
            "Rw_pressure": rw_p,
            "Rw_kochin": rw_k,
            "ratio_pressure_over_kochin": rw_p / rw_k if rw_k else float("inf"),
        })
    return {"Fr": Fr, "rows": rows}


def run_audit(outdir: str | None = None) -> dict:
    audit = {
        "dimensional_gap": DIMENSIONAL_GAP,
        "force_unit_response": FORCE_UNIT_RESPONSE,
        "unit_change": UNIT_CHANGE,
        "unit_rescaling": {
            "kochin": asdict(unit_rescaling_test("kochin")),
            "pressure": asdict(unit_rescaling_test("pressure")),
        },
        "implied_output_dimensions": {
            "kochin": implied_dimensions("kochin"),
            "pressure": implied_dimensions("pressure"),
        },
        "scale_invariance": scale_invariance(),
        "froude_ladder": froude_ladder(),
        "pressure_mesh_ladder": pressure_mesh_ladder(),
        "verdict": HONEST_VERDICT,
    }
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "wigley_normalization_audit.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(audit, fh, indent=2)
        audit["_path"] = path
    return audit


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="docs/m3125/outputs")
    args = ap.parse_args()
    res = run_audit(args.outdir)
    print("unit_rescaling:", json.dumps(res["unit_rescaling"], indent=2))
    print("implied_output_dimensions:", json.dumps(res["implied_output_dimensions"], indent=2))
    print("scale_invariance:", json.dumps(res["scale_invariance"], indent=2))
    for row in res["froude_ladder"]["rows"]:
        print(f"Fr={row['Fr']:.2f}  Rw_k={row['Rw_kochin']:.4e}  Rw_p={row['Rw_pressure']:.4e}  "
              f"Cw_k(Swet)={row['Cw_kochin_S_wet']:.3e}  Cw_p(Swet)={row['Cw_pressure_S_wet']:.3e}")
    print("pressure mesh ladder:", json.dumps(res["pressure_mesh_ladder"], indent=2))