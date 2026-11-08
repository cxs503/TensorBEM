"""KCS-class hull anchors for the diffraction/radiation verification study (m6656).

What this module is
-------------------
A *thin, auditable* bridge between the published KRISO Container Ship (KCS)
principal particulars and the parametric hull generator that already exists in
:mod:`tensorbem.pc6_hull`.  It does **not** re-implement a hull mesher: the
mesh, its `WettedHullMesh` invariants and the ``Cb``/``L``/``B``/``T`` geometry
gate all come from :func:`tensorbem.pc6_hull.ice_class_hull_mesh` and
:func:`tensorbem.pc6_hull.validate_hull_geometry`, which are already covered by
the repository test suite.

Why a parametric hull instead of KCS offsets
--------------------------------------------
The Tokyo 2003 workshop offset table and its measured (EFD) wave-load data are
**not shipped with this repository**, and the machine this study runs on has no
network access to them.  Typing coordinates from memory would be an
un-auditable transcription, so the hull is generated from the PC6-like family

    b(x, z) = (B/2) * wl(u) * f(v),  u = 2x/L,  v = -z/T,
    wl(u)   = max(1 - |u|**n, f_end),   f(v) = max(1 - (1 - c_k) |v|**q, c_k)

with the parameters solved so that the *analytic* coefficients hit the KCS
values that the family can represent:

======================  =============  ===================  ==================
quantity                KCS published  this generator       representable?
======================  =============  ===================  ==================
``Lpp`` (m)             230.0          230.0                exact
``B`` (m)               32.2           32.2                 exact
``T`` (m)               10.8           10.8                 exact
``Cb``                  0.651          0.651                by solving c_k
``Cm``                  0.981          0.9765 (-0.5 %)     by solving c_k
``Cwp``                 0.80           0.667 (-17 %)        **no**
======================  =============  ===================  ==================

The family has a single waterline exponent ``n``, so ``Cwp = n/(n+1)`` and
``Cp = Cb/Cm`` are locked together (``Cwp == Cp``); a real KCS waterplane
(``Cwp = 0.80 > Cp = 0.664``) needs the section fullness to vary along the
length, which this family cannot express.  The 17 % waterplane deficit is a
*known geometry-fidelity error*, and it is the main reason the KCS EFD
comparison of this study is reported as ``WITHHELD`` rather than as a numerical
result (see ``docs/m6656/README.md`` limitations L2 and L3).

What the hull *is* used for
---------------------------
Solver-internal physics that does not depend on the exact hull form:

* 6-DOF exciting-force reciprocity, near-field pressure integral vs
  Haskind/Kochin far-field formula (:func:`tensorbem.free_surface_solver3d.
  haskind_excitation_force_3d`);
* the long-wave Froude-Krylov limit and the short-wave decay of the exciting
  force;
* the Kochin far-field energy identity for a ship-shaped, surface-piercing,
  non-symmetric hull.
"""

from __future__ import annotations

import math

from .pc6_hull import IceClassHull, ice_class_hull_mesh, validate_hull_geometry
from .geometry3d import WettedHullMesh

#: Tokyo 2003 workshop KCS principal particulars (reference anchors).
KCS_REFERENCE = {
    "length": 230.0,
    "breadth": 32.2,
    "draft": 10.8,
    "block_coefficient": 0.651,
    "midship_coefficient": 0.981,
    "waterplane_coefficient": 0.80,
    "displacement_volume": 230.0 * 32.2 * 10.8 * 0.651,
}

#: Deep-water wavelength ratios ``lambda/L`` (head seas) the study sweeps.
WAVE_LENGTH_RATIOS = (0.65, 0.85, 1.15, 1.50, 1.95)

#: Long-wave ratios used for the Froude-Krylov limit check.
LONG_WAVE_RATIOS = (10.0, 20.0, 40.0)

_DEFAULT_GRAVITY = 9.80665

def kcs_class_hull() -> IceClassHull:
    """PC6-like hull tuned to the KCS principal particulars (see module docstring)."""
    n = 2.0
    q = 2.0
    cb = KCS_REFERENCE["block_coefficient"]
    # Cb = n/(n+1) * (1 - (1-c_k)/(q+1))  ->  solve for c_k
    ck = 1.0 - (q + 1.0) * (1.0 - cb * (n + 1.0) / n)
    return IceClassHull(
        length=KCS_REFERENCE["length"],
        breadth=KCS_REFERENCE["breadth"],
        draft=KCS_REFERENCE["draft"],
        waterline_exponent=n,
        section_exponent=q,
        keel_ratio=ck,
        end_ratio=0.02,
    )


def kcs_class_mesh(
    hull: IceClassHull | None = None,
    *,
    n_sections: int = 41,
    n_side: int = 4,
    n_bottom: int = 3,
) -> WettedHullMesh:
    """Wetted-surface mesh of the KCS-class hull (default 41 stations, ~2 k panels)."""
    if hull is None:
        hull = kcs_class_hull()
    return ice_class_hull_mesh(
        hull, n_sections=n_sections, n_side=n_side, n_bottom=n_bottom
    )


def kcs_geometry_report(
    hull: IceClassHull | None = None, mesh: WettedHullMesh | None = None
) -> dict[str, float]:
    """Mesh-vs-anchor geometry audit for the KCS-class hull.

    The mesh invariants (``Cb`` against the parametric quadrature, ``L``, ``B``,
    ``T``) come from :func:`tensorbem.pc6_hull.validate_hull_geometry`; the
    coefficients are then scored against the *published* KCS values, and the
    waterplane deficit of the family is reported explicitly rather than hidden.
    """
    if hull is None:
        hull = kcs_class_hull()
    if mesh is None:
        mesh = kcs_class_mesh(hull)
    base = validate_hull_geometry(hull, mesh)
    n = hull.waterline_exponent
    cwp_generated = n / (n + 1.0)  # wl(u) = 1 - |u|**n  ->  Cwp = n/(n+1)
    cm_generated = 1.0 - (1.0 - hull.keel_ratio) / (hull.section_exponent + 1.0)
    cb_generated = cwp_generated * cm_generated
    return {
        **{f"mesh_{k}": float(v) for k, v in base.items()},
        "panel_count": float(mesh.panel_count),
        "kcs_length": KCS_REFERENCE["length"],
        "kcs_breadth": KCS_REFERENCE["breadth"],
        "kcs_draft": KCS_REFERENCE["draft"],
        "kcs_block_coefficient": KCS_REFERENCE["block_coefficient"],
        "kcs_midship_coefficient": KCS_REFERENCE["midship_coefficient"],
        "kcs_waterplane_coefficient": KCS_REFERENCE["waterplane_coefficient"],
        "generated_block_coefficient": cb_generated,
        "generated_midship_coefficient": cm_generated,
        "generated_prismatic_coefficient": cb_generated / cm_generated,
        "generated_waterplane_coefficient": cwp_generated,
        "block_coefficient_vs_kcs": abs(
            base["block_coefficient"] - KCS_REFERENCE["block_coefficient"]
        )
        / KCS_REFERENCE["block_coefficient"],
        "midship_coefficient_vs_kcs": abs(
            cm_generated - KCS_REFERENCE["midship_coefficient"]
        )
        / KCS_REFERENCE["midship_coefficient"],
        "waterplane_coefficient_vs_kcs": abs(
            cwp_generated - KCS_REFERENCE["waterplane_coefficient"]
        )
        / KCS_REFERENCE["waterplane_coefficient"],
    }


def wave_number_from_length_ratio(
    ratio: float,
    length: float = KCS_REFERENCE["length"],
    gravity: float = _DEFAULT_GRAVITY,
) -> tuple[float, float]:
    """``(nu, omega)`` of a deep-water wave whose wavelength is ``ratio * L``.

    ``nu = 2*pi/lambda = omega**2 / g`` is the wavenumber the zero-speed 3-D
    solver is driven at; ``omega * sqrt(L/g)`` is the non-dimensional frequency
    quoted in the study tables.  ``ratio -> 0`` is the short-wave limit,
    ``ratio -> inf`` the long-wave (Froude-Krylov) limit.
    """
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("ratio must be finite and positive")
    nu = 2.0 * math.pi / (float(ratio) * float(length))
    return nu, math.sqrt(gravity * nu)


def length_ratio_from_wavenumber(
    nu: float, length: float = KCS_REFERENCE["length"]
) -> float:
    """Inverse of :func:`wave_number_from_length_ratio`."""
    if not math.isfinite(nu) or nu <= 0.0:
        raise ValueError("nu must be finite and positive")
    return 2.0 * math.pi / (float(nu) * float(length))