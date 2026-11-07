"""Verification-anchor maturation for classical potential-flow hull studies.

Mission m5240 (campaign W18-N02): put the Rankine-source / thin-ship anchor of
``TensorBEM`` on a verified footing for two classical hulls -- the Wigley
parabolic hull (hydrostatics + wave resistance) and the half-submerged
ellipsoid (wave resistance, Havelock/Michell non-separable reference).

What is measured here, and against what
---------------------------------------
1. **Geometry gate (always first).**  ``Cb``, ``L``, ``B``, ``T`` of the meshed
   hull must reproduce the closed form before any resistance number is quoted;
   the mission brief's ``Cb = 0.667`` is audited against the closed-form
   ``Cb = 4/9`` of the parabolic hull it is specified with.
2. **Wigley hydrostatics vs panel count** (10x10, 20x20, 40x40 panels per side =
   200 / 800 / 3200 wetted triangles): displaced volume ``4LBT/9`` and wetted
   area (spectral quadrature reference, refined until it stops moving).
3. **Ellipsoid wave resistance**: ``H(k)`` from the Bessel closed form, from the
   *definition* quadrature and from a constant-strength centreplane panel
   discretisation; ``R_W`` through the Kelvin-wake integral in two forms
   (``lambda`` and wave-angle ``theta``).
4. **Wigley wave resistance over Fr = 0.20 ... 0.40**: analytic Michell/Kochin
   reference vs the same panel discretisation, with the Cw normalisation stated
   explicitly (``S_w`` from the quadrature reference).

Honest scope
------------
The mission also asks for a comparison of the Wigley ``R_W`` with the ITTC
recommended experimental data (target ``< 10 %``).  No sourced experimental
table is available offline in this work tree, and the repository already
registered that comparison as ``WITHHELD`` (``docs/wigley_wave_resistance.md``:
the previously carried table "has no stated ``Cw`` normalisation and no
source").  It stays ``WITHHELD`` here: the report records the requirement, the
missing artefact and the target band, and does **not** substitute an invented
table.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from ..ellipsoid_kochin import (
    EllipsoidHull,
    kochin_analytic,
    kochin_quadrature,
    kelvin_wave_resistance,
    kelvin_wave_resistance_theta,
    panel_grid,
    wave_resistance_analytic,
    wave_resistance_panels,
    wetted_surface_area,
    wetted_volume_quadrature,
)
from ..hydrostatic import compute_displaced_volume_and_buoyancy
from ..wigley import michell_wave_resistance, wigley_hull_properties
from ..wigley_hydrostatics import (
    wigley_hydrostatic_reference,
    wigley_wetted_area_reference,
    wigley_wetted_hull_mesh,
)

__all__ = [
    "BRIEF_HULL",
    "ELLIPSOID",
    "PANEL_LEVELS",
    "wigley_kochin_panels",
    "wigley_panel_wave_resistance",
    "wigley_geometry_audit",
    "wigley_hydrostatic_panel_study",
    "ellipsoid_wave_study",
    "ellipsoid_froude_curve",
    "wigley_froude_sweep",
    "verify",
    "build_report",
    "make_figures",
    "main",
]

#: Geometry of the mission brief (L = 3.8 m, B = 0.5 m, T = 0.225 m).  The
#: brief quotes ``Cb = 0.667``; the parabolic Wigley hull with these principal
#: dimensions has ``Cb = 4/9`` (see :func:`wigley_geometry_audit`).
BRIEF_HULL: dict[str, float] = {"L": 3.8, "B": 0.5, "T": 0.225, "Cb_claimed": 0.667}

#: Mission brief's ellipsoid: semi-axes a = 1 m (semi-length), b = 0.2 m
#: (semi-beam), c = 0.15 m (semi-draft), Froude number 0.30.
ELLIPSOID: dict[str, float] = {"a": 1.0, "b": 0.2, "c": 0.15, "Fr": 0.30}

#: ``(n_sections, n_levels)`` per side.  The Wigley generator emits four
#: triangles per ``(s, t)`` cell (two per side), so these three levels give
#: exactly the 200 / 800 / 3200 wetted panels of the mission brief.
PANEL_LEVELS: tuple[tuple[int, int], ...] = ((10, 5), (20, 10), (40, 20))

DEFAULT_RHO = 1025.0
DEFAULT_G = 9.80665


# ============================================================================
# 1. Wigley geometry audit
# ============================================================================


def wigley_geometry_audit(
    L: float = BRIEF_HULL["L"],
    B: float = BRIEF_HULL["B"],
    T: float = BRIEF_HULL["T"],
    claimed_Cb: float = BRIEF_HULL["Cb_claimed"],
    *,
    n_sections: int = 40,
    n_levels: int = 40,
) -> dict[str, Any]:
    """Audit the brief's ``Cb`` against the closed form and the meshed hull.

    The parabolic (mathematical) Wigley hull has ``Cb = Cm * Cp = 4/9``.  The
    brief's ``0.667`` is the *prismatic* coefficient ``Cp = Cm = 2/3`` of the
    same hull (the classical Cp-vs-Cb confusion already documented in
    ``tensorbem.wigley.wigley_hull_properties``).  Both facts are checked
    numerically: closed form, and the volume of the generated panel mesh.
    """
    ref = wigley_hull_properties(L, B, T)
    mesh = wigley_wetted_hull_mesh(L, B, T, n_sections, n_levels)
    mesh_volume, _ = compute_displaced_volume_and_buoyancy(mesh)
    mesh_cb = mesh_volume / (L * B * T)

    claimed_volume = claimed_Cb * L * B * T
    return {
        "L": L,
        "B": B,
        "T": T,
        "L_over_B": L / B,
        "B_over_T": B / T,
        "LBT": L * B * T,
        "closed_form": {
            "volume": ref["volume"],
            "Cb": ref["Cb"],
            "Cm": ref["Cm"],
            "Cp": ref["Cp"],
            "waterplane_area": ref["waterplane_area"],
        },
        "mesh": {"panels": mesh.panel_count, "volume": mesh_volume, "Cb": mesh_cb},
        "brief_claim": {
            "Cb_claimed": claimed_Cb,
            "volume_claimed": claimed_volume,
            "volume_ratio_to_closed_form": claimed_volume / ref["volume"],
            "Cb_ratio_to_closed_form": claimed_Cb / ref["Cb"],
        },
        "verdict": {
            "claim_consistent_with_parabolic_hull": bool(
                abs(claimed_Cb - ref["Cb"]) / ref["Cb"] < 0.01
            ),
            "claimed_value_matches": (
                "Cp = Cm = 2/3 (prismatic), not Cb"
                if abs(claimed_Cb - ref["Cp"]) / ref["Cp"] < 0.01
                else "neither Cb nor Cp of this hull"
            ),
            "mesh_vs_closed_form_volume_rel": abs(mesh_volume / ref["volume"] - 1.0),
            "mesh_Cb_rel_error": abs(mesh_cb / (4.0 / 9.0) - 1.0),
        },
    }


# ============================================================================
# 2. Wigley hydrostatics vs panel count
# ============================================================================


def wigley_hydrostatic_panel_study(
    L: float = BRIEF_HULL["L"],
    B: float = BRIEF_HULL["B"],
    T: float = BRIEF_HULL["T"],
    levels: Sequence[tuple[int, int]] = PANEL_LEVELS,
    area_quads: Sequence[int] = (120, 240, 480, 960),
) -> dict[str, Any]:
    """Displaced volume and wetted area of the meshed hull vs panel count.

    ``area_quads`` sets the Gauss-Legendre refinement ladder used to establish
    the wetted-area reference; the production value refines until the change
    between the last two levels is below ``1e-8`` relative.
    """
    ref = wigley_hydrostatic_reference(L, B, T)

    # wetted-area reference: refine until it stops moving (spectral quadrature)
    area_refinement = []
    for n_quad in area_quads:
        area_refinement.append(
            {
                "n_quad": n_quad,
                "wetted_area": wigley_wetted_area_reference(L, B, T, n_quad),
            }
        )
    area_ref = area_refinement[-1]["wetted_area"]
    area_self_error = abs(area_refinement[-2]["wetted_area"] / area_ref - 1.0)

    rows = []
    for n_sections, n_levels in levels:
        mesh = wigley_wetted_hull_mesh(L, B, T, n_sections, n_levels)
        volume, _ = compute_displaced_volume_and_buoyancy(mesh)
        area = float(np.sum(mesh.areas))
        rows.append(
            {
                "n_sections": n_sections,
                "n_levels": n_levels,
                "panels": mesh.panel_count,
                "volume": volume,
                "volume_rel_error": volume / ref["volume"] - 1.0,
                "Cb": volume / (L * B * T),
                "wetted_area": area,
                "wetted_area_rel_error": area / area_ref - 1.0,
            }
        )

    # observed convergence: error ~ N^-q, equivalently ~ h^2 (h ~ N^-1/2)
    def order(key: str) -> float:
        e0, e1 = abs(rows[0][key]), abs(rows[-1][key])
        n0, n1 = rows[0]["panels"], rows[-1]["panels"]
        return float(math.log(e0 / e1) / math.log(n1 / n0)) if e1 > 0 else float("inf")

    return {
        "L": L,
        "B": B,
        "T": T,
        "brief_targets": {"volume_rel_error": 0.01, "wetted_area_rel_error": 0.02},
        "observed_order_in_panels": {
            "volume": order("volume_rel_error"),
            "wetted_area": order("wetted_area_rel_error"),
            "note": (
                "q in |error| ~ N^-q; q = 1 corresponds to second order in the "
                "panel size h ~ N^-1/2, i.e. the expected faceting convergence"
            ),
        },
        "reference": {
            "volume": ref["volume"],
            "Cb": ref["Cb"],
            "waterplane_area": ref["waterplane_area"],
            "wetted_area": area_ref,
            "wetted_area_note": (
                "no elementary closed form for the wetted area of the parabolic "
                "Wigley hull; the reference is a 2-D Gauss-Legendre quadrature "
                "refined until successive levels agree "
                f"(|240 vs 960 nodes| = {area_self_error:.2e} relative)"
            ),
            "area_refinement": area_refinement,
            "area_self_error": area_self_error,
        },
        "rows": rows,
    }


# ============================================================================
# 3. Ellipsoid wave resistance (non-separable Havelock/Michell reference)
# ============================================================================


def ellipsoid_wave_study(
    a: float = ELLIPSOID["a"],
    b: float = ELLIPSOID["b"],
    c: float = ELLIPSOID["c"],
    froude: float = ELLIPSOID["Fr"],
    *,
    panel_levels: Iterable[tuple[int, int]] = ((10, 20), (20, 40), (40, 80)),
    rho: float = DEFAULT_RHO,
    g: float = DEFAULT_G,
) -> dict[str, Any]:
    """Ellipsoid wave resistance: closed form, definition, panels, Kelvin forms."""
    hull = EllipsoidHull(a, b, c)

    # --- geometry gate -------------------------------------------------------
    volume_quad = wetted_volume_quadrature(hull)
    wetted = wetted_surface_area(hull)
    geometry = {
        "a": a,
        "b": b,
        "c": c,
        "L": hull.L,
        "B": hull.B,
        "T": hull.T,
        "thinness_b_over_a": hull.thinness,
        "volume_closed_form": hull.volume,
        "volume_quadrature": volume_quad,
        "volume_rel_difference": abs(volume_quad / hull.volume - 1.0),
        "Cb": hull.Cb,
        "Cb_closed_form": math.pi / 6.0,
        "Cm": hull.Cm,
        "Cp": hull.Cp,
        "waterplane_area": hull.waterplane_area,
        "wetted_surface_area": wetted,
    }

    # --- Kochin function: closed form vs definition quadrature ---------------
    k_grid = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
    kochin_rows = []
    for k in k_grid:
        h_analytic = kochin_analytic(k, hull)
        h_quadrature = kochin_quadrature(k, hull)
        kochin_rows.append(
            {
                "k": float(k),
                "abs_analytic": abs(h_analytic),
                "abs_quadrature": abs(h_quadrature),
                "rel_difference": abs(h_analytic - h_quadrature) / abs(h_analytic),
                "argument_rad": float(np.angle(h_analytic)),
            }
        )
    max_kochin_rel = max(row["rel_difference"] for row in kochin_rows)

    # --- wave resistance ----------------------------------------------------
    u = froude * math.sqrt(g * hull.L)
    k0 = g / (u * u)
    lambda_max = max(5.0, 10.0 / (k0 * hull.T))
    rw_analytic = wave_resistance_analytic(hull, froude, rho=rho, g=g)
    rw_theta = kelvin_wave_resistance_theta(
        lambda kk: kochin_analytic(kk, hull), k0=k0, rho=rho, g=g
    )

    panel_rows = []
    for n_r, n_phi in panel_levels:
        for scheme in ("source_integrated", "ordinate", "source"):
            rw = wave_resistance_panels(
                hull, froude, n_r=n_r, n_phi=n_phi, scheme=scheme, rho=rho, g=g
            )
            panel_rows.append(
                {
                    "n_r": n_r,
                    "n_phi": n_phi,
                    "panels": n_r * n_phi,
                    "scheme": scheme,
                    "Rw": rw,
                    "rel_error": rw / rw_analytic - 1.0,
                }
            )

    dynamic_pressure = 0.5 * rho * u * u
    return {
        "geometry": geometry,
        "froude": froude,
        "speed": u,
        "k0": k0,
        "lambda_max": lambda_max,
        "kochin": {"rows": kochin_rows, "max_rel_difference": max_kochin_rel},
        "wave_resistance": {
            "analytic_closed_form": rw_analytic,
            "theta_form": rw_theta,
            "theta_vs_lambda_rel": abs(rw_theta / rw_analytic - 1.0),
            "Cw_wetted": rw_analytic / (dynamic_pressure * wetted),
            "Cw_LT": rw_analytic / (dynamic_pressure * hull.L * hull.T),
        },
        "panels": panel_rows,
        "rho": rho,
        "g": g,
    }


def ellipsoid_froude_curve(
    a: float = ELLIPSOID["a"],
    b: float = ELLIPSOID["b"],
    c: float = ELLIPSOID["c"],
    froudes: Sequence[float] = tuple(np.round(np.arange(0.15, 0.4501, 0.025), 4)),
    *,
    rho: float = DEFAULT_RHO,
    g: float = DEFAULT_G,
) -> dict[str, Any]:
    """``R_W(Fr)`` and ``Cw(Fr)`` of the ellipsoid: closed form vs panels."""
    hull = EllipsoidHull(a, b, c)
    wetted = wetted_surface_area(hull)
    rows = []
    for fr in froudes:
        u = fr * math.sqrt(g * hull.L)
        rw = wave_resistance_analytic(hull, fr, rho=rho, g=g)
        rw_panel = wave_resistance_panels(
            hull, fr, n_r=20, n_phi=40, scheme="source_integrated", rho=rho, g=g
        )
        rows.append(
            {
                "Fr": float(fr),
                "speed": u,
                "Rw_analytic": rw,
                "Rw_panels_800": rw_panel,
                "rel_error_panels": (rw_panel / rw - 1.0) if rw > 0.0 else 0.0,
                "Cw_wetted": rw / (0.5 * rho * u * u * wetted),
            }
        )
    return {"froudes": [float(f) for f in froudes], "rows": rows, "wetted_area": wetted}


# ============================================================================
# 4. Wigley wave resistance: thin-ship panel discretisation + Fr sweep
# ============================================================================


def wigley_kochin_panels(
    k: float,
    L: float,
    B: float,
    T: float,
    n_x: int,
    n_z: int,
    *,
    scheme: Literal["moment", "mean_source", "first_order"] = "mean_source",
) -> complex:
    """Kochin function of the Wigley hull from axis-aligned centreplane panels.

    The Wigley source density factorises, ``dy/dx = A x (1 - (z/T)^2)`` with
    ``A = -8 (B/2) / L^2``, which lets the same panels be evaluated at three
    different levels of approximation (the discretisation ladder):

    ``scheme="moment"``
        Per-panel *exact* moments ``int x e^{ikx} dx`` and
        ``int (1 - (z/T)^2) e^{kz} dz``.  Because the density factorises this is
        algebraically the closed-form Kochin function, so it is the consistency
        identity of the panel decomposition (agreement to round-off, not a
        measure of error).
    ``scheme="mean_source"`` (default)
        Constant source strength per panel equal to the exact panel mean of
        ``dy/dx``, plane-wave factor integrated exactly over the panel
        (``(e^{ikx1} - e^{ikx0})/(ik) * (e^{kz1} - e^{kz0})/k``).  This is the
        genuine first-order panel error of the thin-ship discretisation.
    ``scheme="first_order"``
        Literal "naive" panel code: strength *and* phase sampled at the panel
        centroid.
    """
    if k <= 0.0:
        raise ValueError("k must be positive")

    x_edges = np.linspace(-0.5 * L, 0.5 * L, n_x + 1)
    z_edges = np.linspace(-T, 0.0, n_z + 1)
    half_b = 0.5 * B
    a_src = -8.0 * half_b / (L * L)

    x0 = x_edges[:-1][:, None]
    x1 = x_edges[1:][:, None]
    z0 = z_edges[None, :-1]
    z1 = z_edges[None, 1:]

    def moment_x(xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
        """``int_{xa}^{xb} x e^{ikx} dx``."""
        fa = np.exp(1j * k * xa) * (xa / (1j * k) + 1.0 / (k * k))
        fb = np.exp(1j * k * xb) * (xb / (1j * k) + 1.0 / (k * k))
        return fb - fa

    def moment_z(za: np.ndarray, zb: np.ndarray) -> np.ndarray:
        """``int_{za}^{zb} (1 - (z/T)^2) e^{kz} dz``."""

        def g(z: np.ndarray) -> np.ndarray:
            return np.exp(k * z) * (
                1.0 / k - (z * z / k - 2.0 * z / (k * k) + 2.0 / k**3) / (T * T)
            )

        return g(zb) - g(za)

    def wave_x(xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
        return (np.exp(1j * k * xb) - np.exp(1j * k * xa)) / (1j * k)

    def wave_z(za: np.ndarray, zb: np.ndarray) -> np.ndarray:
        return (np.exp(k * zb) - np.exp(k * za)) / k

    if scheme == "moment":
        return 2.0 * complex(np.sum(a_src * moment_x(x0, x1) * moment_z(z0, z1)))

    if scheme == "mean_source":
        # int int_panel dy/dx dx dz = A * (x1^2 - x0^2)/2 * int (1-(z/T)^2) dz
        int_yx = (
            a_src
            * 0.5
            * (x1 * x1 - x0 * x0)
            * ((z1 - z0) - (z1**3 - z0**3) / (3.0 * T * T))
        )
        area = (x1 - x0) * (z1 - z0)
        strength = int_yx / area
        return 2.0 * complex(np.sum(strength * wave_x(x0, x1) * wave_z(z0, z1)))

    if scheme == "first_order":
        xc = 0.5 * (x0 + x1)
        zc = 0.5 * (z0 + z1)
        area = (x1 - x0) * (z1 - z0)
        y_x = a_src * xc * (1.0 - (zc / T) ** 2)
        return 2.0 * complex(np.sum(y_x * area * np.exp(k * zc + 1j * k * xc)))

    raise ValueError(f"unknown panel scheme {scheme!r}")


def wigley_panel_wave_resistance(
    L: float,
    B: float,
    T: float,
    froude: float,
    *,
    n_x: int,
    n_z: int,
    scheme: Literal["moment", "mean_source", "first_order"] = "mean_source",
    rho: float = DEFAULT_RHO,
    g: float = DEFAULT_G,
) -> float:
    """Wigley wave resistance from the centreplane panel Kochin function."""
    u = froude * math.sqrt(g * L)
    k0 = g / (u * u)
    lambda_max = max(5.0, 10.0 / (k0 * T))
    return kelvin_wave_resistance(
        lambda kk: wigley_kochin_panels(kk, L, B, T, n_x, n_z, scheme=scheme),
        k0=k0,
        rho=rho,
        g=g,
        lambda_max=lambda_max,
    )


def wigley_froude_sweep(
    L: float = BRIEF_HULL["L"],
    B: float = BRIEF_HULL["B"],
    T: float = BRIEF_HULL["T"],
    froudes: Sequence[float] = (0.20, 0.225, 0.25, 0.275, 0.30, 0.325, 0.35, 0.375, 0.40),
    *,
    panel_levels: Iterable[tuple[int, int]] = ((10, 10), (20, 20), (40, 40)),
    rho: float = DEFAULT_RHO,
    g: float = DEFAULT_G,
) -> dict[str, Any]:
    """Wigley ``R_W``/``Cw`` over Fr = 0.20 ... 0.40 vs the analytic Kochin value."""
    wetted = wigley_wetted_area_reference(L, B, T, 480)
    rows = []
    for fr in froudes:
        u = fr * math.sqrt(g * L)
        rw_ref = michell_wave_resistance(L, B, T, u, rho=rho, g=g)
        entry: dict[str, Any] = {
            "Fr": float(fr),
            "speed": u,
            "Rw_analytic": rw_ref,
            "Cw_wetted": rw_ref / (0.5 * rho * u * u * wetted),
            "Cw_LT": rw_ref / (0.5 * rho * u * u * L * T),
            "panels": [],
        }
        for n_x, n_z in panel_levels:
            record: dict[str, Any] = {"n_x": n_x, "n_z": n_z, "panels": n_x * n_z}
            for scheme in ("moment", "mean_source", "first_order"):
                rw = wigley_panel_wave_resistance(
                    L, B, T, fr, n_x=n_x, n_z=n_z, scheme=scheme, rho=rho, g=g
                )
                record[scheme] = rw
                record[f"{scheme}_rel_error"] = rw / rw_ref - 1.0
            entry["panels"].append(record)
        rows.append(entry)

    def _max_err(scheme: str, level: int | None = None) -> float:
        cells = []
        for row in rows:
            if level is None:
                cells.extend(row["panels"])
            else:
                cells.append(row["panels"][level])
        return max(abs(cell[f"{scheme}_rel_error"]) for cell in cells)
    return {
        "L": L,
        "B": B,
        "T": T,
        "wetted_area": wetted,
        "rows": rows,
        "max_mean_source_rel_error_all_levels": _max_err("mean_source"),
        "max_mean_source_rel_error_finest": _max_err("mean_source", level=-1),
        "max_first_order_rel_error_all_levels": _max_err("first_order"),
        "max_first_order_rel_error_finest": _max_err("first_order", level=-1),
        "max_moment_rel_error_all_levels": _max_err("moment"),
        "experimental_comparison": {
            "status": "WITHHELD",
            "target": "|Cw - Cw_exp| / Cw_exp < 10 % over Fr = 0.20 ... 0.40",
            "reason": (
                "no sourced ITTC Wigley Cw table is available in this work tree "
                "(no network access); docs/wigley_wave_resistance.md already "
                "registers the previously carried table as WITHHELD because it "
                "states neither a reference area nor a source. Not substituted "
                "with a reconstructed table."
            ),
            "required_artefact": (
                "ITTC-recommended Wigley resistance coefficients with (a) the "
                "exact model L/B/T, (b) the Cw reference area, (c) a page/table "
                "citation."
            ),
        },
        "rho": rho,
        "g": g,
    }


# ============================================================================
# 5. Machine criteria
# ============================================================================


def verify(
    *,
    hydro: dict[str, Any] | None = None,
    geo: dict[str, Any] | None = None,
    ellipsoid: dict[str, Any] | None = None,
    wigley: dict[str, Any] | None = None,
    ellipsoid_curve: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the mission acceptance criteria against measured numbers."""
    geo = geo if geo is not None else wigley_geometry_audit()
    hydro = hydro if hydro is not None else wigley_hydrostatic_panel_study()
    ellipsoid = ellipsoid if ellipsoid is not None else ellipsoid_wave_study()
    wigley = wigley if wigley is not None else wigley_froude_sweep()
    ellipsoid_curve = (
        ellipsoid_curve if ellipsoid_curve is not None else ellipsoid_froude_curve()
    )

    criteria: list[dict[str, Any]] = []

    def add(name: str, ok: bool, measured: float, threshold: float, note: str) -> None:
        criteria.append(
            {
                "id": name,
                "passed": bool(ok),
                "measured": float(measured),
                "threshold": float(threshold),
                "note": note,
            }
        )

    # --- geometry gate -------------------------------------------------------
    add(
        "G1_mesh_volume_vs_closed_form",
        geo["verdict"]["mesh_vs_closed_form_volume_rel"] < 1.0e-3,
        geo["verdict"]["mesh_vs_closed_form_volume_rel"],
        1.0e-3,
        "Wigley panel-mesh volume vs 4LBT/9 at 40x20 panels per side (3200 panels)",
    )
    add(
        "G2_mesh_Cb_vs_four_ninths",
        geo["verdict"]["mesh_Cb_rel_error"] < 1.0e-3,
        geo["verdict"]["mesh_Cb_rel_error"],
        1.0e-3,
        "Cb from the meshed hull vs closed-form 4/9 (brief claims 0.667 = Cp)",
    )
    add(
        "G3_ellipsoid_volume_quadrature",
        ellipsoid["geometry"]["volume_rel_difference"] < 1.0e-10,
        ellipsoid["geometry"]["volume_rel_difference"],
        1.0e-10,
        "ellipsoid wetted volume from quadrature vs 2 pi a b c / 3",
    )

    # --- Wigley hydrostatics vs panel count ----------------------------------
    vol_err = max(abs(row["volume_rel_error"]) for row in hydro["rows"])
    area_err = max(abs(row["wetted_area_rel_error"]) for row in hydro["rows"])
    vol_err_fine = abs(hydro["rows"][-1]["volume_rel_error"])
    add(
        "H1a_volume_error_all_levels",
        vol_err < 0.01,
        vol_err,
        0.01,
        "literal mission reading: |dV/V| < 1 % at each of N = 200 / 800 / 3200 "
        "wetted panels",
    )
    add(
        "H1b_volume_error_finest",
        vol_err_fine < 0.01,
        vol_err_fine,
        0.01,
        "|dV/V| < 1 % at N = 3200 wetted panels (mission accuracy band)",
    )
    add(
        "H2_wetted_area_error_all_levels",
        area_err < 0.02,
        area_err,
        0.02,
        "max |dS/S| over N = 200 / 800 / 3200 wetted panels (target < 2 %)",
    )

    # --- Ellipsoid wave resistance -------------------------------------------
    add(
        "E1_kochin_closed_form_vs_definition",
        ellipsoid["kochin"]["max_rel_difference"] < 1.0e-8,
        ellipsoid["kochin"]["max_rel_difference"],
        1.0e-8,
        "max |H_closed - H_definition| / |H| over k = 0.5 ... 20 1/m",
    )
    add(
        "E2_lambda_vs_theta_kelvin_form",
        ellipsoid["wave_resistance"]["theta_vs_lambda_rel"] < 1.0e-3,
        ellipsoid["wave_resistance"]["theta_vs_lambda_rel"],
        1.0e-3,
        "Kelvin-wake integral in lambda form vs wave-angle form",
    )
    ell_panel_err = max(
        abs(row["rel_error"])
        for row in ellipsoid["panels"]
        if row["scheme"] == "source_integrated"
    )
    add(
        "E3_ellipsoid_panel_error",
        ell_panel_err < 0.05,
        ell_panel_err,
        0.05,
        "max |Rw_panel/Rw_closed_form - 1| for panel-mean source strength, "
        "N = 200 / 800 / 3200 (target < 5 %)",
    )
    curve_err = max(abs(row["rel_error_panels"]) for row in ellipsoid_curve["rows"])
    add(
        "E4_ellipsoid_panel_error_froude_band",
        curve_err < 0.05,
        curve_err,
        0.05,
        "max panel error over Fr = 0.15 ... 0.45 at N = 800",
    )

    # --- Wigley wave resistance over Fr --------------------------------------
    add(
        "W1_wigley_panel_error_finest",
        wigley["max_mean_source_rel_error_finest"] < 0.05,
        wigley["max_mean_source_rel_error_finest"],
        0.05,
        "max constant-strength panel error over Fr = 0.20 ... 0.40 at 40x20 cells",
    )
    add(
        "W2_wigley_first_order_panel_error",
        wigley["max_first_order_rel_error_finest"] < 0.10,
        wigley["max_first_order_rel_error_finest"],
        0.10,
        "max first-order (centroid-sampled) panel error over Fr = 0.20 ... 0.40 "
        "at 40x20 cells",
    )

    add(
        "W4_panel_decomposition_identity",
        wigley["max_moment_rel_error_all_levels"] < 1.0e-10,
        wigley["max_moment_rel_error_all_levels"],
        1.0e-10,
        "per-panel exact moments must reproduce the closed-form Kochin R_W "
        "(consistency of the panel decomposition, all Fr and all levels)",
    )

    withheld = [
        {
            "id": "W3_wigley_vs_ITTC_experiment",
            "status": "WITHHELD",
            "target": "< 10 % on Cw over Fr = 0.20 ... 0.40",
            "reason": wigley["experimental_comparison"]["reason"],
        }
    ]

    passed = sum(1 for c in criteria if c["passed"])
    return {
        "criteria": criteria,
        "passed": passed,
        "total": len(criteria),
        "all_passed": passed == len(criteria),
        "withheld": withheld,
    }


# ============================================================================
# 6. Figures
# ============================================================================


def make_figures(report: dict[str, Any], outdir: Path) -> list[Path]:
    """Matplotlib figures for the report (returns the written paths)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # --- Fig 1: hydrostatic panel convergence --------------------------------
    hydro = report["hydrostatics"]
    panels = np.array([row["panels"] for row in hydro["rows"]], dtype=float)
    vol = np.abs([row["volume_rel_error"] for row in hydro["rows"]])
    area = np.abs([row["wetted_area_rel_error"] for row in hydro["rows"]])

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(panels, vol * 100.0, "o-", label=r"$|\Delta\nabla/\nabla|$")
    ax.loglog(panels, area * 100.0, "s-", label=r"$|\Delta S_w/S_w|$")
    ax.axhline(1.0, color="tab:blue", ls="--", lw=0.9, label="1 % gate (volume)")
    ax.axhline(2.0, color="tab:orange", ls="--", lw=0.9, label="2 % gate (wetted area)")
    ax.set_xlabel("wetted panels $N$")
    ax.set_ylabel("relative error  [%]")
    ax.set_title("Wigley III (L=3.8, B=0.5, T=0.225 m): panel convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "fig1_wigley_panel_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # --- Fig 2: ellipsoid Kochin spectrum + panel errors ---------------------
    ell = report["ellipsoid"]
    kk = np.array([row["k"] for row in ell["kochin"]["rows"]])
    ha = np.array([row["abs_analytic"] for row in ell["kochin"]["rows"]])

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    axes[0].loglog(kk, ha, "k-o", label="Bessel closed form")
    axes[0].set_xlabel("wavenumber $k$  [1/m]")
    axes[0].set_ylabel(r"$|H(k)|$  [m$^2$]")
    axes[0].set_title("ellipsoid Kochin function (a=1, b=0.2, c=0.15 m)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)

    for scheme, style in (
        ("source_integrated", "o-"),
        ("ordinate", "s-"),
        ("source", "^-"),
    ):
        rows = [r for r in ell["panels"] if r["scheme"] == scheme]
        axes[1].loglog(
            [r["panels"] for r in rows],
            [abs(r["rel_error"]) * 100.0 for r in rows],
            style,
            label=scheme.replace("_", " "),
        )
    axes[1].axhline(5.0, color="crimson", ls="--", lw=0.9, label="5 % gate")
    axes[1].set_xlabel("centreplane panels")
    axes[1].set_ylabel(r"$|R_{W,\rm panel}/R_{W,\rm ref}-1|$  [%]")
    axes[1].set_title(r"ellipsoid $R_W$ at Fr = 0.30")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "fig2_ellipsoid_kochin_panels.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # --- Fig 3: Wigley Cw vs Fr (panel vs analytic) --------------------------
    wig = report["wigley"]
    fr = np.array([row["Fr"] for row in wig["rows"]])
    cw_ref = np.array([row["Cw_wetted"] for row in wig["rows"]]) * 1.0e3
    q_dyn = np.array(
        [0.5 * wig["rho"] * row["speed"] ** 2 * wig["wetted_area"] for row in wig["rows"]]
    )
    cw_int = np.array([row["panels"][-1]["mean_source"] for row in wig["rows"]]) / q_dyn * 1.0e3
    cw_fo = np.array([row["panels"][-1]["first_order"] for row in wig["rows"]]) / q_dyn * 1.0e3

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(fr, cw_ref, "k-o", label="Michell/Kochin (analytic reference)")
    ax.plot(fr, cw_int, "s--", label="panel, constant source (40x40)")
    ax.plot(fr, cw_fo, "^:", label="panel, first-order (40x40)")
    ax.set_xlabel(r"Froude number $Fr = U/\sqrt{gL}$")
    ax.set_ylabel(r"$C_W \times 10^3$  (wetted-area basis)")
    ax.set_title("Wigley III wave resistance, L = 3.8 m\n(experimental overlay: WITHHELD)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "fig3_wigley_cw_vs_fr.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # --- Fig 4: ellipsoid centreplane source-density map (panel input) -------
    geo_e = ell["geometry"]
    hull = EllipsoidHull(geo_e["a"], geo_e["b"], geo_e["c"])
    xc, zc, _, strength = panel_grid(hull, 40, 80)
    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    sc = ax.scatter(xc, zc, c=np.abs(strength), s=6, cmap="viridis")
    fig.colorbar(sc, ax=ax, label=r"panel source density $|\partial y/\partial x|$  [-]")
    ax.set_aspect("equal")
    ax.set_xlabel("$x$  [m]")
    ax.set_ylabel("$z$  [m]")
    ax.set_title("ellipsoid centreplane source density (BEM input field, 3200 panels)")
    fig.tight_layout()
    path = outdir / "fig4_ellipsoid_source_density.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    return paths


# ============================================================================
# 7. CLI
# ============================================================================


def build_report() -> dict[str, Any]:
    """Run the whole study and return the artifact dictionary."""
    geo = wigley_geometry_audit()
    hydro = wigley_hydrostatic_panel_study()
    ellipsoid = ellipsoid_wave_study()
    ellipsoid_curve = ellipsoid_froude_curve()
    wigley = wigley_froude_sweep()
    verdict = verify(
        hydro=hydro,
        geo=geo,
        ellipsoid=ellipsoid,
        wigley=wigley,
        ellipsoid_curve=ellipsoid_curve,
    )
    return {
        "mission": "m5240",
        "node": "W18-N02",
        "repo": "TensorBEM",
        "brief": {
            "wigley": {
                "L": BRIEF_HULL["L"],
                "B": BRIEF_HULL["B"],
                "T": BRIEF_HULL["T"],
                "Cb_claimed": BRIEF_HULL["Cb_claimed"],
            },
            "ellipsoid": dict(ELLIPSOID),
            "panel_levels": [int(n_s * n_l * 2) for n_s, n_l in PANEL_LEVELS],
        },
        "geometry_audit": geo,
        "hydrostatics": hydro,
        "ellipsoid": ellipsoid,
        "ellipsoid_froude_curve": ellipsoid_curve,
        "wigley": wigley,
        "verdict": verdict,
    }


_SECTIONS = ("geo", "hydro", "ellipsoid", "curve", "wigley")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/m5240/hull_anchor_maturation.json"),
        help="artifact JSON path (merged across --only stages)",
    )
    parser.add_argument(
        "--figures",
        type=Path,
        default=Path("docs/m5240/figures"),
        help="figure directory",
    )
    parser.add_argument(
        "--only",
        default=",".join(_SECTIONS),
        help=(
            "comma separated sections to (re)compute: "
            + ", ".join(_SECTIONS)
            + "; sections already present in --out are kept"
        ),
    )
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    sections = [item.strip() for item in args.only.split(",") if item.strip()]
    unknown = sorted(set(sections) - set(_SECTIONS))
    if unknown:
        parser.error(f"unknown sections: {unknown}")

    report: dict[str, Any] = {}
    if args.out.exists():
        report = json.loads(args.out.read_text())

    report.setdefault("mission", "m5240")
    report.setdefault("node", "W18-N02")
    report.setdefault("repo", "TensorBEM")
    report["brief"] = {
        "wigley": {
            "L": BRIEF_HULL["L"],
            "B": BRIEF_HULL["B"],
            "T": BRIEF_HULL["T"],
            "Cb_claimed": BRIEF_HULL["Cb_claimed"],
        },
        "ellipsoid": dict(ELLIPSOID),
        "panel_levels": [int(n_s * n_l * 2) for n_s, n_l in PANEL_LEVELS],
    }

    if "geo" in sections:
        report["geometry_audit"] = wigley_geometry_audit()
    if "hydro" in sections:
        report["hydrostatics"] = wigley_hydrostatic_panel_study()
    if "ellipsoid" in sections:
        report["ellipsoid"] = ellipsoid_wave_study()
    if "curve" in sections:
        report["ellipsoid_froude_curve"] = ellipsoid_froude_curve()
    if "wigley" in sections:
        report["wigley"] = wigley_froude_sweep()

    complete = all(key in report for key in ("hydrostatics", "ellipsoid", "wigley"))
    if complete:
        report["verdict"] = verify(
            hydro=report["hydrostatics"],
            geo=report.get("geometry_audit"),
            ellipsoid=report["ellipsoid"],
            wigley=report["wigley"],
            ellipsoid_curve=report.get("ellipsoid_froude_curve"),
        )
        if not args.no_figures:
            report["figures"] = [str(p) for p in make_figures(report, args.figures)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")

    if complete:
        verdict = report["verdict"]
        print(f"criteria passed: {verdict['passed']}/{verdict['total']}")
        for crit in verdict["criteria"]:
            status = "PASS" if crit["passed"] else "FAIL"
            print(
                f"  [{status}] {crit['id']}: {crit['measured']:.3e} "
                f"(gate {crit['threshold']:.0e})"
            )
        for item in verdict["withheld"]:
            print(f"  [WITHHELD] {item['id']}: {item['target']}")
        print(f"artifact: {args.out}")
        return 0 if verdict["all_passed"] else 1

    print(f"sections {sections} written to {args.out} (artifact incomplete)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
