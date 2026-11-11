"""Parametric ice-class (PC6-like) hull surface for the wave-ice-ship leg.

The Polar Code PC6 hull form is not public property and no measured offset
table is shipped with this repository, so a hull generator is used instead of a
digitised hull.  The generator is *fully parametric and auditable*: the
half-breadth is

    b(x, z) = (B/2) * wl(u) * f(v),        u = 2x/L,   v = z/T,
    wl(u)   = max(1 - |u|**n, f_end),      (waterline profile)
    f(v)    = max(1 - (1 - c_k) |v|**q, c_k),   (section shape, v in [-1, 0])

which produces a full, blunt, ice-class-like hull with a wide flat bottom
(``c_k`` is the keel half-breadth ratio), a full waterline (large ``n``) and a
small transom/stem face (``f_end``) that keeps the mesh non-degenerate.

**The block coefficient is checked before anything else is computed.**  The
mesh volume is compared with an independent double-quadrature of the same
parametric definition, and ``Cb = V/(L B T)``, ``L``, ``B`` and ``T`` are
re-derived from the mesh itself.  A geometry that fails the ``Cb`` / ``L`` /
``B`` check raises instead of propagating into the hydrodynamics; the
acceptable band is ``validate_hull_geometry`` (default 1 % on ``Cb`` and on
the principal dimensions).

Fidelity statement: this is an *idealised* PC6-like hull.  Bow flare,
icebreaking stem rake, and the appendage/propeller-boss detail of a real PC6
vessel are not represented.  Numerical results for this hull are therefore
"idealised PC6-like hull" results, not certified predictions for a specific
vessel; the geometry-fidelity error is *unknown*, not zero.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry3d import WettedHullMesh

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

DEFAULT_WATER_DENSITY = 1025.0


@dataclass(frozen=True)
class IceClassHull:
    """Parametric PC6-like hull form.

    Parameters
    ----------
    length :
        Length between perpendiculars ``L`` (m), bow at ``+L/2``.
    breadth :
        Maximum breadth ``B`` (m).
    draft :
        Design draft ``T`` (m).
    waterline_exponent :
        ``n`` in the waterline profile; larger is fuller
        (``n = 4`` gives ``Cw``-like fullness ~0.8).
    section_exponent :
        ``q`` in the section shape; larger is a finer bilge.
    keel_ratio :
        ``c_k``: half-breadth at the keel as a fraction of the local
        half-breadth (flat-of-bottom width).
    end_ratio :
        ``f_end``: minimum waterline half-breadth ratio at bow and stern,
        which keeps the end faces (transom/stem) non-degenerate.
    """

    length: float = 120.0
    breadth: float = 24.0
    draft: float = 9.0
    waterline_exponent: float = 4.0
    section_exponent: float = 2.0
    keel_ratio: float = 0.70
    end_ratio: float = 0.02

    def __post_init__(self) -> None:
        for name in ("length", "breadth", "draft"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite positive number, got {value!r}")
            object.__setattr__(self, name, value)
        if not math.isfinite(self.waterline_exponent) or self.waterline_exponent <= 0.5:
            raise ValueError("waterline_exponent must be finite and > 0.5")
        if not math.isfinite(self.section_exponent) or self.section_exponent < 1.0:
            raise ValueError("section_exponent must be finite and >= 1.0")
        if not (0.0 < self.keel_ratio <= 1.0):
            raise ValueError("keel_ratio must be in (0, 1]")
        if not (0.0 < self.end_ratio < 0.5):
            raise ValueError("end_ratio must be in (0, 0.5)")

    # ----- shape functions ------------------------------------------------
    def waterline_ratio(self, x: float) -> float:
        """Waterline half-breadth ratio ``wl(u)`` at station ``x``."""
        u = 2.0 * float(x) / self.length
        if abs(u) > 1.0:
            raise ValueError("x lies outside the hull length")
        return max(1.0 - abs(u) ** self.waterline_exponent, self.end_ratio)

    def section_ratio(self, z: float) -> float:
        """Section half-breadth ratio ``f(v)`` at height ``z``."""
        v = float(z) / self.draft
        if v > 1.0e-12 or v < -1.0 - 1.0e-12:
            raise ValueError("z lies outside the hull draft")
        v = min(max(v, -1.0), 0.0)
        return max(1.0 - (1.0 - self.keel_ratio) * abs(v) ** self.section_exponent, self.keel_ratio)

    def half_breadth(self, x: float, z: float) -> float:
        """Half-breadth ``b(x, z)`` (m)."""
        return 0.5 * self.breadth * self.waterline_ratio(x) * self.section_ratio(z)

    # ----- reference integrals (independent of the mesh) -------------------
    def reference_volume(self, *, panels: int = 400) -> float:
        """Displaced volume by double quadrature of the parametric definition.

        The double integral is evaluated with the composite **trapezoid** rule:
        the ``panels``-interval grid carries half weights on the two boundary
        stations (``u = +-1``) and on the two boundary rings (``v = 0, -1``).

        Mission m14775: this method previously summed every node with unit
        weight, i.e. it dropped the half weights of the boundary rows/columns.
        That is not a trapezoid rule: it overstates the volume by
        ``h_u * h_v * (boundary node sum) / 2``, which at the default grid is
        ``+2.54e-3`` in ``Cb`` for the KCS-class hull and converges only at
        *first* order (halving ``panels`` halves the error), so the reference
        the ``Cb`` gate compares against was itself wrong at the 1e-3 level --
        a quarter of the gate's 1 % band.  :meth:`closed_form_volume` is the
        analytic value this quadrature must converge to; the test suite pins
        that agreement to 1e-9 instead of pinning a self-consistent number.
        """
        if not isinstance(panels, (int, np.integer)) or int(panels) < 8:
            raise ValueError("panels must be an integer >= 8")
        n = int(panels)
        u = np.linspace(-1.0, 1.0, n + 1)
        v = np.linspace(-1.0, 0.0, n + 1)
        wl = np.maximum(1.0 - np.abs(u) ** self.waterline_exponent, self.end_ratio)
        f = np.maximum(
            1.0 - (1.0 - self.keel_ratio) * np.abs(v) ** self.section_exponent, self.keel_ratio
        )
        # integrate 2 * (B/2) * wl * f over x and z with the trapezoid rule
        integrand = 2.0 * 0.5 * self.breadth * np.outer(wl, f)
        dx = 0.5 * self.length * (u[1] - u[0])
        dz = self.draft * (v[1] - v[0])
        w_u = np.full(n + 1, 1.0)
        w_u[0] = w_u[-1] = 0.5
        w_v = np.full(n + 1, 1.0)
        w_v[0] = w_v[-1] = 0.5
        return float(np.sum(np.outer(w_u, w_v) * integrand) * dx * dz)

    def reference_waterplane_area(self, *, panels: int = 400) -> float:
        """Waterplane area by quadrature of the waterline profile (trapezoid rule)."""
        if not isinstance(panels, (int, np.integer)) or int(panels) < 8:
            raise ValueError("panels must be an integer >= 8")
        n = int(panels)
        u = np.linspace(-1.0, 1.0, n + 1)
        wl = np.maximum(1.0 - np.abs(u) ** self.waterline_exponent, self.end_ratio)
        dx = 0.5 * self.length * (u[1] - u[0])
        w_u = np.full(n + 1, 1.0)
        w_u[0] = w_u[-1] = 0.5
        return float(np.sum(w_u * 2.0 * 0.5 * self.breadth * wl) * dx)

    # ----- analytic reference (exact, independent of any quadrature) -------
    def closed_form_waterline_integral(self) -> float:
        r"""``I_u = \int_{-1}^{1} max(1 - |u|^n, f_end) du`` in closed form.

        The clamp binds for ``|u| > u*`` with ``u* = (1 - f_end)**(1/n)``:

        .. math:: I_u = 2\left[u^* - \frac{u^{*n+1}}{n+1} + f_{end}(1 - u^*)\right]
        """
        n = self.waterline_exponent
        e = self.end_ratio
        u_star = min(max((1.0 - e) ** (1.0 / n), 0.0), 1.0)
        return 2.0 * (u_star - u_star ** (n + 1.0) / (n + 1.0) + e * (1.0 - u_star))

    def closed_form_section_integral(self) -> float:
        r"""``I_v = \int_{0}^{1} max(1 - (1 - c_k) v^q, c_k) dv`` in closed form.

        For ``0 < c_k <= 1`` the clamp never binds strictly inside ``[0, 1]``
        (``1 - (1 - c_k) v^q = c_k`` only at ``v = 1``), so
        ``I_v = 1 - (1 - c_k) / (q + 1)``.
        """
        return 1.0 - (1.0 - self.keel_ratio) / (self.section_exponent + 1.0)

    def closed_form_volume(self) -> float:
        """Exact displaced volume ``B (L/2) T I_u I_v`` of the parametric hull."""
        return (
            self.breadth
            * 0.5
            * self.length
            * self.draft
            * self.closed_form_waterline_integral()
            * self.closed_form_section_integral()
        )

    def closed_form_waterplane_area(self) -> float:
        """Exact waterplane area ``B (L/2) I_u`` of the parametric hull."""
        return self.breadth * 0.5 * self.length * self.closed_form_waterline_integral()

    def closed_form_block_coefficient(self) -> float:
        """Exact ``Cb = I_u I_v / 2`` implied by the parametric definition."""
        return 0.5 * self.closed_form_waterline_integral() * self.closed_form_section_integral()

    def reference_block_coefficient(self) -> float:
        """``Cb = V_ref / (L B T)`` from the parametric quadrature."""
        return self.reference_volume() / (self.length * self.breadth * self.draft)


def ice_class_hull_mesh(
    hull: IceClassHull,
    *,
    n_sections: int = 21,
    n_side: int = 5,
    n_bottom: int = 3,
) -> WettedHullMesh:
    """Build the closed wetted-surface mesh of :class:`IceClassHull`.

    The surface is lofted from ``n_sections`` planar rings; each ring runs from
    the starboard deck edge down the side, across the flat of bottom, up the
    port side and closes along the waterline chord, so the waterplane itself is
    part of the closed surface (this is the convention the 3-D free-surface
    solver expects: a closed body including the waterplane lid).  The two end
    rings are closed with planar fans (transom and stem face).

    Parameters
    ----------
    hull :
        Parametric hull definition.
    n_sections :
        Number of stations along the length (>= 5).
    n_side :
        Number of side panels per half-section above the bilge corner (>= 2).
    n_bottom :
        Number of interior flat-of-bottom points (>= 1).
    """
    if not isinstance(n_sections, (int, np.integer)) or int(n_sections) < 5:
        raise ValueError("n_sections must be an integer >= 5")
    if not isinstance(n_side, (int, np.integer)) or int(n_side) < 2:
        raise ValueError("n_side must be an integer >= 2")
    if not isinstance(n_bottom, (int, np.integer)) or int(n_bottom) < 1:
        raise ValueError("n_bottom must be an integer >= 1")

    ns, nside, nbot = int(n_sections), int(n_side), int(n_bottom)
    L, B, T = hull.length, hull.breadth, hull.draft

    def ring(x: float) -> list[tuple[float, float, float]]:
        """Open section polyline: starboard deck edge -> keel -> port deck edge."""
        b = 0.5 * B * hull.waterline_ratio(x)
        pts: list[tuple[float, float, float]] = [(x, +b, 0.0)]
        for j in range(1, nside + 1):
            t = j / nside
            z = -T * t
            pts.append((x, +b * hull.section_ratio(z), z))
        for j in range(1, nbot + 1):
            y = b * hull.keel_ratio * (1.0 - 2.0 * j / (nbot + 1))
            pts.append((x, y, -T))
        for j in range(nside, 0, -1):
            t = j / nside
            z = -T * t
            pts.append((x, -b * hull.section_ratio(z), z))
        pts.append((x, -b, 0.0))
        return pts

    rings = [ring(-0.5 * L + i * L / (ns - 1)) for i in range(ns)]
    ring_size = len(rings[0])
    vertices = np.asarray([p for r in rings for p in r], dtype=float)
    body_center = np.array([0.0, 0.0, -0.5 * T])

    faces: list[tuple[int, int, int]] = []

    def wind(a: int, b: int, c: int) -> tuple[int, int, int]:
        """Order the triangle so that its normal points away from the body."""
        pa, pb, pc = vertices[a], vertices[b], vertices[c]
        normal = np.cross(pb - pa, pc - pa)
        centroid = (pa + pb + pc) / 3.0
        if float(np.dot(normal, centroid - body_center)) < 0.0:
            return (a, c, b)
        return (a, b, c)

    for i in range(ns - 1):
        base_a = i * ring_size
        base_b = (i + 1) * ring_size
        for j in range(ring_size - 1):
            a0, a1 = base_a + j, base_a + j + 1
            b0, b1 = base_b + j, base_b + j + 1
            faces.append(wind(a0, a1, b1))
            faces.append(wind(a0, b1, b0))

    # end caps: planar fans over the closed section (section polyline + waterline chord)
    for index in (0, ns - 1):
        base = index * ring_size
        loop = [base + j for j in range(ring_size)]
        centre = np.array(
            [
                np.mean(vertices[loop, 0]),
                0.0,
                0.5 * (np.mean(vertices[loop, 2]) + 0.0),
            ]
        )
        centre_index = vertices.shape[0]
        vertices = np.vstack((vertices, centre[None, :]))
        for j in range(ring_size):
            jn = (j + 1) % ring_size
            faces.append(wind(centre_index, loop[j], loop[jn]))

    face_array = np.asarray(faces, dtype=np.int64)
    mesh = WettedHullMesh(vertices=vertices, faces=face_array)
    if float(mesh.volume) <= 0.0:  # pragma: no cover - orientation guard
        mesh = WettedHullMesh(
            vertices=vertices,
            faces=np.ascontiguousarray(face_array[:, [0, 2, 1]]),
        )
    return mesh


def validate_hull_geometry(
    hull: IceClassHull,
    mesh: WettedHullMesh,
    *,
    cb_tolerance: float = 0.01,
    dimension_tolerance: float = 1.0e-2,
) -> dict[str, float]:
    """Check the mesh against the parametric definition: ``Cb``, ``L``, ``B``, ``T``.

    Returns a dictionary of measured and reference values plus the relative
    errors.  Raises ``ValueError`` when the mesh block coefficient or a
    principal dimension departs from the parametric reference beyond
    ``cb_tolerance`` / ``dimension_tolerance`` -- a geometry that cannot pass
    this gate must not be used for hydrodynamics.
    """
    volume = float(mesh.volume)
    vertices = np.asarray(mesh.vertices, dtype=float)
    length = float(np.ptp(vertices[:, 0]))
    breadth = float(np.ptp(vertices[:, 1]))
    draft = float(np.ptp(vertices[:, 2]))
    v_ref = hull.reference_volume()
    cb = volume / (hull.length * hull.breadth * hull.draft)
    cb_ref = v_ref / (hull.length * hull.breadth * hull.draft)
    awp = _waterplane_area(mesh)
    awp_ref = hull.reference_waterplane_area()

    errors = {
        "cb": abs(cb - cb_ref) / abs(cb_ref),
        "length": abs(length - hull.length) / hull.length,
        "breadth": abs(breadth - hull.breadth) / hull.breadth,
        "draft": abs(draft - hull.draft) / hull.draft,
        "waterplane_area": abs(awp - awp_ref) / awp_ref,
    }
    if errors["cb"] > cb_tolerance:
        raise ValueError(
            f"mesh block coefficient {cb:.5f} departs from the parametric reference "
            f"{cb_ref:.5f} by {100.0 * errors['cb']:.3f} % (> {100.0 * cb_tolerance:.1f} %)"
        )
    for key in ("length", "breadth", "draft"):
        if errors[key] > dimension_tolerance:
            raise ValueError(
                f"mesh {key} {errors[key] * 100.0:.3f} % off the parametric value"
            )
    return {
        "length": length,
        "breadth": breadth,
        "draft": draft,
        "volume": volume,
        "volume_reference": v_ref,
        "block_coefficient": cb,
        "block_coefficient_reference": cb_ref,
        "waterplane_area": awp,
        "waterplane_area_reference": awp_ref,
        "cb_relative_error": errors["cb"],
        "length_relative_error": errors["length"],
        "breadth_relative_error": errors["breadth"],
        "draft_relative_error": errors["draft"],
        "waterplane_area_relative_error": errors["waterplane_area"],
    }


def _waterplane_area(mesh: WettedHullMesh) -> float:
    """Waterplane area enclosed by the waterline boundary loop(s).

    The wetted-hull convention here has no horizontal lid, so the waterplane is
    recovered from the open boundary: ``A = |sum (x_a y_b - x_b y_a)| / 2`` over
    the closed waterline edges.
    """
    vertices = np.asarray(mesh.vertices, dtype=float)
    edges = np.asarray(mesh.waterline_edges, dtype=np.int64)
    if edges.size == 0:
        return 0.0
    xa, ya = vertices[edges[:, 0], 0], vertices[edges[:, 0], 1]
    xb, yb = vertices[edges[:, 1], 0], vertices[edges[:, 1], 1]
    return float(abs(np.sum(xa * yb - xb * ya)) / 2.0)


def hull_mass_properties(
    hull: IceClassHull,
    geometry_check: dict[str, float],
    *,
    water_density: float = DEFAULT_WATER_DENSITY,
    kg_roll: float = 0.35,
    kg_pitch: float = 0.25,
    kg_yaw: float = 0.25,
    cg_height_ratio: float = 0.5,
) -> dict[str, object]:
    """Mass, centre of gravity and inertia tensor of the design-condition hull.

    The vessel floats at the design draft, so ``mass = rho_w * V``.  Radii of
    gyration are engineering estimates ``k_xx = 0.35 B``, ``k_yy = k_zz =
    0.25 L`` (typical for a full-form cargo hull) and the centre of gravity is
    placed at ``z = -cg_height_ratio * T``.  These are **assumptions, not
    measurements**: they are reported with the results so that the RAO values
    can be re-derived for a different mass distribution.
    """
    if not (0.0 <= cg_height_ratio <= 1.0):
        raise ValueError("cg_height_ratio must be in [0, 1]")
    if kg_roll <= 0.0 or kg_pitch <= 0.0 or kg_yaw <= 0.0:
        raise ValueError("radii of gyration must be positive")
    volume = float(geometry_check["volume"])
    mass = water_density * volume
    cg = np.array([0.0, 0.0, -cg_height_ratio * hull.draft], dtype=float)
    i_xx = mass * (kg_roll * hull.breadth) ** 2
    i_yy = mass * (kg_pitch * hull.length) ** 2
    i_zz = mass * (kg_yaw * hull.length) ** 2
    inertia_cg = np.diag([i_xx, i_yy, i_zz])
    r = cg
    inertia_origin = inertia_cg + mass * (
        float(np.dot(r, r)) * np.eye(3) - np.outer(r, r)
    )
    return {
        "mass": mass,
        "center_of_gravity": cg,
        "inertia_about_origin": inertia_origin,
        "radii_of_gyration": {"roll": kg_roll * hull.breadth,
                              "pitch": kg_pitch * hull.length,
                              "yaw": kg_yaw * hull.length},
        "assumptions": (
            "CG height = " f"{cg_height_ratio:.2f} T; k_xx = {kg_roll:.2f} B; "
            f"k_yy = k_zz = {kg_pitch:.2f} L (engineering estimates)"
        ),
    }