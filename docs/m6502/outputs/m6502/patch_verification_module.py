"""Patch the tank free-surface verification module: add the energy-Hessian check (C8).

Kept as an audit trail of exactly which strings were changed (mission m6502).
"""
from pathlib import Path

p = Path("/root/tp-agent/work/m6502/src/tensorbem/benchmarks/tank_free_surface_verification.py")
s = p.read_text()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new)


# ---- 1. module docstring: the energy path is now closed ---------------------
rep(
    """A fourth path -- differentiating the liquid *potential energy* on a graded grid
to confirm the signs from energy alone -- was implemented and **did not close**:
the grid helper returned a curvature four times the analytic value on the
rectangle, so the numbers were discarded rather than published.  The sign of
every entry therefore rests on the second-order derivation in the module
docstring plus the hand-checked triangle/L-shape moments; the gap is recorded in
``docs/tank_free_surface_correction.md`` §5.
""",
    """5. **Liquid potential energy** (:func:`_energy_hessian`, ``checks.C8``) -- the
   potential energy of the slack tank minus that of the *same* liquid frozen to
   the hull, differentiated twice by central differences over a footprint grid.
   Subtracting the frozen reference is essential: it cancels the rigid
   weight-at-height bookkeeping (the ``(1/2) H^2 A`` term of the surface
   integral and the ``cos`` factor), and what survives is the free-surface
   correction alone.  An earlier attempt (mission m6452) kept those terms and
   had no reference state, so it returned a curvature four times the analytic
   value; the path was abandoned then and is closed now.  The *signs* of
   ``dC[3,3]``, ``dC[4,4]`` and ``dC[3,4]`` therefore rest on energy as well as
   on the derivation, at the price of the same first-order free-surface plane
   ``h = H - phi y + theta x`` the module itself assumes.
""",
)

# ---- 2. helpers before run() ------------------------------------------------
helpers = '''
def _inside_polygon(x, y, verts) -> "np.ndarray":
    """Vectorised ray-crossing point-in-polygon test (any simple polygon)."""
    inside = np.zeros(np.shape(x), dtype=bool)
    n = verts.shape[0]
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        if y1 == y2:  # horizontal edge cannot be crossed by the ray
            continue
        straddles = (y1 > y) != (y2 > y)
        x_at = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= straddles & (x < x_at)
    return inside


def _grid_polygon(n: int, verts) -> tuple:
    """Midpoint cells inside ``verts``: ``(area, x, y)`` of the retained cells."""
    v = np.asarray(verts, dtype=float)
    lo, hi = v.min(axis=0), v.max(axis=0)
    hx, hy = (hi[0] - lo[0]) / n, (hi[1] - lo[1]) / n
    xc = lo[0] + (np.arange(n) + 0.5) * hx
    yc = lo[1] + (np.arange(n) + 0.5) * hy
    xx, yy = np.meshgrid(xc, yc, indexing="xy")
    mask = _inside_polygon(xx, yy, v)
    return (hx * hy * mask)[mask], xx[mask], yy[mask]


def _liquid_energies(verts, n, depth, phi, theta, density, g, z_bottom=0.0) -> dict:
    """Liquid potential energy: free surface vs liquid frozen to the hull.

    Body-to-inertial vertical mapping ``z_I = z cos(phi) cos(theta) + y sin(phi)
    - x sin(theta)`` (roll about x, pitch about y, the convention of
    ``tensorbem.hydrostatic``, ``w = y phi - x theta``).  The free surface in
    body coordinates is the plane ``h = H - phi y + theta x``; the footprint is
    centroidal, so the linear terms integrate to zero and ``H = depth`` keeps the
    volume ``A H``.  With ``z = z_bottom + zeta`` and ``int zeta dzeta = h^2/2``,

        U = rho g int_A [ h (z_bottom cos + y sin phi - x sin theta)
                          + (1/2) h^2 cos phi cos theta ] dA.

    ``free`` uses ``h``, ``frozen`` uses the constant ``depth`` (the same liquid
    carried rigidly by the hull), ``delta`` is their difference and
    ``delta_direct`` the same difference formed inside the sum (no cancellation).
    """
    w, x, y = _grid_polygon(n, verts)
    area = float(w.sum())
    cos = math.cos(phi) * math.cos(theta)
    zb = z_bottom * cos + y * math.sin(phi) - x * math.sin(theta)
    h = depth - phi * y + theta * x
    k = density * g
    u_free = k * float((w * h * zb).sum() + (w * (0.5 * h * h) * cos).sum())
    u_frozen = k * float((w * depth * zb).sum() + (w * (0.5 * depth * depth) * cos).sum())
    delta_direct = k * float(
        (w * (h - depth) * zb).sum() + (w * 0.5 * (h * h - depth * depth) * cos).sum()
    )
    return {
        "area": area,
        "free": u_free,
        "frozen": u_frozen,
        "delta": u_free - u_frozen,
        "delta_direct": delta_direct,
        "volume_relerr": abs(float((w * h).sum()) - area * depth) / (area * depth),
    }


def _energy_hessian(verts, n, depth, eps, density, g) -> dict:
    """Central-difference Hessian of ``U_free - U_frozen`` in ``(phi, theta)``."""

    def delta(phi, theta):
        return _liquid_energies(verts, n, depth, phi, theta, density, g)["delta"]

    zero = delta(0.0, 0.0)
    roll = (delta(eps, 0.0) - 2.0 * zero + delta(-eps, 0.0)) / eps ** 2
    pitch = (delta(0.0, eps) - 2.0 * zero + delta(0.0, -eps)) / eps ** 2
    cross = (
        delta(eps, eps) - delta(eps, -eps) - delta(-eps, eps) + delta(-eps, -eps)
    ) / (4.0 * eps ** 2)
    probe = _liquid_energies(verts, n, depth, eps, eps, density, g)
    return {
        "roll": roll,
        "pitch": pitch,
        "roll_pitch": cross,
        "rest": zero,
        "cancellation_relerr": abs(probe["delta"] - probe["delta_direct"])
        / max(abs(probe["delta_direct"]), 1.0),
        "volume_relerr": probe["volume_relerr"],
    }


'''
rep("\ndef run(quick: bool = False) -> dict:", helpers + "\ndef run(quick: bool = False) -> dict:")

# ---- 3. the C8 check --------------------------------------------------------
c8 = '''    # ---- C8: independent energy Hessian (free surface minus frozen liquid) ---
    # Analytic targets are the entries the module claims:
    #   dC[3,3] = -rho g i_xx,  dC[4,4] = -rho g i_yy,  dC[3,4] = +rho g i_xy.
    # Two FD steps are reported so the O(eps^2) truncation of the sin(phi) ~ phi
    # linearisation is visible (that is the only error source: the integrand is
    # exactly quadratic in phi, theta).
    depth = 3.0
    steps = (1.0e-2, 1.0e-3)
    c8: dict = {"depth": depth, "fd_steps": list(steps), "cases": {}}
    for label, verts, moments in (("rectangle", RECT, rect), ("Lshape", L_SHAPE, l_m)):
        analytic = {
            "roll": -rho_w * g * moments.i_xx,
            "pitch": -rho_w * g * moments.i_yy,
            "roll_pitch": rho_w * g * moments.i_xy,
        }
        rows = []
        for eps in steps:
            h = _energy_hessian(verts, grids[-1], depth, eps, rho_w, g)
            h_coarse = _energy_hessian(verts, grids[0], depth, eps, rho_w, g)
            rows.append(
                {
                    "eps": eps,
                    "roll": h["roll"],
                    "pitch": h["pitch"],
                    "roll_pitch": h["roll_pitch"],
                    "relerr": {
                        k: abs(h[k] - analytic[k]) / abs(analytic[k])
                        for k in analytic
                        if analytic[k] != 0.0
                    },
                    "cross_abs_err": abs(h["roll_pitch"] - analytic["roll_pitch"]),
                    "grid_independence_relerr": max(
                        abs(h[k] - h_coarse[k]) / max(abs(analytic[k]), 1.0) for k in analytic
                    ),
                    "cancellation_relerr": h["cancellation_relerr"],
                    "volume_relerr": h["volume_relerr"],
                }
            )
        c8["cases"][label] = {
            "grid": grids[-1],
            "coarse_grid": grids[0],
            "analytic": analytic,
            "rows": rows,
            "worst_rel_error": max(max(r["relerr"].values()) for r in rows),
            "cross_abs_err": max(r["cross_abs_err"] for r in rows),
            "max_cancellation_relerr": max(r["cancellation_relerr"] for r in rows),
            "max_volume_relerr": max(r["volume_relerr"] for r in rows),
            "max_grid_independence_relerr": max(
                r["grid_independence_relerr"] for r in rows
            ),
        }
    out["checks"]["C8_energy_hessian"] = c8

'''
rep(
    "    # ---- demo: applying it to a hydrostatic restoring matrix -----------------",
    c8 + "    # ---- demo: applying it to a hydrostatic restoring matrix -----------------",
)

# ---- 4. not_run: the energy gap is closed, record what is genuinely open ----
rep(
    '''    out["not_run"] = {
        "energy_hessian": (
            "graded-grid differentiation of the liquid potential energy was implemented but its "
            "helper returned a curvature 4x the analytic value on the rectangle; discarded, not "
            "published (see report section 5)"
        )
    }''',
    '''    out["not_run"] = {
        "large_angle_shape_stability": (
            "GZ curves and top-out/empty-out of the free surface are outside the linearised "
            "correction; no large-angle energy check is run here"
        ),
        "sloshing_and_baffles": (
            "ideal liquid only: no sloshing dynamics, no baffle damping; a baffled tank must be "
            "entered as separate free-surface polygons by the caller"
        ),
        "experimental_validation": (
            "no model-test or full-scale data available in-repo; every check here is code "
            "verification against closed-form references"
        ),
    }''',
)

p.write_text(s)
print("patched", p)