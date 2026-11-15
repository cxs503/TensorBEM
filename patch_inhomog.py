"""One-shot patch: inhomogeneous (field) density support on the mesh mass route."""
from pathlib import Path

ROOT = Path("/srv/tensor/agent/work/m21422")
p = ROOT / "src/tensorbem/rigid_body_mass.py"
src = p.read_text()

# ---------------------------------------------------------------- constants ---
old_exports = '''__all__ = [
    "MODES",
'''
new_exports = '''__all__ = [
    "MODES",
'''
assert old_exports in src
src = src.replace(old_exports, new_exports, 1)

anchor = '''#: Water density used when a caller does not supply one (ITTC sea water).
DEFAULT_DENSITY = 1025.0
'''
assert anchor in src
helper = anchor + '''

def _barycentric_triple_moments() -> FloatArray:
    """``int lambda_k lambda_m lambda_n dV = 6 V (prod multiplicities!) / 6!``.

    ``lambda_k`` are the barycentric coordinates of a tetrahedron, so this
    tensor of coefficients (divided by ``V``) integrates every cubic monomial in
    closed form.  Values: ``V/20`` (all three indices equal), ``V/60`` (two
    equal, one distinct), ``V/120`` (all distinct).
    """
    coefficients = np.empty((4, 4, 4), dtype=float)
    factorial = [math.factorial(i) for i in range(7)]
    for k in range(4):
        for m in range(4):
            for n in range(4):
                counts: dict[int, int] = {}
                for index in (k, m, n):
                    counts[index] = counts.get(index, 0) + 1
                product = 1
                for multiplicity in counts.values():
                    product *= factorial[multiplicity]
                coefficients[k, m, n] = 6.0 * product / factorial[6]
    return coefficients


#: ``int lambda_k lambda_m dV = V/20 (1 + delta_km)``.
_BARYCENTRIC_PAIR_MOMENTS = (np.ones(4, dtype=float) + np.eye(4)) / 20.0
#: ``int lambda_k lambda_m lambda_n dV`` (per unit volume).
_BARYCENTRIC_TRIPLE_MOMENTS = _barycentric_triple_moments()
'''
src = src.replace(anchor, helper, 1)

# ---------------------------------------------------- mesh_mass_properties ---
old_sig = '''def mesh_mass_properties(
    vertices: ArrayLike,
    faces: ArrayLike,
    density: float = DEFAULT_DENSITY,
    *,
    reference_point: ArrayLike = (0.0, 0.0, 0.0),
) -> dict[str, object]:'''
new_sig = '''def mesh_mass_properties(
    vertices: ArrayLike,
    faces: ArrayLike,
    density: float = DEFAULT_DENSITY,
    *,
    reference_point: ArrayLike = (0.0, 0.0, 0.0),
    density_field: "Callable[[FloatArray], ArrayLike] | None" = None,
) -> dict[str, object]:'''
assert old_sig in src
src = src.replace(old_sig, new_sig, 1)

old_doc = '''    reference_point:
        Point about which ``inertia_reference`` is taken (default: the
        coordinate origin).  ``inertia_cg`` and ``centroid`` are referred to the
        same point: translating mesh *and* ``reference_point`` by the same
        vector leaves both unchanged (tested in
        ``tests/test_rigid_body_mass.py``).
'''
new_doc = '''    reference_point:
        Point about which ``inertia_reference`` is taken (default: the
        coordinate origin).  ``inertia_cg`` and ``centroid`` are referred to the
        same point: translating mesh *and* ``reference_point`` by the same
        vector leaves both unchanged (tested in
        ``tests/test_rigid_body_mass.py``).
    density_field:
        Optional *dimensionless* shape factor ``f(x) >= 0`` applied as
        ``rho(x) = density * f(x)`` -- the inhomogeneous counterpart of the
        uniform ``density``.  ``f`` is called once with the ``(4 F, 3)`` corner
        coordinates of every signed tetrahedron (including the apex, i.e. the
        vertex mean of the mesh), so a *linearly varying* density is integrated
        **exactly**: the tetrahedral decomposition interpolates ``f`` linearly
        from its four corners and every resulting cubic monomial has a closed
        form here (:data:`_BARYCENTRIC_TRIPLE_MOMENTS`).  A quadratic or steeper
        field is integrated with second-order accuracy in the mesh size, exactly
        like the geometry itself.  ``None`` (default) is the homogeneous body and
        keeps the pre-existing arithmetic unchanged.
'''
assert old_doc in src
src = src.replace(old_doc, new_doc, 1)

old_body = '''    vertex_sum = rel_a + rel_b + rel_c  # the apex contributes zero
    first_moment = np.einsum("i,ij->j", tet_vol, vertex_sum) / 4.0
    centroid_apex = first_moment / volume

    # int xi xj dV per tetra, then summed, in coordinates relative to the apex.
    sum_sq = np.einsum("fi,fj->fij", rel_a, rel_a)
    sum_sq += np.einsum("fi,fj->fij", rel_b, rel_b)
    sum_sq += np.einsum("fi,fj->fij", rel_c, rel_c)
    sum_first = np.einsum("fi,fj->fij", vertex_sum, vertex_sum)
    # Second moments of volume, scaled by density so every tensor below is in
    # SI mass units (kg m^2) rather than geometric units (m^5).
    second = rho * np.einsum("f,fij->ij", tet_vol, sum_sq + sum_first) / 20.0

    second = 0.5 * (second + second.T)
    inertia_apex = np.trace(second) * np.eye(3) - second

    mass = rho * volume
    centroid = p0 + apex + centroid_apex
'''
new_body = '''    if density_field is None:
        vertex_sum = rel_a + rel_b + rel_c  # the apex contributes zero
        first_moment = np.einsum("i,ij->j", tet_vol, vertex_sum) / 4.0
        # int xi xj dV per tetra, then summed, in coordinates relative to the apex.
        sum_sq = np.einsum("fi,fj->fij", rel_a, rel_a)
        sum_sq += np.einsum("fi,fj->fij", rel_b, rel_b)
        sum_sq += np.einsum("fi,fj->fij", rel_c, rel_c)
        sum_first = np.einsum("fi,fj->fij", vertex_sum, vertex_sum)
        # Second moments of volume, scaled by density so every tensor below is in
        # SI mass units (kg m^2) rather than geometric units (m^5).
        second = rho * np.einsum("f,fij->ij", tet_vol, sum_sq + sum_first) / 20.0
        mass = rho * volume
        density_min = density_max = rho
        density_field_kind = "uniform"
    else:
        corners = np.empty((tet_vol.shape[0], 4, 3), dtype=float)
        corners[:, 1:] = np.stack((a, b, c), axis=1)
        corners[:, 0] = p0 + apex  # relative coordinates of the apex are zero
        shape = np.asarray(density_field(corners.reshape(-1, 3)), dtype=float)
        if shape.size != corners.shape[0] * 4:
            raise ValueError(
                "density_field must return one value per point: expected "
                f"{corners.shape[0] * 4}, got {shape.size}"
            )
        shape = shape.reshape(corners.shape[0], 4)
        if not np.all(np.isfinite(shape)):
            raise ValueError("density_field returned non-finite values")
        rho_corner = rho * shape
        if np.any(rho_corner <= 0.0):
            raise ValueError(
                "density_field must be positive on every tetrahedron corner "
                f"(min {float(rho_corner.min()):.3g} kg/m^3)"
            )
        rel = corners - p0 - apex
        corner_sum = rel.sum(axis=1)
        # int x rho dV, with rho = sum_k lambda_k rho_k and
        # int lambda_k lambda_m dV = V/20 (1 + delta_km).
        first_moment = np.einsum(
            "f,fk,fkb->b", tet_vol / 20.0, rho_corner, corner_sum[:, None, :] + rel
        )
        # int xi xj rho dV, with int lambda_k lambda_m lambda_n dV from the
        # closed-form barycentric cube moments.
        scaled_cube = tet_vol[:, None, None, None] * _BARYCENTRIC_TRIPLE_MOMENTS
        second = np.einsum(
            "f,fk,fkmn,fmb,fnc->bc", rho_corner, scaled_cube, rel, rel
        )
        mass = float(np.sum(tet_vol * rho_corner.sum(axis=1) / 4.0))
        density_min = float(rho_corner.min())
        density_max = float(rho_corner.max())
        density_field_kind = "field"

    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError(
            f"integrated mass is not positive ({mass:.3g} kg); check the density field"
        )
    centroid_apex = first_moment / mass

    second = 0.5 * (second + second.T)
    inertia_apex = np.trace(second) * np.eye(3) - second

    centroid = p0 + apex + centroid_apex
'''
assert old_body in src
src = src.replace(old_body, new_body, 1)

old_ret = '''        "face_count": int(tris.shape[0]),
        "density": rho,
    }
'''
new_ret = '''        "face_count": int(tris.shape[0]),
        "density": rho,
        "density_field": density_field_kind,
        "density_min": float(density_min),
        "density_max": float(density_max),
    }
'''
assert old_ret in src
src = src.replace(old_ret, new_ret, 1)

# ------------------------------------------------- export the constants used ---
# --------------------------------------------------- wetted_hull pass-through ---
old_wh = '''def wetted_hull_mass_properties(
    vertices: ArrayLike,
    faces: ArrayLike,
    density: float = DEFAULT_DENSITY,
    *,
    reference_point: ArrayLike = (0.0, 0.0, 0.0),
) -> dict[str, object]:'''
new_wh = '''def wetted_hull_mass_properties(
    vertices: ArrayLike,
    faces: ArrayLike,
    density: float = DEFAULT_DENSITY,
    *,
    reference_point: ArrayLike = (0.0, 0.0, 0.0),
    density_field: "Callable[[FloatArray], ArrayLike] | None" = None,
) -> dict[str, object]:'''
assert old_wh in src
src = src.replace(old_wh, new_wh, 1)

old_wh_body = '''    props = mesh_mass_properties(
        solid["vertices"], solid["faces"], density, reference_point=reference_point
    )'''
new_wh_body = '''    props = mesh_mass_properties(
        solid["vertices"],
        solid["faces"],
        density,
        reference_point=reference_point,
        density_field=density_field,
    )'''
assert old_wh_body in src
src = src.replace(old_wh_body, new_wh_body, 1)

old_wh_doc = '''    ``solid`` (the closure report of :func:`wetted_hull_solid`) and
    ``waterplane_lid_area``.
    """'''
new_wh_doc = '''    ``solid`` (the closure report of :func:`wetted_hull_solid`) and
    ``waterplane_lid_area``.

    ``density_field`` is the inhomogeneous-mass hook of
    :func:`mesh_mass_properties` (``rho(x) = density * f(x)``, ``f`` evaluated on
    the hull nodes *and* the waterline-lid fan nodes, so the field is continuous
    across the closure).  With it the ``centroid`` returned here is the **mass**
    centroid rather than the volume centroid -- that is the whole point, since a
    non-uniform mass distribution moves the CG away from the volume centroid and
    a CG displaced by ``r`` adds ``-m S(r)`` blocks to ``M_RB``.
    """'''
assert old_wh_doc in src
src = src.replace(old_wh_doc, new_wh_doc, 1)

# `Callable` import
old_imp = '''import math
from typing import Sequence
'''
new_imp = '''import math
from typing import Callable, Sequence
'''
assert old_imp in src
src = src.replace(old_imp, new_imp, 1)

p.write_text(src)
print("rigid_body_mass.py patched")

# ------------------------------------------------------------------ hydrostatic ---
h = ROOT / "src/tensorbem/hydrostatic.py"
hs = h.read_text()
old_h_sig = '''def mass_matrix_from_wetted_hull(
    hull: "WettedHullMesh",
    mass: float | None = None,
    center_of_gravity: FloatArray | None = None,
    *,
    density: float = _DEFAULT_DENSITY,
    reference_point: FloatArray = (0.0, 0.0, 0.0),
    inertia: FloatArray | None = None,
) -> FloatArray:'''
new_h_sig = '''def mass_matrix_from_wetted_hull(
    hull: "WettedHullMesh",
    mass: float | None = None,
    center_of_gravity: FloatArray | None = None,
    *,
    density: float = _DEFAULT_DENSITY,
    reference_point: FloatArray = (0.0, 0.0, 0.0),
    inertia: FloatArray | None = None,
    density_field: "Callable[[FloatArray], ArrayLike] | None" = None,
) -> FloatArray:'''
assert old_h_sig in hs
hs = hs.replace(old_h_sig, new_h_sig, 1)

old_h_body = '''    props = wetted_hull_mass_properties(
        hull.vertices, hull.faces, density, reference_point=reference_point
    )
    body_mass = float(density * props["displaced_volume"]) if mass is None else float(mass)'''
new_h_body = '''    props = wetted_hull_mass_properties(
        hull.vertices,
        hull.faces,
        density,
        reference_point=reference_point,
        density_field=density_field,
    )
    if mass is None:
        # Neutral buoyancy: the displaced mass.  With a density field that is the
        # integrated ``int rho dV`` over the closed hull solid, which coincides
        # with ``density * displaced_volume`` for the homogeneous body.
        body_mass = (
            float(props["mass"])
            if density_field is not None
            else float(density * props["displaced_volume"])
        )
    else:
        body_mass = float(mass)'''
assert old_h_body in hs
hs = hs.replace(old_h_body, new_h_body, 1)

old_h_doc = '''    inertia :
        Optional explicit tensor about the CG; when given it is passed through
        unchanged, so this function can serve as the single assembly entry point.

    Returns'''
new_h_doc = '''    inertia :
        Optional explicit tensor about the CG; when given it is passed through
        unchanged, so this function can serve as the single assembly entry point.
    density_field :
        Optional dimensionless shape factor ``f(x)`` with ``rho(x) = density *
        f(x)``, forwarded to :func:`tensorbem.rigid_body_mass.
        wetted_hull_mass_properties`.  This is how a hull whose mass is *not*
        uniformly distributed (ballast forward, machinery aft) is represented
        without hand-supplying an inertia tensor: the CG becomes the mass
        centroid and the rotational block is integrated from the same field, so
        the parallel-axis shift and the products of inertia stay consistent.

    Returns'''
assert old_h_doc in hs
hs = hs.replace(old_h_doc, new_h_doc, 1)

if "from typing import Callable" not in hs:
    import re

    m = re.search(r"^from typing import .*$", hs, flags=re.MULTILINE)
    assert m, "typing import not found"
    line = m.group(0)
    names = [n.strip() for n in line[len("from typing import "):].split(",")]
    if "Callable" not in names:
        names = sorted(set(names + ["Callable"]))
        hs = hs.replace(line, "from typing import " + ", ".join(names), 1)

h.write_text(hs)
print("hydrostatic.py patched")