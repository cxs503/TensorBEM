"""Guards for the hydrostatic restoring matrix on curved walls (m4750).

``docs/hydrostatic_restoring_couplings.md`` §5 item 3 left an explicit
verification debt when the coupled matrix was completed (m3196): every case in
the suite was a **wall-sided barge with a rectangular waterplane**, and the note
believed a curved hull would need "curved-surface clipping" for an energy
reference.  It does not — the reference clips *panels*, which is agnostic to the
panel's curvature — so the debt is payable and these tests pay it.

Each guard is independent of the implementation it polices:

* **energy Hessian** — ``m4750``'s vectorised/Richardson reference, which knows
  only Archimedes plus the weight of the body and evaluates ``∫ z dV`` by a
  flux integral over the clipped panels.  It never touches a waterplane formula.
  Its integral is pinned against the original face-by-face reference
  (:mod:`tensorbem.benchmarks.hydrostatic_energy_reference`) below, so the
  speed-up costs no independence.
* **closed form** — circle/ellipse waterplane integrals plus the spherical-cap
  volume and centroid for the *exact* geometry.  This measures discretisation
  (faceting) error rather than formula validity, which is why the contract is
  "under 3 %" and "shrinks when the hull is refined", not "1e-6".

The physical claim under test is that the published waterplane formula
``C[2,2] = ρgA``, ``C[2,3] = ρgS_y``, ``C[2,4] = -ρgS_x``,
``C[3,3] = ρgI_xx + ρg∇(z_B - z_G)``, ``C[3,4] = -ρgI_xy`` is exact for
**non-wall-sided** hulls as well, i.e. that nothing beyond the waterplane
integrals enters the linear stiffness of a smoothly curved body.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.benchmarks import hydrostatic_curved_waterplane as curved
from tensorbem.benchmarks.hydrostatic_energy_reference import (
    restoring_matrix_from_energy,
    submerged_first_moment_z,
)
from tensorbem.geometry3d import floating_sphere_hull_mesh
from tensorbem.hydrostatic import compute_hydrostatic_restoring_matrix

RHO = 1025.0
G = 9.80665

CASE_KEYS = ("eccentric_sphere", "yawed_spheroid", "eccentric_yawed_spheroid")

#: Criterion of ``code vs energy``: the reference is extrapolated, so this is a
#: formula-level contract, not a difference-scheme tolerance.
ENERGY_TOLERANCE = 1.0e-6

#: Criterion of ``code vs exact geometry``: faceting of the geodesic mesh at
#: ``subdivisions=3``.  The measured worst entry is 1.5 % (``C[3,4]`` of the
#: eccentric yawed spheroid) and it falls by ~4x per 4x refinement.
CLOSED_FORM_TOLERANCE = 3.0e-2


def _case(key: str, subdivisions: int = 2) -> curved.Case:
    return {case.key: case for case in curved.build_cases(subdivisions)}[key]


# ---------------------------------------------------------------------------
# The reference itself: vectorised integral and Richardson extrapolation
# ---------------------------------------------------------------------------


def test_vectorised_first_moment_matches_the_face_by_face_reference() -> None:
    """The fast flux integral equals the Sutherland-Hodgman reference exactly."""
    case = _case("eccentric_sphere")
    rng = np.random.default_rng(20261105)
    for _ in range(4):
        xi = rng.normal(scale=0.05, size=6)
        rotation = curved._rotation_matrix(xi[3:6])
        moved = case.hull.vertices @ rotation.T + xi[0:3]
        fast = curved._restricted_second_moment_z(moved, case.hull.faces)
        reference = submerged_first_moment_z(moved, case.hull.faces)
        assert fast == pytest.approx(reference, rel=1.0e-12, abs=1.0e-12)


def test_fast_hessian_matches_the_reference_hessian_at_equal_step() -> None:
    """Same stencil, same numbers: the speed-up is not a second discretisation."""
    case = _case("eccentric_sphere", subdivisions=1)
    volume = compute_hydrostatic_restoring_matrix(
        case.hull, 0.0, np.asarray(case.center_of_gravity), RHO, G
    ).displaced_volume
    kwargs = dict(
        mass=RHO * volume,
        center_of_gravity=case.center_of_gravity,
        density=RHO,
        gravity=G,
        step=2.0e-4,
    )
    reference = restoring_matrix_from_energy(case.hull.vertices, case.hull.faces, **kwargs)
    fast = curved.restoring_matrix_from_energy_fast(
        case.hull.vertices, case.hull.faces, **kwargs
    )
    # The two sum in different orders, and the second difference divides by
    # step**2 = 4e-8, so agreement is limited to ~1e-7 of the largest entry.
    scale = np.max(np.abs(reference))
    assert np.max(np.abs(fast - reference)) <= 1.0e-6 * scale


# ---------------------------------------------------------------------------
# Formula validity on curved hulls
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", CASE_KEYS)
def test_curved_hull_matches_the_energy_hessian(key: str) -> None:
    """Every audited entry of a curved hull matches the independent energy Hessian."""
    record = curved.audit_case(_case(key, subdivisions=3))
    worst = max(row["relerr_code_vs_energy"] for row in record["rows"])
    detail = [(row["symbol"], f"{row['relerr_code_vs_energy']:.2e}") for row in record["rows"]]
    assert worst < ENERGY_TOLERANCE, (
        f"{key}: worst code-vs-energy relative deviation {worst:.3e}; entries {detail}"
    )


@pytest.mark.parametrize("key", CASE_KEYS)
def test_curved_hull_matches_the_closed_form_within_faceting_error(key: str) -> None:
    """The faceted hull's entries are within 3 % of the exact-geometry formula."""
    record = curved.audit_case(_case(key, subdivisions=3))
    worst_row = max(record["rows"], key=lambda row: row["relerr_code_vs_closed_form"])
    assert worst_row["relerr_code_vs_closed_form"] < CLOSED_FORM_TOLERANCE, (
        f"{key}: {worst_row['symbol']} deviates by "
        f"{worst_row['relerr_code_vs_closed_form']:.3e} from the closed form"
    )


def test_faceting_error_falls_under_refinement() -> None:
    """4x the panels buys ~4x less deviation: the ledger is discretisation, not bias."""
    coarse = {case.key: curved.audit_case(case, ) for case in curved.build_cases(2)}
    fine = {case.key: curved.audit_case(case) for case in curved.build_cases(3)}

    def worst(record: dict) -> float:
        return max(row["relerr_code_vs_closed_form"] for row in record["rows"])

    for key in CASE_KEYS:
        improvement = worst(coarse[key]) / worst(fine[key])
        assert improvement > 2.5, (
            f"{key}: refining {coarse[key]['panels']} -> {fine[key]['panels']} panels "
            f"only improved the closed-form deviation {improvement:.2f}x"
        )


def test_the_energy_reference_residual_is_difference_scheme_truncation() -> None:
    """The raw (non-extrapolated) Hessian residual must vanish roughly linearly in ``step``.

    A faceted hull's clipped area is only piecewise smooth, so its second
    difference is ``O(step)``; that is *why* the reference extrapolates with
    exponent 1 rather than the textbook 2.  If this ever turned quadratic the
    extrapolation would be wrong, so the behaviour is pinned.
    """
    study = curved.fd_step_study(
        steps=(4.0e-4, 2.0e-4, 1.0e-4), subdivisions=2, density=RHO, gravity=G
    )
    for key in CASE_KEYS:
        series = study["series"][key]
        coarse, fine = series[0]["worst_relative"], series[-1]["worst_relative"]
        # Refining the step 4x must buy between 2x and 6x (fitted order 0.5-1.3);
        # individual halvings are irregular because which panels get clipped by
        # the stencil changes discretely, so only the trend is pinned.
        ratio = coarse / fine
        assert 2.0 < ratio < 6.0, f"{key}: 4x finer step gave {ratio:.2f}x"
        assert study["fitted_orders"][key] < 1.4


# ---------------------------------------------------------------------------
# Structure and identity checks that do not need any reference at all
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", CASE_KEYS)
def test_surge_sway_yaw_stay_identically_zero(key: str) -> None:
    """Curvature cannot create restoring in surge/sway/yaw: vertical forces have no moment."""
    record = curved.audit_case(_case(key))
    assert record["null_block_relative"] == 0.0
    assert record["symmetry_residual"] == 0.0


def test_off_centre_curved_waterplane_couples_heave_into_roll_and_pitch() -> None:
    """``C[2,3]/C[2,2] = y_f`` and ``C[2,4]/C[2,2] = -x_f`` hold for a *circle* waterplane too.

    This is an identity of the implementation, not a reference comparison: the
    ratio is the waterplane centroid, so a curved waterline can be off-centre
    exactly like a rectangle's and the heave-roll / heave-pitch couplings follow.
    """
    case = _case("eccentric_sphere")
    props = compute_hydrostatic_restoring_matrix(
        case.hull, 0.0, np.asarray(case.center_of_gravity), RHO, G
    )
    matrix = props.restoring_matrix
    x_f, y_f = props.waterplane_center
    assert matrix[2, 2] > 0.0
    assert matrix[2, 3] / matrix[2, 2] == pytest.approx(y_f, rel=1.0e-9)
    assert matrix[2, 4] / matrix[2, 2] == pytest.approx(-x_f, rel=1.0e-9)
    # ...and they are large compared with the diagonal, so this is not a corner case.
    assert abs(matrix[2, 3]) > 0.1 * matrix[2, 2]
    assert abs(matrix[2, 4]) > 0.1 * matrix[2, 2]


def test_centred_sphere_leaves_the_heave_roll_pitch_block_diagonal() -> None:
    """The docstring's "a floating hemisphere gets a strictly diagonal block" claim.

    A centred sphere has a circular waterplane on the origin: the couplings are
    zero by symmetry, which is precisely why the historical barge/hemisphere
    cases could not see the missing terms.  Pinned on the *curved* mesh.
    """
    hull = floating_sphere_hull_mesh(2, 1.0, center=(0.0, 0.0, -0.5))
    matrix = compute_hydrostatic_restoring_matrix(
        hull, 0.0, np.array([0.0, 0.0, -0.5]), RHO, G
    ).restoring_matrix
    scale = matrix[2, 2]
    assert abs(matrix[2, 3]) < 1.0e-9 * scale
    assert abs(matrix[2, 4]) < 1.0e-9 * scale
    assert abs(matrix[3, 4]) < 1.0e-9 * scale


def test_a_yawed_elliptical_waterplane_produces_a_roll_pitch_coupling() -> None:
    """A curved waterplane with a product moment couples roll and pitch.

    ``C[3,4] = -ρg I_xy`` is the entry that a rectangle cannot show unless it is
    rotated; on the yawed spheroid it is ~93 % of the roll stiffness, so the
    term is structural rather than a perturbation.
    """
    record = curved.audit_case(_case("yawed_spheroid"))
    entries = {row["symbol"]: row for row in record["rows"]}
    coupling = entries["C[3,4]"]["code"]
    assert coupling < 0.0
    assert abs(coupling) > 0.5 * entries["C[3,3]"]["code"]
    # The centred spheroid has S_x = S_y = 0 while I_xy != 0: the two families of
    # coupling are physically independent.
    assert abs(entries["C[2,3]"]["code"]) < 1.0e-6 * entries["C[2,2]"]["code"]


def test_sphere_metacentric_height_reduces_to_the_classical_cap_formula() -> None:
    """``GM = C[3,3] / (ρg∇)`` equals ``I/∇ + z_B - z_G`` for the curved hull.

    Independent of the implementation: the same identity is reconstructed from
    the *closed-form* cap volume, centroid and waterplane inertia.
    """
    radius, depth = 1.0, 0.5
    closed = curved.sphere_closed_form(radius, depth, (0.0, 0.0), RHO, G)
    height = radius - depth
    cap_volume = math.pi * height**2 * (3.0 * radius - height) / 3.0
    cap_centroid = 3.0 * (2.0 * radius - height) ** 2 / (4.0 * (3.0 * radius - height))
    volume = 4.0 * math.pi * radius**3 / 3.0 - cap_volume
    assert closed["displaced_volume"] == pytest.approx(volume, rel=1.0e-14)
    z_b = -depth - cap_volume * cap_centroid / volume
    assert closed["center_of_buoyancy_z"] == pytest.approx(z_b, rel=1.0e-14)
    gm = closed["c33"] / (RHO * G * volume)
    assert gm == pytest.approx(closed["i_xx"] / volume + z_b + depth, rel=1.0e-12)
    # The historical value for a half-submerged sphere is GM = 0 (neutral roll):
    # BM = 3R/8 cancels BG = 3R/8 exactly, a sharp corner check of the cap algebra.
    half = curved.sphere_closed_form(radius, 0.0, (0.0, 0.0), RHO, G)
    assert half["c33"] / (RHO * G * half["displaced_volume"]) == pytest.approx(0.0, abs=1.0e-12)