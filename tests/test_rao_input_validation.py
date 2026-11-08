"""Non-physical RAO inputs must raise, not return a plausible-looking answer.

The impedance assembly ``Z = -omega^2 (M + A) + i omega B + C`` is a linear
solve, so *any* finite input produces a finite output.  Before this module
existed, the following all returned an RAO with no error:

======================  ======================================================
input                   what the old code did
======================  ======================================================
``mass=-1000``          solved with a negative inertia; heave amplitude ~24 m/m
``mass=0``              singular-in-mass system, still "solved"
``mass=nan``            every RAO entry came back NaN
``omega=-1``            ``_omega_to_nu`` squares omega, so it returned the RAO
                        of ``omega=+1`` exactly
``center_of_gravity=nan``  every RAO entry came back NaN
``center_of_gravity`` of  length 2   raised a bare ``IndexError`` from ``x, y, z = cg``
``inertia`` 2x2        raised a cryptic numpy broadcast error
``inertia`` asymmetric  silently used ``J + m S S^T`` with the asymmetry intact
``extra_damping`` (6,)  broadcast a vector over a matrix
``extra_damping`` = -1e6*I  negative damping (an energy source) accepted
``omegas=[-1, -2]``     returned the positive-frequency spectrum
======================  ======================================================

Both the physics convention of this package (``tensorbem.ice``,
``tensorbem.time_domain``, ``tensorbem.drift_force`` all reject non-physical
input) and the audit rule "geometry/state is validated before the
hydrodynamics run" require the entry point to raise ``ValueError`` with a
message that names the offending quantity.

``test_valid_inputs_are_untouched`` is the regression guard: for a valid input
the validating entry point must be **bit-identical** to the unvalidated
assembly ``_assemble_rao_at_frequency`` (no silent change of the physics).
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem.geometry3d import hemisphere_hull_mesh
from tensorbem.rao import (
    _assemble_rao_at_frequency,
    compute_rao_at_frequency,
    compute_rao_spectrum,
    estimate_natural_frequency,
)

OMEGA = 1.0
MASS = 1000.0
CG = np.array([0.0, 0.0, -0.1])


@pytest.fixture(scope="module")
def hull():
    """Coarse hemisphere (16 panels) — one solve costs ~0.4 s."""
    return hemisphere_hull_mesh(subdivisions=1)


def _call(hull, **overrides):
    kwargs = dict(
        hull=hull,
        omega=OMEGA,
        wave_angle=0.0,
        mass=MASS,
        center_of_gravity=CG,
    )
    kwargs.update(overrides)
    return compute_rao_at_frequency(**kwargs)


# ---------------------------------------------------------------------------
# the valid path is unchanged
# ---------------------------------------------------------------------------


def test_valid_inputs_are_untouched(hull):
    """Validation must not perturb the physics of a well-posed problem."""
    validated = _call(hull)
    raw = _assemble_rao_at_frequency(
        hull, OMEGA, 0.0, MASS, CG.copy(), None, 1.0, 1025.0, 9.80665,
        None, None, "auto",
    )
    assert np.array_equal(np.asarray(validated.rao), np.asarray(raw.rao))
    assert validated.impedance_residual == raw.impedance_residual
    assert np.all(np.isfinite(validated.rao_amplitude))


def test_valid_symmetric_inertia_is_used_verbatim(hull):
    """A symmetric inertia tensor must reach the solve unchanged."""
    inertia = np.diag([2.0e3, 2.0e3, 2.0e3])
    validated = _call(hull, inertia=inertia)
    raw = _assemble_rao_at_frequency(
        hull, OMEGA, 0.0, MASS, CG.copy(), None, 1.0, 1025.0, 9.80665,
        inertia, None, "auto",
    )
    assert np.array_equal(np.asarray(validated.rao), np.asarray(raw.rao))


def test_zero_extra_damping_is_a_no_op(hull):
    """A zero 6x6 extra-damping matrix must not change the RAO."""
    base = _call(hull)
    damped = _call(hull, extra_damping=np.zeros((6, 6)))
    assert np.array_equal(np.asarray(base.rao), np.asarray(damped.rao))


# ---------------------------------------------------------------------------
# non-physical scalars
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_mass", [-1000.0, 0.0, float("nan"), float("inf")])
def test_non_positive_or_non_finite_mass_is_rejected(hull, bad_mass):
    with pytest.raises(ValueError, match="mass must be positive and finite"):
        _call(hull, mass=bad_mass)


@pytest.mark.parametrize("bad_omega", [-1.0, 0.0, float("nan"), float("inf")])
def test_non_positive_or_non_finite_omega_is_rejected(hull, bad_omega):
    with pytest.raises(ValueError, match="omega must be positive and finite"):
        _call(hull, omega=bad_omega)


def test_negative_frequency_no_longer_masquerades_as_its_magnitude(hull):
    """omega**2 hides the sign, so ``-omega`` used to return the +omega RAO."""
    with pytest.raises(ValueError, match="omega must be positive"):
        _call(hull, omega=-OMEGA)


@pytest.mark.parametrize("bad_density", [0.0, -1025.0, float("nan")])
def test_non_positive_density_is_rejected(hull, bad_density):
    with pytest.raises(ValueError, match="density must be positive and finite"):
        _call(hull, density=bad_density)


@pytest.mark.parametrize("bad_gravity", [0.0, -9.80665, float("nan")])
def test_non_positive_gravity_is_rejected(hull, bad_gravity):
    with pytest.raises(ValueError, match="gravity must be positive and finite"):
        _call(hull, gravity=bad_gravity)


@pytest.mark.parametrize("bad_amplitude", [0.0, -1.0, float("nan")])
def test_non_positive_amplitude_is_rejected(hull, bad_amplitude):
    with pytest.raises(ValueError, match="amplitude must be positive and finite"):
        _call(hull, amplitude=bad_amplitude)


def test_non_finite_wave_angle_is_rejected(hull):
    with pytest.raises(ValueError, match="wave_angle must be finite"):
        _call(hull, wave_angle=float("nan"))


# ---------------------------------------------------------------------------
# centre of gravity and inertia
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_cg",
    [
        np.array([0.0, 0.0]),
        np.array([[0.0, 0.0, 0.0]]),
        np.array([0.0, 0.0, float("nan")]),
        np.array([0.0, np.inf, 0.0]),
    ],
)
def test_malformed_centre_of_gravity_is_rejected(hull, bad_cg):
    with pytest.raises(ValueError, match="center_of_gravity"):
        _call(hull, center_of_gravity=bad_cg)


@pytest.mark.parametrize(
    "bad_inertia",
    [
        np.eye(2),
        np.eye(4),
        np.full((3, 3), np.nan),
        np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ],
)
def test_malformed_inertia_is_rejected(hull, bad_inertia):
    with pytest.raises(ValueError, match="inertia"):
        _call(hull, inertia=bad_inertia)


# ---------------------------------------------------------------------------
# extra damping
# ---------------------------------------------------------------------------


def test_extra_damping_vector_is_not_broadcast(hull):
    """A (6,) damping vector used to be broadcast over the 6x6 matrix."""
    with pytest.raises(ValueError, match=r"extra_damping must have shape \(6, 6\)"):
        _call(hull, extra_damping=np.zeros(6))


@pytest.mark.parametrize(
    "bad_damping",
    [
        np.full((6, 6), np.nan),
        np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]] * 6),
        -1.0e6 * np.eye(6),
    ],
)
def test_non_physical_extra_damping_is_rejected(hull, bad_damping):
    with pytest.raises(ValueError, match="extra_damping"):
        _call(hull, extra_damping=bad_damping)


def test_negative_damping_is_named_as_an_energy_source(hull):
    with pytest.raises(ValueError, match="energy source"):
        _call(hull, extra_damping=-1.0e6 * np.eye(6))


# ---------------------------------------------------------------------------
# frequency sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_omegas,message",
    [
        ([-1.0, -2.0], "strictly positive"),
        ([1.0, 0.0], "strictly positive"),
        ([1.0, float("nan")], "must be finite"),
        ([[1.0, 2.0]], "one-dimensional"),
    ],
)
def test_malformed_frequency_sweep_is_rejected(hull, bad_omegas, message):
    with pytest.raises(ValueError, match=message):
        compute_rao_spectrum(
            hull=hull,
            omegas=bad_omegas,
            wave_angle=0.0,
            mass=MASS,
            center_of_gravity=CG,
        )


def test_sweep_rejects_non_physical_body_mass(hull):
    with pytest.raises(ValueError, match="mass must be positive and finite"):
        compute_rao_spectrum(
            hull=hull,
            omegas=[0.5, 1.0],
            wave_angle=0.0,
            mass=-1.0,
            center_of_gravity=CG,
        )


def test_sweep_still_matches_single_frequency_calls(hull):
    """The validating sweep must agree with the validating single-frequency path."""
    omegas = np.array([0.6, 1.2])
    spectrum = compute_rao_spectrum(
        hull=hull, omegas=omegas, wave_angle=0.0, mass=MASS, center_of_gravity=CG
    )
    for index, omega in enumerate(omegas):
        single = _call(hull, omega=float(omega))
        assert np.array_equal(spectrum.rao_amplitudes[index], single.rao_amplitude)


# ---------------------------------------------------------------------------
# natural-frequency helper
# ---------------------------------------------------------------------------


def test_natural_frequency_rejects_malformed_matrices():
    with pytest.raises(ValueError, match=r"added_mass_inf must have shape \(6, 6\)"):
        estimate_natural_frequency(
            MASS, np.zeros((3, 3)), np.eye(6), 2
        )
    with pytest.raises(ValueError, match=r"restoring_matrix must have shape \(6, 6\)"):
        estimate_natural_frequency(MASS, np.zeros((6, 6)), np.eye(3), 2)


def test_natural_frequency_rejects_non_finite_matrices():
    added = np.zeros((6, 6))
    added[2, 2] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        estimate_natural_frequency(MASS, added, np.eye(6), 2)


def test_natural_frequency_matches_sqrt_c_over_m_plus_a():
    added = np.zeros((6, 6))
    added[2, 2] = 500.0
    restoring = np.zeros((6, 6))
    restoring[2, 2] = 2000.0 * MASS
    omega_n = estimate_natural_frequency(MASS, added, restoring, 2)
    assert omega_n == pytest.approx(np.sqrt(2000.0 * MASS / (MASS + 500.0)), rel=1e-12)

# ---------------------------------------------------------------------------
# the reproducibility driver must classify rejections correctly
# ---------------------------------------------------------------------------


def test_probe_driver_classifies_rejections(hull):
    """``benchmarks.rao_input_validation`` marks the bad case as rejected, not returned."""
    from tensorbem.benchmarks import rao_input_validation as probe

    names = [name for name, _, _ in probe.CASES]
    assert any("mass" in name for name in names)
    assert any(name.startswith("baseline") for name in names)

    rejected = probe._attempt(probe._public, hull=hull, mass=-1000.0)
    assert rejected["returned"] is False
    assert rejected["error_type"] == "ValueError"
    assert "mass" in rejected["error"]
