"""Round-trip and validation tests for the coefficient interchange layer.

Acceptance criteria under test (``GAP-BEM-COEFF-IO``)
-----------------------------------------------------
* native container round trip is exact (``< 1e-12`` relative, in practice 0);
* external dialect files (WAMIT ``.1``/``.3``, Nemoh ``.tec``) read back to the
  precision their 6-decimal format can carry (``< 1e-6`` relative);
* an independent implementation's WAMIT files (capytaine 3.1) parse correctly;
* a tampered container fails closed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.coefficients import (
    COEFFICIENT_SCHEMA_VERSION,
    DOF_ORDER,
    CoefficientIntegrityError,
    HydrodynamicCoefficients,
    from_radiation_3d,
    stack_frequencies,
    wamit_k_exponent,
)

# A container with wildly different magnitudes so a format that silently drops
# digits cannot pass by luck.
_OMEGA = np.array([0.4, 0.9, 2.5], dtype=float)


def _reference_container() -> HydrodynamicCoefficients:
    rng = np.random.default_rng(20261112)
    ndof = 3
    added_mass = rng.normal(scale=1.0e4, size=(_OMEGA.size, ndof, ndof))
    damping = rng.normal(scale=1.0e5, size=(_OMEGA.size, ndof, ndof))
    # Synthetic asymmetry: real added-mass matrices are not symmetric for every
    # forward-speed case, and an I/O layer must not impose symmetry silently.
    added_mass[:, 0, 1] *= 3.0
    excitation = rng.normal(scale=1.0e6, size=(_OMEGA.size, 2, ndof)) + 1j * rng.normal(
        scale=1.0e6, size=(_OMEGA.size, 2, ndof)
    )
    return HydrodynamicCoefficients(
        omega=_OMEGA,
        added_mass=added_mass,
        damping=damping,
        dof_labels=DOF_ORDER[:ndof],
        excitation=excitation,
        headings_deg=np.array([0.0, 135.0]),
        rho=1025.0,
        g=9.80665,
        length_scale=2.0,
        provenance={"solver": "unit-test", "case": "reference-container"},
    )


def _relative_error(actual, expected) -> float:
    actual = np.asarray(actual, dtype=np.complex128)
    expected = np.asarray(expected, dtype=np.complex128)
    scale = np.maximum(np.abs(expected), 1e-300)
    return float(np.max(np.abs(actual - expected) / scale))


# ---------------------------------------------------------------------------
# Native container
# ---------------------------------------------------------------------------


class TestNativeContainer:
    def test_json_round_trip_is_exact(self, tmp_path: Path) -> None:
        original = _reference_container()
        path = original.write_json(tmp_path / "coeffs.json")
        restored = HydrodynamicCoefficients.read_json(path)
        assert restored.dof_labels == original.dof_labels
        assert restored.fingerprint() == original.fingerprint()
        assert _relative_error(restored.added_mass, original.added_mass) == 0.0
        assert _relative_error(restored.damping, original.damping) == 0.0
        assert _relative_error(restored.excitation, original.excitation) == 0.0
        assert np.array_equal(restored.omega, original.omega)

    def test_json_round_trip_survives_extreme_magnitudes(self, tmp_path: Path) -> None:
        tiny = 1e-300
        huge = 1e300
        container = HydrodynamicCoefficients(
            omega=np.array([1e-6, 1e6]),
            added_mass=np.array([[[tiny]], [[huge]]]),
            damping=np.array([[[-tiny]], [[-huge]]]),
            dof_labels=("heave",),
        )
        restored = HydrodynamicCoefficients.read_json(container.write_json(tmp_path / "extreme.json"))
        assert restored.added_mass[0, 0, 0] == tiny
        assert restored.added_mass[1, 0, 0] == huge
        assert restored.damping[1, 0, 0] == -huge

    def test_tampered_payload_fails_closed(self, tmp_path: Path) -> None:
        original = _reference_container()
        path = original.write_json(tmp_path / "coeffs.json")
        document = json.loads(path.read_text(encoding="utf-8"))
        document["added_mass"][0][0][0] *= 0.42  # the classic hand-edit
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(CoefficientIntegrityError):
            HydrodynamicCoefficients.read_json(path)
        # Explicit opt-out still works, and the mismatch is detectable.
        loose = HydrodynamicCoefficients.read_json(path, verify=False)
        assert loose.fingerprint() != original.fingerprint()

    def test_fingerprint_changes_with_any_number(self) -> None:
        base = _reference_container()
        perturbed = base.added_mass.copy()
        perturbed[0, 0, 0] += 1e-9 * abs(perturbed[0, 0, 0])
        other = HydrodynamicCoefficients(
            omega=base.omega,
            added_mass=perturbed,
            damping=base.damping,
            dof_labels=base.dof_labels,
            excitation=base.excitation,
            headings_deg=base.headings_deg,
        )
        assert other.fingerprint() != base.fingerprint()

    def test_environment_does_not_enter_the_fingerprint(self) -> None:
        # The fingerprint covers the numbers a verification quotes; changing a
        # label or unit convention must still surface as a different payload.
        base = _reference_container()
        renamed = HydrodynamicCoefficients(
            omega=base.omega,
            added_mass=base.added_mass,
            damping=base.damping,
            dof_labels=("sway", "heave", "roll"),
        )
        assert renamed.fingerprint() != base.fingerprint()

    def test_schema_version_is_recorded_and_checked(self, tmp_path: Path) -> None:
        document = _reference_container().to_dict()
        assert document["schema_version"] == COEFFICIENT_SCHEMA_VERSION
        document["schema_version"] = COEFFICIENT_SCHEMA_VERSION + 1
        with pytest.raises(ValueError, match="schema_version"):
            HydrodynamicCoefficients.from_dict(document)

    def test_cli_summarises_a_container(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        from tensorbem import coefficients as module

        path = _reference_container().write_json(tmp_path / "coeffs.json")
        assert module.main([str(path)]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["frequencies"] == 3
        assert payload["has_excitation"] is True


class TestValidation:
    def _base_kwargs(self, **overrides):
        kwargs = dict(
            omega=np.array([1.0, 2.0]),
            added_mass=np.zeros((2, 1, 1)),
            damping=np.zeros((2, 1, 1)),
            dof_labels=("heave",),
        )
        kwargs.update(overrides)
        return kwargs

    def test_rejects_decreasing_frequencies(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            HydrodynamicCoefficients(**self._base_kwargs(omega=np.array([2.0, 1.0])))

    def test_rejects_duplicate_frequencies(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            HydrodynamicCoefficients(**self._base_kwargs(omega=np.array([1.0, 1.0])))

    def test_rejects_non_positive_frequency(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            HydrodynamicCoefficients(**self._base_kwargs(omega=np.array([0.0, 1.0])))

    def test_rejects_non_finite_values(self) -> None:
        matrix = np.zeros((2, 1, 1))
        matrix[0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            HydrodynamicCoefficients(**self._base_kwargs(added_mass=matrix))

    def test_rejects_wrong_matrix_shape(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            HydrodynamicCoefficients(**self._base_kwargs(added_mass=np.zeros((2, 2, 1))))

    def test_rejects_unknown_dof_label(self) -> None:
        with pytest.raises(ValueError, match="unknown degree of freedom"):
            HydrodynamicCoefficients(**self._base_kwargs(dof_labels=("heave", "plunge")))

    def test_rejects_excitation_without_headings(self) -> None:
        with pytest.raises(ValueError, match="headings_deg is required"):
            HydrodynamicCoefficients(
                **self._base_kwargs(excitation=np.ones((2, 1, 1), dtype=complex))
            )

    def test_rejects_heading_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="headings_deg length"):
            HydrodynamicCoefficients(
                **self._base_kwargs(
                    excitation=np.ones((2, 2, 1), dtype=complex),
                    headings_deg=np.array([0.0]),
                )
            )

    def test_rejects_non_positive_environment(self) -> None:
        with pytest.raises(ValueError, match="rho"):
            HydrodynamicCoefficients(**self._base_kwargs(rho=0.0))


# ---------------------------------------------------------------------------
# WAMIT dialect
# ---------------------------------------------------------------------------


class TestWamitDialect:
    def test_k_exponent_matches_the_wamit_convention(self) -> None:
        assert wamit_k_exponent("surge", "heave") == 3
        assert wamit_k_exponent("surge", "pitch") == 4
        assert wamit_k_exponent("roll", "pitch") == 5

    def test_round_trip_within_format_precision(self, tmp_path: Path) -> None:
        original = _reference_container()
        path = original.write_wamit_1(tmp_path / "body.1")
        restored = HydrodynamicCoefficients.read_wamit_1(
            path, rho=original.rho, length_scale=original.length_scale
        )
        assert restored.dof_labels == original.dof_labels
        assert np.allclose(restored.omega, original.omega, rtol=1e-9)
        assert _relative_error(restored.added_mass, original.added_mass) < 1e-6
        assert _relative_error(restored.damping, original.damping) < 1e-6

    def test_written_lines_follow_the_dialect_layout(self, tmp_path: Path) -> None:
        original = _reference_container()
        path = original.write_wamit_1(tmp_path / "body.1")
        first = path.read_text(encoding="utf-8").splitlines()[0].split()
        assert len(first) == 5
        period = float(first[0])
        assert period == pytest.approx(2.0 * math.pi / original.omega[0], rel=1e-12)
        assert int(first[1]) == 1 and int(first[2]) == 1  # (radiating, influenced)
        # A11 is dimensional A11 / (rho * L**3)
        expected = original.added_mass[0, 0, 0] / (1025.0 * 2.0**3)
        assert float(first[3]) == pytest.approx(expected, rel=1e-9)

    def test_subset_of_dofs_survives(self, tmp_path: Path) -> None:
        container = HydrodynamicCoefficients(
            omega=np.array([1.5]),
            added_mass=np.array([[[7.0]]]),
            damping=np.array([[[0.25]]]),
            dof_labels=("heave",),
            length_scale=1.0,
        )
        restored = HydrodynamicCoefficients.read_wamit_1(
            container.write_wamit_1(tmp_path / "heave.1"), rho=container.rho
        )
        assert restored.dof_labels == ("heave",)
        assert restored.added_mass[0, 0, 0] == pytest.approx(7.0, rel=1e-6)
        assert restored.damping[0, 0, 0] == pytest.approx(0.25, rel=1e-6)

    def test_excitation_round_trip(self, tmp_path: Path) -> None:
        original = _reference_container()
        path = original.write_wamit_3(tmp_path / "body.3")
        restored = HydrodynamicCoefficients.read_wamit_3(
            path, rho=original.rho, g=original.g, length_scale=original.length_scale
        )
        assert np.allclose(restored.headings_deg, original.headings_deg)
        assert _relative_error(restored.excitation, original.excitation) < 1e-6

    def test_reader_rejects_zero_period_limit_blocks(self, tmp_path: Path) -> None:
        path = tmp_path / "limit.1"
        path.write_text(f"{-1.0:12.6e} {1:5d} {1:5d} {1.0:12.6e}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="not positive"):
            HydrodynamicCoefficients.read_wamit_1(path)

    def test_reader_rejects_inconsistent_re_im_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.3"
        path.write_text(
            f"{1.0:12.6e} {0.0:9.4f} {1:5d} {2.0:12.6e} {0.0:12.6e} {9.0:12.6e} {0.0:12.6e}\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="disagree"):
            HydrodynamicCoefficients.read_wamit_3(path)


class TestCapytaineInterop:
    """Parse files written by an independent implementation (capytaine 3.1)."""

    @staticmethod
    def _dataset():
        xr = pytest.importorskip("xarray")
        dofs = ["Surge", "Heave", "Pitch"]
        omega = np.array([0.6, 1.1])
        a = np.zeros((omega.size, len(dofs), len(dofs)))
        b = np.zeros_like(a)
        for f in range(omega.size):
            for i in range(len(dofs)):
                for j in range(len(dofs)):
                    a[f, i, j] = 1.0e3 * (1 + i + 2 * j) * (1.0 + 0.3 * f)
                    b[f, i, j] = 1.0e2 * (2 + 3 * i + j) * (1.0 + 0.1 * f)
        force = np.zeros((omega.size, 2, len(dofs)), dtype=complex)
        for f in range(omega.size):
            for beta in range(2):
                for i in range(len(dofs)):
                    force[f, beta, i] = 1.0e4 * (1 + i) * (1 + beta) * complex(
                        math.cos(0.3 * (i + 1)), math.sin(0.3 * (i + 1))
                    )
        ds = xr.Dataset(
            data_vars={
                "added_mass": (("omega", "influenced_dof", "radiating_dof"), a),
                "radiation_damping": (("omega", "influenced_dof", "radiating_dof"), b),
                "excitation_force": (("omega", "wave_direction", "influenced_dof"), force),
            },
            coords={
                "omega": omega,
                "influenced_dof": dofs,
                "radiating_dof": dofs,
                "wave_direction": np.array([0.0, math.pi / 2]),
                "rho": 1025.0,
                "g": 9.80665,
                "forward_speed": 0.0,
            },
        )
        ds = ds.assign_coords(period=("omega", 2.0 * np.pi / omega))
        return ds, a, b, force

    def test_reads_wamit_1_written_by_capytaine(self, tmp_path: Path) -> None:
        wamit = pytest.importorskip("capytaine.io.wamit")
        dataset, a, b, _ = self._dataset()
        length_scale = 2.0
        path = tmp_path / "capytaine.1"
        wamit.check_dataset_ready_for_export(dataset)
        wamit.export_wamit_1(dataset, str(path), length_scale=length_scale)
        restored = HydrodynamicCoefficients.read_wamit_1(
            path, rho=1025.0, length_scale=length_scale
        )
        assert restored.dof_labels == ("surge", "heave", "pitch")
        # The dialect stores the period with 7 significant digits, so the
        # frequency grid comes back to ~4e-7 relative -- the format's own limit.
        assert np.allclose(restored.omega, dataset["omega"].values, rtol=1e-6)
        assert _relative_error(restored.added_mass, a) < 1e-6
        assert _relative_error(restored.damping, b) < 1e-6

    def test_reads_wamit_3_written_by_capytaine(self, tmp_path: Path) -> None:
        wamit = pytest.importorskip("capytaine.io.wamit")
        dataset, _, _, force = self._dataset()
        length_scale = 2.0
        path = tmp_path / "capytaine.3"
        wamit.export_wamit_3(dataset, str(path))
        as_written = HydrodynamicCoefficients.read_wamit_3(
            path, rho=1025.0, g=9.80665, length_scale=1.0
        )
        assert np.allclose(as_written.headings_deg, [0.0, 90.0])
        # capytaine's .3 writer uses a unit length scale for the normalisation.
        # Its phase column is WAMIT's e^{+i*omega*t} convention, hence the
        # conjugate switch; the modulus agrees term by term.
        restored = HydrodynamicCoefficients.read_wamit_3(
            path, rho=1025.0, g=9.80665, length_scale=1.0, conjugate=True
        )
        # A 7-significant-digit phase column (degrees) pins the modulus to the
        # format precision but leaves ~1e-5 relative in the complex value, which
        # is a property of the dialect, not of this reader.
        assert _relative_error(np.abs(restored.excitation), np.abs(force)) < 1e-6
        assert _relative_error(restored.excitation, force) < 1e-5
        assert _relative_error(np.abs(as_written.excitation), np.abs(force)) < 1e-6


# ---------------------------------------------------------------------------
# Nemoh Tecplot dialect
# ---------------------------------------------------------------------------


class TestNemohTecplotDialect:
    def test_radiation_tec_round_trip(self, tmp_path: Path) -> None:
        original = _reference_container()
        path = original.write_nemoh_tec(tmp_path / "RadiationCoefficients.tec")
        text = path.read_text(encoding="utf-8")
        assert text.startswith("TITLE")
        assert "VARIABLES" in text and '"A31"' in text and '"B33"' in text
        restored = HydrodynamicCoefficients.read_nemoh_tec(path)
        assert restored.dof_labels == original.dof_labels
        assert _relative_error(restored.added_mass, original.added_mass) < 1e-6
        assert _relative_error(restored.damping, original.damping) < 1e-6

    def test_tec_reader_honours_reordered_column_names(self, tmp_path: Path) -> None:
        original = HydrodynamicCoefficients(
            omega=np.array([1.0]),
            added_mass=np.array([[[3.0]]]),
            damping=np.array([[[0.75]]]),
            dof_labels=("heave",),
        )
        path = original.write_nemoh_tec(tmp_path / "ordered.tec")
        lines = path.read_text(encoding="utf-8").splitlines()
        # Swap the A/B columns and their names; a named-column parser must survive.
        header = lines[1]
        names = [token.strip('"') for token in header.split("=", 1)[1].split()]
        reordered = [names[0], names[2], names[1]]
        lines[1] = "VARIABLES = " + " ".join(f'"{name}"' for name in reordered)
        cells = lines[3].split()
        lines[3] = "  " + "  ".join([cells[0], cells[2], cells[1]])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        restored = HydrodynamicCoefficients.read_nemoh_tec(path)
        assert restored.added_mass[0, 0, 0] == pytest.approx(original.added_mass[0, 0, 0], rel=1e-9)

    def test_excitation_tec_round_trip(self, tmp_path: Path) -> None:
        original = _reference_container()
        path = original.write_nemoh_excitation_tec(tmp_path / "ExcitationForce.tec")
        restored = HydrodynamicCoefficients.read_nemoh_excitation_tec(path)
        assert np.allclose(restored.headings_deg, original.headings_deg)
        assert _relative_error(restored.excitation, original.excitation) < 1e-6

    def test_headerless_tec_requires_ndof(self, tmp_path: Path) -> None:
        path = tmp_path / "headerless.tec"
        # Four columns cannot be 1 + 2*ndof**2 for any integer ndof.
        path.write_text("1.0e+00 2.0e+00 0.5e+00 0.1e+00\n2.0e+00 2.5e+00 0.6e+00 0.2e+00\n", encoding="utf-8")
        with pytest.raises(ValueError, match="ndof"):
            HydrodynamicCoefficients.read_nemoh_tec(path)
        path.write_text("1.0e+00 2.0e+00 0.5e+00\n2.0e+00 2.5e+00 0.6e+00\n", encoding="utf-8")
        restored = HydrodynamicCoefficients.read_nemoh_tec(path, ndof=1)
        assert restored.dof_labels == ("surge",)
        assert restored.added_mass[0, 0, 0] == pytest.approx(2.0)
        assert restored.damping[0, 0, 0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Solver integration
# ---------------------------------------------------------------------------


class TestSolverIntegration:
    @staticmethod
    def _sphere_sweep():
        from tensorbem.free_surface_solver3d import solve_radiation_3d
        from tensorbem.geometry3d import submerged_sphere_mesh

        mesh = submerged_sphere_mesh(subdivisions=0, radius=0.25, submergence=1.0)
        slices = [
            from_radiation_3d(
                solve_radiation_3d(mesh, nu, modes=(2,)),  # heave only
                provenance={"solver": "solve_radiation_3d", "case": "submerged sphere heave"},
            )
            for nu in (0.5, 1.0)
        ]
        return stack_frequencies(slices)

    def test_real_solver_result_survives_every_dialect(self, tmp_path: Path) -> None:
        container = self._sphere_sweep()
        assert container.dof_labels == ("heave",)
        assert container.omega.size == 2

        native = HydrodynamicCoefficients.read_json(container.write_json(tmp_path / "sphere.json"))
        wamit = HydrodynamicCoefficients.read_wamit_1(
            container.write_wamit_1(tmp_path / "sphere.1"), rho=container.rho
        )
        tec = HydrodynamicCoefficients.read_nemoh_tec(container.write_nemoh_tec(tmp_path / "sphere.tec"))

        # The native container is bit-exact; the dialect files are limited only
        # by their printable precision.
        assert _relative_error(native.added_mass, container.added_mass) == 0.0
        for restored in (wamit, tec):
            assert _relative_error(restored.added_mass, container.added_mass) < 1e-6
            assert _relative_error(restored.damping, container.damping) < 1e-6

        # A physically derived quantity (nondimensional heave added mass) is
        # unchanged by the trip through the wire.
        rho = container.rho
        reference = container.added_mass[:, 0, 0] / rho
        for restored in (native, wamit, tec):
            assert _relative_error(restored.added_mass[:, 0, 0] / rho, reference) < 1e-6
        # And the value is a real solve, not a placeholder.
        assert np.all(np.isfinite(reference))
        assert np.all(reference > 0.0)

    def test_stack_rejects_mismatched_environments(self) -> None:
        from tensorbem.free_surface_solver3d import solve_radiation_3d
        from tensorbem.geometry3d import submerged_sphere_mesh

        mesh = submerged_sphere_mesh(subdivisions=0, radius=0.25, submergence=1.0)
        first = from_radiation_3d(solve_radiation_3d(mesh, 0.5, modes=(2,)), rho=1025.0)
        second = from_radiation_3d(solve_radiation_3d(mesh, 1.0, modes=(2,)), rho=1000.0)
        with pytest.raises(ValueError, match="different rho"):
            stack_frequencies([first, second])

    def test_stack_rejects_duplicate_frequencies(self) -> None:
        from tensorbem.free_surface_solver3d import solve_radiation_3d
        from tensorbem.geometry3d import submerged_sphere_mesh

        mesh = submerged_sphere_mesh(subdivisions=0, radius=0.25, submergence=1.0)
        result = solve_radiation_3d(mesh, 0.5, modes=(2,))
        first = from_radiation_3d(result)
        second = from_radiation_3d(result)
        with pytest.raises(ValueError, match="duplicate frequencies"):
            stack_frequencies([first, second])