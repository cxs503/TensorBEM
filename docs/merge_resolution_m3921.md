# Merge Resolution Report: PR #104 (agent/m2930 → main)

**Mission**: m3921  
**Date**: 2026-11-05  
**Resolver**: AI Agent (TensorPlatform)

## Conflict Summary

PR #104 (branch `agent/m2930`) became unmergeable with `main` due to add/add conflicts in two files:

1. `src/tensorbem/finite_depth3d.py` (953 lines in conflict)
2. `tests/test_finite_depth3d.py` (445 lines in conflict)

Two other files from `agent/m2930` merged cleanly:
- `docs/finite_depth3d_offaxis_diagnosis.md` (268 lines)
- `src/tensorbem/benchmarks/finite_depth3d_verification.py` (225 lines)

## Nature of Conflicts

Both conflicting files contained **completely different implementations** of the 3D finite-depth free-surface Green function:

### Main Branch Implementation (John Eigenseries)
- **Method**: John (1950) modal expansion with vertical eigenfunctions
- **Approach**: Spectral decomposition using propagating mode (k0) and evanescent modes (μn)
- **Special functions**: Modified Bessel K₀ and Hankel H₀⁽¹⁾
- **Status**: **VERIFIED** with comprehensive test suite (26 tests)
- **Key features**:
  - Pure NumPy Bessel function implementations
  - Exponential convergence of evanescent sum
  - Machine-precision satisfaction of free-surface and bottom conditions
  - Deep-water limit cross-check against independent Kelvin kernel (<1e-3 error)
  - Shallow-water asymptotic validation

### Agent/m2930 Implementation (Integral Representation)
- **Method**: Integral representation with Gauss-Legendre quadrature
- **Approach**: Direct numerical integration of spectral integral with principal value handling
- **Special functions**: J₀, Y₀ Bessel functions via quadrature
- **Status**: **UNVERIFIED** (explicitly marked in docstring)
- **Known issues**:
  - Deep-water cross-check fails for off-axis field points (relative errors: 0.34, 0.56, 3.2)
  - On-axis point converges to 1.9e-6 at νh=36
  - Self-documented as "do not use in a solver"

## Resolution Strategy

**Decision**: Keep main branch implementation as primary code.

**Rationale**:
1. Main branch version is **verified** with 26 comprehensive physics-based tests
2. Agent/m2930 version is explicitly marked **UNVERIFIED** with known failures
3. Main branch implementation follows the classical John (1950) eigenseries approach (roadmap item 6)
4. No silent loss of content: agent/m2930's diagnostic documentation preserved

**Semantic Union**:
- ✅ Main's verified implementation retained (`finite_depth3d.py`, `test_finite_depth3d.py`)
- ✅ Agent/m2930's diagnostic documentation preserved (`finite_depth3d_offaxis_diagnosis.md`)
- ✅ Agent/m2930's verification benchmark preserved (`finite_depth3d_verification.py`)
- ❌ Agent/m2930's unverified implementation superseded (documented, not lost)

## Test Evidence

### Main Branch Tests (26/26 passed)
```
tests/test_finite_depth3d.py::test_dispersion_relation_closes (9 variants) PASSED
tests/test_finite_depth3d.py::test_shallow_water_wavenumber_asymptote PASSED
tests/test_finite_depth3d.py::test_evanescent_roots_bracket_dispersion_relation PASSED
tests/test_finite_depth3d.py::test_vertical_modes_are_orthogonal_with_analytic_norms PASSED
tests/test_finite_depth3d.py::test_wave_amplitude_coefficient_normalisation PASSED
tests/test_finite_depth3d.py::test_free_surface_condition_is_satisfied (3 depths) PASSED
tests/test_finite_depth3d.py::test_bottom_is_impermeable PASSED
tests/test_finite_depth3d.py::test_analytic_z_gradient_matches_central_difference PASSED
tests/test_finite_depth3d.py::test_kernel_is_harmonic_at_field_points PASSED
tests/test_finite_depth3d.py::test_source_singularity_is_unit_strength PASSED
tests/test_finite_depth3d.py::test_deep_water_limit_recovers_kelvin_kernel PASSED
tests/test_finite_depth3d.py::test_shallow_water_wave_term_matches_closed_form PASSED
tests/test_finite_depth3d.py::test_far_field_is_outgoing_with_wavenumber_k0 PASSED
tests/test_finite_depth3d.py::test_evanescent_truncation_self_convergence PASSED
tests/test_finite_depth3d.py::test_bessel_helpers_match_scipy PASSED
tests/test_finite_depth3d.py::test_invalid_inputs_are_rejected PASSED
```

### Related Module Tests (18/18 passed)
```
tests/test_finite_depth.py (14 tests) PASSED
tests/test_free_surface3d.py (4 tests) PASSED
```

## Files Changed

### Added (from agent/m2930, merged cleanly)
- `docs/finite_depth3d_offaxis_diagnosis.md` - Diagnostic documentation for off-axis failures
- `src/tensorbem/benchmarks/finite_depth3d_verification.py` - Verification benchmark suite

### Resolved (conflicts fixed by keeping main)
- `src/tensorbem/finite_depth3d.py` - John eigenseries implementation (main)
- `tests/test_finite_depth3d.py` - Comprehensive verification suite (main)

## Conclusion

PR #104 is now mergeable. The merge preserves:
- Main's verified, production-ready implementation
- Agent/m2930's diagnostic documentation and benchmark infrastructure
- No silent loss of content from either branch

The unverified integral representation from agent/m2930 is superseded but documented in the diagnostic report for future reference.

## Next Steps

1. Merge PR #104 into main
2. Close PR #104
3. Optional: Investigate whether the integral representation approach can be repaired (see `docs/finite_depth3d_offaxis_diagnosis.md`)
