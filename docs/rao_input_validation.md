# RAO input validation: garbage-in / garbage-out in the impedance solve

**Scope**: `src/tensorbem/rao.py` — `compute_rao_at_frequency`,
`compute_rao_spectrum`, `estimate_natural_frequency`.
**Artifacts**: `docs/rao_input_validation/scan.json` (generated),
`src/tensorbem/benchmarks/rao_input_validation.py` (driver),
`tests/test_rao_input_validation.py` (44 tests).
**Mission**: m6375 (self-directed, TensorBEM).

---

## 1. Case: what the entry point accepted

`compute_rao_at_frequency` assembles the frequency-domain equation of motion

    Z(ω) ξ = F_exc,     Z = -ω²(M + A(ω)) + iω B(ω) + C

from a radiation solve, a diffraction solve and a hydrostatic restoring matrix.
`Z` is solved with `numpy.linalg.solve`, so **every finite input produces a
finite output**. Nothing in the function checked the body state, the frequency,
the inertia tensor or the extra damping matrix. The rest of the package does
check its state (`tensorbem.ice` `_positive`/`_non_negative`,
`tensorbem.time_domain` rejects negative damping and Nyquist violations,
`tensorbem.drift_force` rejects malformed sweeps, `tensorbem.pc6_hull`
validates `Cb/L/B` before any hydrodynamics), so the RAO path was the outlier —
and it is the path the ship-level RAO results go through.

The probe below is a hemisphere of 16 panels, `ω = 1 rad/s`, `m = 1000 kg`,
`cg = (0, 0, -0.1) m`, `ρ = 1025 kg/m³`, `g = 9.80665 m/s²`.

## 2. Method

* Split the former body of `compute_rao_at_frequency` into
  `_assemble_rao_at_frequency` (raw algebra, documented as unvalidated) and a
  validating public wrapper. Keeping the raw assembly callable is deliberate:
  it makes the *before* column of the table below reproducible after the fix
  (`benchmarks/rao_input_validation.py`), instead of quoting a scan from an old
  revision.
* New guards, in the package's existing style (`ValueError` naming the
  offending quantity): `_positive`, `_finite_scalar`, `_real_vector`,
  `_symmetric_matrix` (with an optional non-negative-diagonal rule for
  damping), `_omega_sweep`, plus a shared `_validated_common` used by both the
  single-frequency and the sweep entry point.
* `estimate_natural_frequency` now also requires `added_mass_inf` and
  `restoring_matrix` to be finite `(6, 6)` matrices (it indexed them blindly).
* Regression guard: for a well-posed input the validating path must be
  **bit-identical** (`np.array_equal`, same `impedance_residual`) to the raw
  assembly — validation must not silently change the physics.

Reproduce:

```
PYTHONPATH=src python3 -m tensorbem.benchmarks.rao_input_validation \
    --json docs/rao_input_validation/scan.json
PYTHONPATH=src python3 -m pytest tests/test_rao_input_validation.py -q
```

## 3. Results (numbers from `docs/rao_input_validation/scan.json`)

### 3.1 Single frequency: 13/14 malformed inputs now rejected

| input | raw assembly (the old behaviour) | public API now |
|---|---|---|
| baseline (valid) | heave 0.981479 m/m, max 1.200980 | accepted (bit-identical) |
| `mass = -1000 kg` | **returns** heave 0.910908, max **23.962598** m/m | `ValueError` |
| `mass = 0 kg` | **returns** heave 0.944878 m/m | `ValueError` |
| `mass = NaN` | **returns** 6/6 NaN entries | `ValueError` |
| `omega = -1 rad/s` | **returns** exactly the `omega = +1` RAO (0.981479) | `ValueError` |
| `omega = NaN` | rejected downstream (nu check) | `ValueError` |
| `cg = (0, 0, NaN)` | **returns** 6/6 NaN entries | `ValueError` |
| `cg` shape (2,) | `IndexError` from `x, y, z = cg` | `ValueError` |
| `inertia` 2×2 | numpy broadcast error | `ValueError` |
| `inertia` asymmetric | **returns** heave 0.981479 (asymmetry used as given) | `ValueError` |
| `inertia` NaN | **returns** 6/6 NaN entries | `ValueError` |
| `extra_damping` shape (6,) | **returns**, vector broadcast over the matrix | `ValueError` |
| `extra_damping` NaN | **returns** 6/6 NaN entries | `ValueError` |
| `extra_damping = -1e6·I` | **returns** heave 0.025335 (energy source accepted) | `ValueError` |

### 3.2 Frequency sweep: 4/5 malformed sweeps now rejected

| input | before | now |
|---|---|---|
| `omegas = [-1, -2]` | **returns** the positive-frequency spectrum | `ValueError` |
| `omegas` contains 0 | **returns** | `ValueError` |
| `omegas` contains NaN | rejected downstream | `ValueError` |
| `omegas` shape (1, 2) | **returns** | `ValueError` |
| `omegas = [0.6, 1.2]` (valid) | returns heave 0.993484, 0.972951 | unchanged |

### 3.3 Verification numbers

| check | reference | measured | ≤ 3 % ? |
|---|---|---|---|
| valid path vs raw assembly (`rao`, `impedance_residual`) | bit equality | `np.array_equal` = True, residual identical | n/a (exact) |
| sweep vs single-frequency calls | `np.array_equal` | True for both frequencies | n/a (exact) |
| `estimate_natural_frequency` vs `sqrt(C/(m+A))` | closed form | rel. error ≤ 1e-12 (test tolerance) | yes |
| impedance residual, valid baseline | ≈ 0 | 3.67e-12 | yes |

## 4. Figures

None. This change is an input-contract fix in the impedance assembly: there is
no field quantity, geometry or frequency sweep whose discretisation behaviour
changes, so a contour plot or a convergence curve would be decoration rather
than evidence. The evidence is the categorical reject/return table above, which
is machine-generated (`scan.json`) rather than transcribed.

## 5. Caveats / scope

* **Archimedes equilibrium is deliberately not enforced.** `mass` versus
  `ρ·∇` is *not* checked: the historical test cases and several benchmark
  drivers legitimately use `m ≠ ρ∇` (e.g. `tests/test_hydrostatic_rao.py` uses
  `m = 1000 kg` for a hemisphere displacing 2147 kg). A mismatch shows up
  physically as a long-wave heave RAO that does not tend to 1, which
  `tests/test_hydrostatic_analytic_benchmark.py` already exercises. Raising on
  it here would break documented cases for no diagnostic gain; the docstring
  now states this explicitly.
* **Symmetry of `inertia` / `extra_damping` is required** (relative tolerance
  `1e-8`); an asymmetric tensor is non-physical, and the solver previously used
  the asymmetry verbatim. No caller in the repository passes an asymmetric
  tensor (checked: `benchmarks/hydrostatic_rao_verification.py`,
  `benchmarks/ice_ship_verification.py`, `tests/test_ice_rao.py`,
  `tests/test_hydrostatic_rao.py`).
* **Non-negative damping diagonal only.** Full passivity of `B + extra_damping`
  (positive semi-definiteness) is not checked; only the diagonal is, because
  off-diagonal negative entries are not per se non-physical for a coupled
  matrix. A near-singular `Z` at resonance still raises `LinAlgError` from
  `numpy`; it is now preceded by validated inputs, so it can no longer be a
  silent NaN.
* The validation does not touch the BEM kernels, the hydrostatic integration or
  the CHIEF suppression; the numbers of a well-posed solve are unchanged
  (bit-identical), so no previously published RAO number moves.
* Empty `omegas` is still allowed (returns zero-length arrays); it is a
  degenerate but well-defined request, not an inconsistent state.

## 6. Error verdict versus references

| quantity | baseline | criterion | verdict |
|---|---|---|---|
| valid-solve equality (validated vs raw) | exact bit equality of `rao`, identical residual 3.67e-12 | must be exact | **pass (0 %)** |
| sweeps vs single-frequency | exact | must be exact | **pass (0 %)** |
| natural frequency vs closed form | ≤ 1e-12 rel. | < 3 % | **pass** |
| malformed inputs rejected | 13/14 single-frequency, 4/5 sweep (the 14th/5th are the valid control cases) | 100 % of non-physical cases | **pass** |

No experimental or literature reference applies to this change: it is an
input-contract defect, not a discretisation or modelling error, so there is no
"error against measurement" to report — the honest statement is that the
defect was a *false acceptance*, and the fix is measured by the categorical
scan, not by a percentage.