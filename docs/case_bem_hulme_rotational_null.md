# Case-pool extension — rotational null modes of the floating hemisphere

**Anchor**: `CASE-BEM-HULME-HEMISPHERE` (算例池·Hulme 浮半球辐射/绕射（3D）, tolerance <5 %).
This document adds the four remaining entries of the 6×6 radiation matrix
(roll, pitch, yaw and their couplings) to that case, with an **analytic**
reference instead of a tabulated one.

Artifacts (all numbers below are rendered from them, nothing is transcribed by hand):

| artifact | path |
|---|---|
| study (source) | `src/tensorbem/benchmarks/hemisphere_rotational_null.py` |
| producer (third-party code) | `scripts/capytaine_hemisphere_rotational_null.py` |
| measured results | `benchmarks/verified/casepool_hulme_rotational_null/results.json` |
| rendered report | `benchmarks/verified/casepool_hulme_rotational_null/report.md` |
| figure | `benchmarks/verified/casepool_hulme_rotational_null/figure.png` |
| frozen foreign-code reference | `benchmarks/verified/casepool_hulme_rotational_null/capytaine_rotational_null.json` |
| tests | `tests/test_hemisphere_rotational_null.py` |

## 1. Why the graded case stops at surge and heave

The graded case reports `A11, B11, A33, B33` and the two exciting forces. That
looks like a coverage gap for a "radiation/diffraction (3-D)" case. It is not.

The wetted hull of a floating hemisphere whose rotation centre coincides with
the centre of the sphere that carries the hull satisfies `n = r/a` pointwise,
hence

```
n . (Omega x r) = Omega . (r x n) = 0      for every rotation axis through the centre
```

The rotational radiating modes therefore have a **homogeneous** body boundary
condition; the only field compatible with the free-surface condition, the
radiation condition and `phi = 0` on the hull is `phi = 0` itself. At zero
forward speed the added-mass and damping matrices are symmetric, and every entry
of column `j` is the projection of `phi_j` onto a rigid-body flux, so

```
A_ij = A_ji = B_ij = B_ji = 0     for i in {surge, sway, heave}, j in {roll, pitch, yaw}
A_44 = A_55 = A_66 = B_44 = B_55 = B_66 = 0
```

i.e. **the whole rotational block vanishes identically**, including the
roll–sway (`A42`) and pitch–surge (`A51`) couplings that mirror symmetry alone
would allow. Reporting a non-zero "roll added mass" for this body would be
discretisation noise presented as data — which is the trap this extension
documents.

## 2. What is verified, and how

1. **Null on the discrete hull.** A polyhedron's facet normals are not radial, so
   the measured rotational block is small but non-zero. It is therefore *pure
   discretisation error* and must fall to zero under refinement: measured on the
   mesh ladder (64 / 256 / 1024 constant panels) over `ka = 0.2 … 2.0`.
   Two mechanisms are separated because they converge at different orders:
   * self terms (`phi_j = 0` itself) → `O(h^3..4)`;
   * reciprocity-forced couplings (`n_roll` acting on the non-zero `phi_sway`,
     whose facet-normal error is `O(h)`) → `O(h^2)`, two decades larger.
2. **Discriminating test (the null alone is not enough).** A null cannot
   distinguish a correct moment arm from a wrong one. Rolling about an axis at
   `z = -d` changes the flux to `n_4 - d n_2`, so by linearity of the radiation
   solve the exact identity

   ```
   A_44^c = A_44 - 2 d A_42 + d^2 A_22          (same for B)
   ```

   must hold to round-off, and the null mode turns into the large, predictable
   `d^2 A_22` term. A wrong rotation centre, a transposed moment arm or a missing
   translational coupling breaks this identity by O(1).
3. **Foreign code.** `capytaine` (third party; never imported from `src/`) is run
   over its own mesh ladder under the same conditions and its rotation-centre
   convention is probed with the same offset identity, so the null is shown to be
   a property of the geometry rather than of this solver.
4. **No disturbance of the graded numbers.** The 6-DOF solve reuses the same
   operator and reproduces the single-DOF surge/sway/heave values, so the
   extension cannot have shifted anything the anchor already grades.

Reference value = **0 exactly** for every rotational entry, and it never touches
the free-surface Green function. This makes the axis *kernel-independent*, which
matters because the Hulme series reference used by the main case shares
`wave_function_3d` with the solver (a limitation registered in
`docs/bem_hulme_hemisphere_case.md`).

## 3. Pitfalls recorded

- **Degenerate mode, not a missing mode.** Any case pool entry for a
  sphere-based body must not grade roll/pitch/yaw about the sphere centre; the
  correct negative control is "this entry is a null".
- **A small number is not a small error.** The rotational residual of a *bad*
  implementation (moment arm taken about the panel centroid, axis sign flipped
  into a translation, `n x r` instead of `r x n`) is also "small" at a coarse
  mesh. Only the refinement order plus the offset identity separate the two.
- **Refinement order is the honest currency.** Quote the order, not just the
  residual: the couplings converge at `O(h^2)` and sit two decades above the
  self terms.
- **Scope limit.** Zero speed, linear potential flow, infinite depth, rigid body.
  With forward speed the steady wetted surface is no longer a sphere through the
  rotation axis and the identity must not be reused.
- **Yaw is often exactly zero on lat-lon meshes** (facet-normal deviation is
  meridional); on the projected-octahedron mesh used here it is only zero up to
  the symmetry gate.

## 4. Gates (pre-registered on the 64/256-panel pilot, before the case mesh)

| gate | criterion |
|---|---|
| `G1_rotational_block_null_below_gate` | whole rotational block < 5e-3 (dimensionless, `rho a^5` / `rho omega a^5`) on the finest mesh |
| `G2_rotational_self_null_below_gate` | self terms `A44, A55, A66, B44, B55, B66` < 1e-4 |
| `G3_symmetry_nulls_machine_zero` | mirror-symmetry nulls < 1e-6 |
| `G4_null_removal_order` | observed order of the residual under refinement ≥ 1 |
| `G5_offset_axis_superposition` | offset identity < 1e-3 relative, every `d` |
| `G6_graded_single_dof_untouched` | 6-DOF path reproduces the single-DOF values to < 1e-9 |
| `G7_foreign_code_null_agrees` | capytaine residual < 5e-3 and decreasing |
| `G8_foreign_code_superposition` | capytaine offset identity < 1e-3 |

Verdicts and measured values: `benchmarks/verified/casepool_hulme_rotational_null/report.md`.

## 5. Reproduction

```bash
# third-party half (needs capytaine; the only script that imports it)
python scripts/capytaine_hemisphere_rotational_null.py \
    --out benchmarks/verified/casepool_hulme_rotational_null/capytaine_rotational_null.json

# measured half (levels 2,3 are seconds; level 4 is ~1-2 min per ka)
PYTHONPATH=src python -m tensorbem.benchmarks.hemisphere_rotational_null \
    --levels 2 3 4 --out-dir benchmarks/verified/casepool_hulme_rotational_null

# chunked (e.g. on a queued cluster): one ka at a time, results are merged
PYTHONPATH=src python -m tensorbem.benchmarks.hemisphere_rotational_null \
    --levels 4 --band 1.0 --skip-probe --append \
    --out-dir benchmarks/verified/casepool_hulme_rotational_null

pytest tests/test_hemisphere_rotational_null.py -v
```