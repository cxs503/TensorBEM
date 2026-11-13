# Case-pool extension — rotational null modes of the floating hemisphere

Anchor `CASE-BEM-HULME-HEMISPHERE` grades surge and heave radiation plus the two exciting forces of a floating hemisphere against the Hulme multipole series. This extension covers the remaining four entries of the 6x6 radiation matrix (roll, pitch, yaw and their couplings) with an *analytic* reference: all of them are exactly zero, because the wetted hull is a spherical surface concentric with the rotation axis.

## 1. Case, geometry, parameters

| item | value |
|---|---|
| body | surface-piercing hemisphere, radius a = 1 m, waterplane z = 0 |
| rotation centre | sphere centre (0, 0, 0) |
| displaced volume (exact) | 2/3 pi a^3 = 2.094395 m^3 |
| waterplane area (exact) | pi a^2 = 3.141593 m^2 |
| fluid | rho = 1025 kg/m^3, g = 9.80665 m/s^2, infinite depth |
| ka band | 0.2, 0.5, 0.8, 1, 1.5, 2 (ka = omega^2 a / g, a = 1 m) |
| meshes | subdivisions 2, subdivisions 3, subdivisions 4 constant panels |
| normalisation | A_ij / (rho a^(e_i+e_j)), B_ij / (rho omega a^(e_i+e_j)), e = 1 translation, 2 rotation |
| time convention | exp(-i omega t) |

## 2. Method

1. **Analytic fact.** On the sphere carrying the hull, `n = r/a`, hence `n . (Omega x r) = Omega . (r x n) = 0` for every rotation axis through the centre. The rotational radiating modes therefore have a homogeneous body boundary condition, and the only solution compatible with the free-surface and radiation conditions is `phi = 0`: the rotational block of `A` and `B` is exactly zero, for every frequency, and for any non-zero forward speed-free linear potential model.
2. **Reciprocity closes the rows.** At zero forward speed the added-mass and damping matrices are symmetric, and every entry of column `j` is the projection of `phi_j` onto a rigid-body flux. With `phi_4 = phi_5 = phi_6 = 0` whole columns 4..6 vanish, and symmetry gives rows 4..6 as well -- including the couplings such as `A_42` (roll-sway) that mirror symmetry alone allows.
3. **What is measured.** The same statement on the discrete hull: a polyhedron's facet normals are not radial, so the residual is pure discretisation error and must vanish under refinement. Two orders are reported separately: the self terms (`phi_j = 0` itself) and the reciprocity-forced couplings (`n_roll` acting on a non-zero `phi_sway`).
3. **Discriminating test.** The null alone cannot tell a correct moment arm from a wrong one. Moving the rotation axis to `z = -d` makes the flux `n . (x_axis x (r + d z_axis)) = n_4 - d n_2`, so by linearity of the radiation solve the exact identity `A_44^c = A_44 - 2 d A_42 + d^2 A_22` must hold to round-off. A wrong rotation centre, a transposed moment arm or a missing translational coupling breaks it immediately.
5. **Foreign code.** `capytaine` (third party, never imported from `src/`) is run over its own mesh ladder on the same body and band, to show that the null is a property of the geometry rather than of this solver, and its rotation centre convention is probed with the same offset identity.

## 3. Results

### 3.1 Rotational block: residual in the case band

| ka | panels | A44 | A55 | A66 | B44 | B55 | B66 | A42 | A24 | A51 | block max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 64 | 7.729e-04 | 7.729e-04 | 5.925e-04 | 2.286e-06 | 2.286e-06 | 6.178e-15 | -0.012 | -0.012 | 0.012 | 0.012 |
| 0.2 | 256 | 8.845e-05 | 8.845e-05 | 7.220e-05 | 1.826e-07 | 1.826e-07 | 4.403e-16 | -0.004 | -0.004 | 0.004 | 0.004 |
| 0.5 | 64 | 8.232e-04 | 8.232e-04 | 5.966e-04 | 3.372e-05 | 3.372e-05 | 2.030e-11 | -0.015 | -0.015 | 0.015 | 0.015 |
| 0.5 | 256 | 9.252e-05 | 9.252e-05 | 7.252e-05 | 2.706e-06 | 2.706e-06 | 1.456e-12 | -0.004 | -0.004 | 0.004 | 0.004 |
| 0.8 | 64 | 8.485e-04 | 8.485e-04 | 6.014e-04 | 1.080e-04 | 1.080e-04 | 1.196e-09 | -0.016 | -0.016 | 0.016 | 0.016 |
| 0.8 | 256 | 9.454e-05 | 9.454e-05 | 7.290e-05 | 8.666e-06 | 8.666e-06 | 8.632e-11 | -0.005 | -0.005 | 0.005 | 0.005 |
| 0.8 | 1024 | 1.009e-05 | 1.009e-05 | 8.536e-06 | 5.788e-07 | 5.788e-07 | 5.596e-12 | -0.001 | -0.001 | 0.001 | 0.001 |
| 1 | 64 | 8.356e-04 | 8.356e-04 | 6.051e-04 | 1.605e-04 | 1.605e-04 | 8.023e-09 | -0.015 | -0.015 | 0.015 | 0.015 |
| 1 | 256 | 9.349e-05 | 9.349e-05 | 7.319e-05 | 1.283e-05 | 1.283e-05 | 5.814e-10 | -0.004 | -0.004 | 0.004 | 0.004 |
| 1 | 1024 | 1.002e-05 | 1.002e-05 | 8.556e-06 | 8.563e-07 | 8.563e-07 | 3.779e-11 | -0.001 | -0.001 | 0.001 | 0.001 |
| 1.5 | 64 | 7.544e-04 | 7.544e-04 | 6.169e-04 | 2.321e-04 | 2.321e-04 | 2.343e-07 | -0.009 | -0.009 | 0.009 | 0.013 |
| 1.5 | 256 | 8.711e-05 | 8.711e-05 | 7.411e-05 | 1.835e-05 | 1.835e-05 | 1.721e-08 | -0.003 | -0.003 | 0.003 | 0.004 |
| 1.5 | 1024 | 9.599e-06 | 9.599e-06 | 8.619e-06 | 1.222e-06 | 1.222e-06 | 1.126e-09 | -6.556e-04 | -6.565e-04 | 6.556e-04 | 0.001 |
| 2 | 64 | 6.818e-04 | 6.818e-04 | 6.347e-04 | 2.421e-04 | 2.421e-04 | 2.328e-06 | -0.005 | -0.005 | 0.005 | 0.013 |
| 2 | 256 | 8.161e-05 | 8.161e-05 | 7.550e-05 | 1.905e-05 | 1.905e-05 | 1.746e-07 | -0.001 | -0.001 | 0.001 | 0.004 |
| 2 | 1024 | 9.238e-06 | 9.238e-06 | 8.713e-06 | 1.267e-06 | 1.267e-06 | 1.153e-08 | -3.436e-04 | -3.455e-04 | 3.436e-04 | 9.497e-04 |

### 3.2 Non-zero block (shown so the null is not a solver failure)

| ka | panels | A11 | A22 | A33 | A14 (pitch-surge) | A42 (roll-sway) |
|---|---|---|---|---|---|---|
| 0.2 | 64 | 1.085 | 1.085 | 1.620 | 0.012 | -0.012 |
| 0.2 | 256 | 1.137 | 1.137 | 1.652 | 0.004 | -0.004 |
| 0.5 | 64 | 1.254 | 1.254 | 1.205 | 0.015 | -0.015 |
| 0.5 | 256 | 1.324 | 1.324 | 1.222 | 0.004 | -0.004 |
| 0.8 | 64 | 1.259 | 1.259 | 0.962 | 0.016 | -0.016 |
| 0.8 | 256 | 1.322 | 1.322 | 0.979 | 0.005 | -0.005 |
| 0.8 | 1024 | 1.339 | 1.339 | 0.983 | 0.001 | -0.001 |
| 1 | 64 | 1.141 | 1.141 | 0.871 | 0.015 | -0.015 |
| 1 | 256 | 1.186 | 1.186 | 0.891 | 0.004 | -0.004 |
| 1 | 1024 | 1.198 | 1.198 | 0.896 | 0.001 | -0.001 |
| 1.5 | 64 | 0.767 | 0.767 | 0.772 | 0.009 | -0.009 |
| 1.5 | 256 | 0.771 | 0.771 | 0.804 | 0.003 | -0.003 |
| 1.5 | 1024 | 0.771 | 0.771 | 0.812 | 6.556e-04 | -6.556e-04 |
| 2 | 64 | 0.536 | 0.536 | 0.749 | 0.005 | -0.005 |
| 2 | 256 | 0.527 | 0.527 | 0.797 | 0.001 | -0.001 |
| 2 | 1024 | 0.523 | 0.523 | 0.809 | 3.436e-04 | -3.436e-04 |

### 3.3 Offset-axis superposition (ka = 1, 256 panels)

| d / a | A44^c | predicted | rel dev | B44^c | predicted | rel dev | A44^c / (d^2 A22) |
|---|---|---|---|---|---|---|---|
| 0 | 0.0958 | 0.0958 | 0 | 0.0412 | 0.0412 | 0 | n/a |
| 0.25 | 78.2617 | 78.2618 | 4.792e-07 | 148.1843 | 148.1770 | 4.891e-05 | 1.0297 |
| 0.5 | 308.4363 | 308.4363 | 2.432e-07 | 582.8980 | 582.8835 | 2.487e-05 | 1.0145 |
| 0.75 | 690.6194 | 690.6195 | 1.629e-07 | 1304.1824 | 1304.1607 | 1.667e-05 | 1.0096 |

### 3.4 Foreign code (capytaine) rotational residual

| panels | residual (dimensionless) |
|---|---|
| 64 | 1.527e-04 |
| 256 | 1.055e-05 |
| 576 | 2.098e-06 |
| 1024 | 6.613e-07 |

observed order 3.93; provenance: capytaine 3.1.dev0 on loongarch64.

## 4. Figures

`figure.png` — left: rotational null residual vs panel count for every ka (gate line dashed); right: A44^c vs rotation-axis offset with the exact superposition curve.

## 5. Notes, pitfalls, scope

- **Why the graded case stops at surge/heave**: it is not an omission. Roll/pitch/yaw about the sphere centre are null modes; reporting a non-zero 'roll added mass' for this body would be pure discretisation noise.
- **The residual is not error-free freedom**: it is bounded by this study (`0.001` dimensionless at the case mesh) and must keep falling; a genuine implementation defect (e.g. a moment arm about the wrong point) produces a residual that does *not* fall and breaks the offset identity.
- **Kernel independence**: the reference value zero never touches the free-surface Green function, so this axis is independent of the shared-kernel caveat that applies to the Hulme series comparison.
- **Scope**: zero-speed linear potential flow, infinite depth, rigid body. Forward speed would add a `m`-term and destroy the identity on the hull (the steady-flow wetted surface is no longer a sphere through the rotation axis); the null statement must not be reused there.
- **Yaw is exactly zero even on the mesh** for lat-lon-style meshes because the facet-normal deviation is meridional; the projected-octahedron mesh used here only cancels it up to the symmetry gate.

## 6. Error judgement

| gate | measured | criterion | verdict |
|---|---|---|---|
| G1_rotational_block_null_below_gate | 0.001 | max dimensionless |A_ij|, |B_ij| over the rotational block (rows and columns roll/pitch/yaw) < 0.005 on the case mesh, every ka | PASS |
| G2_rotational_self_null_below_gate | 1.009e-05 | max |A_jj|, |B_jj| for roll/pitch/yaw < 0.0001 on the case mesh | PASS |
| G3_symmetry_nulls_machine_zero | 6.132e-10 | mirror-symmetry nulls (surge/heave, sway/heave, roll/pitch, roll/yaw, pitch/yaw) < 1e-06 dimensionless (pairwise cancellation) | PASS |
| G4_null_removal_order | 3.715 | observed order of the rotational residual, coarse -> case mesh, >= 1 | PASS |
| G5_offset_axis_superposition | 4.891e-05 | A_44^c = A_44 - 2 d A_42 + d^2 A_22 and the same for B, every offset, relative deviation < 0.001 | PASS |
| G6_graded_single_dof_untouched | 0 | the 6-DOF solve reproduces the single-DOF surge/sway/heave values to < 1e-09 relative on every level and ka | PASS |
| G7_foreign_code_null_agrees | 6.613e-07 | capytaine's own rotational residual < 0.005 dimensionless at its finest level and decreasing with refinement | PASS |
| G8_foreign_code_superposition | 1.828e-06 | capytaine reproduces A_44^c = A_44 - 2 d A_42 + d^2 A_22 (rotation centre is the sphere centre) to < 0.001 relative | PASS |

Analytic reference = 0 for every rotational entry, so the '<3 %' question is answered in absolute terms rather than against a table:

| quantity | case-mesh residual (dimensionless) | % of rho a^5 | <3 %? |
|---|---|---|---|
| rotational self terms | 1.009e-05 | 0.001 | yes |
| full rotational block (incl. A42/A51) | 0.001 | 0.121 | yes |

Both are discretisation residuals, not offsets: the observed order under refinement is 3.72 (>= 1 required), and the offset-axis identity holds to 4.891e-05 relative -- a wrong rotation centre or a transposed moment arm would break it by O(1).

**Verdict: PASS**

