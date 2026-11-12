# Wall-sided barge: corrected capytaine cross-check

*Anchor **FEAT-OSS-INVENTORY-BEM** (deep-dive wave 2). This document is rendered by `tensorbem.benchmarks.barge_crosscheck.render_document`; every number comes from `benchmarks/verified/barge_crosscheck/{results.json,capytaine_ref.json}` and is re-checked by `tests/test_barge_crosscheck.py`.*

**Verdict: FAIL** -- worst deviation on the finest mesh (640 panels): bare 24.40 %, lid 43.26 % against a pre-fixed 5 % band.

## 1. What was wrong with the recorded negative control

The old reference file excluded the barge because *"spurious irregular frequencies ... no converged reference exists"*. The producer script built the hull as `mesh_parallelepiped(size=(2,1,1), center=(0,0,-0.5)).immersed_part()`. `immersed_part()` clips only above `z = 0`, so the box's waterplane face survives: 16 faces lie on `z = 0`, the volume is 2.000 m^3 (the case is labelled `T = 0.5`, i.e. 1.0 m^3) and the heave damping comes out negative:

| nu | sealed-box heave added mass | sealed-box heave damping |
|---:|---:|---:|
| 0.500 | 1653.89 | -957.80 |
| 1.000 | 2626.84 | -3003.76 |
| 2.000 | -2134.66 | -885.43 |

A negative radiation damping is non-physical: the case was never evidence about sharp corners, it was a Neumann waterplane panel sitting exactly on the free surface.

## 2. Corrected case

| item | value |
|---|---|
| geometry | `L x B x T = 2.0 x 1.0 x 0.5 m`, flat bottom + four sides, **no** waterplane face |
| displaced volume | 1.0000 m^3 |
| fluid | rho = 1025.0 kg/m^3, g = 9.80665 m/s^2, infinite depth |
| frequencies | nu = 0.5, 1, 2 m^-1 |
| first irregular frequency (parallelepiped estimate) | nu_irr = 3.728 m^-1 |
| panel meshes | 40, 160, 640 panels (identical triangulation in both codes) |

The tested band `nu in [0.5, 2.0]` lies entirely below `nu_irr`, so the case is solvable *without* a lid; the lid run is kept as a bias measurement, not as a fix.

## 3. Cross-code results, finest mesh

| route | nu | quantity | capytaine | TensorBEM | deviation |
|---|---:|---|---:|---:|---:|
| bare | 0.5 | added mass | 952.07 | 957.63 | +0.58% |
| bare | 0.5 | damping | 747.70 | 735.16 | -1.68% |
| lid | 0.5 | added mass | 959.27 | 957.73 | -0.16% |
| lid | 0.5 | damping | 663.03 | 735.20 | +10.89% |
| bare | 1 | added mass | 779.22 | 788.07 | +1.14% |
| bare | 1 | damping | 786.31 | 764.78 | -2.74% |
| lid | 1 | added mass | 821.59 | 788.18 | -4.07% |
| lid | 1 | damping | 690.08 | 764.94 | +10.85% |
| bare | 2 | added mass | 736.74 | 750.29 | +1.84% |
| bare | 2 | damping | 363.55 | 342.82 | -5.70% |
| lid | 2 | added mass | 787.14 | 750.38 | -4.67% |
| lid | 2 | damping | 417.99 | 343.34 | -17.86% |

## 4. Convergence toward the foreign reference (worst cell per level)

| panels | worst dev. bare | worst dev. lid |
|---:|---:|---:|
| 40 | 24.40 % | 43.26 % |
| 160 | 12.54 % | 25.55 % |
| 640 | 5.70 % | 17.86 % |

## 5. Figures

* `docs/fig_barge_crosscheck_convergence.png` -- heave added mass and damping vs panel count, TensorBEM (bare) against capytaine (bare), three frequencies.

## 6. Error verdict

Cells outside the band on the finest mesh:

* panels 640 nu 0.50 lid damping: +10.89% (TensorBEM 735.20 vs capytaine 663.03)
* panels 640 nu 1.00 lid damping: +10.85% (TensorBEM 764.94 vs capytaine 690.08)
* panels 640 nu 2.00 bare damping: -5.70% (TensorBEM 342.82 vs capytaine 363.55)
* panels 640 nu 2.00 lid damping: -17.86% (TensorBEM 343.34 vs capytaine 417.99)

**Is the key quantity within 3 %?** No. The worst cell is 43.26 %, reported honestly as *not met*; the band itself is 5 %.

## 7. Caveats

* The waterplane is rectangular: the lid generator `waterplane_lid_points` supports star-shaped waterplanes only; a strongly non-convex (twin-hull) waterplane must fall back to another construction.
* Both codes are first-order constant-strength panel methods; agreement at 5 % does not certify the physics, only the implementation (both share the same linearised potential-flow model).
* The old `capytaine_crosscheck` entry is left untouched (it is the record of what was claimed); this module is the corrected measurement.

