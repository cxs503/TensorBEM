# CASE-BEM-HULME-DIFF-VERIF: Hulme (1982) Floating Hemisphere Diffraction Verification

**Case Code**: BEM-HULME-HEMISPHERE  
**Mission**: m17427 (rework of m17278)  
**Reference**: Hulme, A. (1982). "The wave forces acting on a floating hemisphere undergoing forced periodic oscillations." *J. Fluid Mech.* **121**, 443-463.  
**Tolerance**: <5% relative error on all non-dimensional hydrodynamic quantities  
**Status**: HYDRODYNAMIC PASS (G1), GEOMETRY CONDITIONAL (G2 at sub=3, PASS at sub=4)

---

## 1. Problem Description and Reference

### Physical Problem
A surface-piercing hemisphere of radius $a = 1.0$ m floating in deep water, subjected to regular head-on waves. The flat face is flush with the mean free surface ($z = 0$), and the curved surface is submerged ($z < 0$).

### Geometry
- **Radius**: $a = 1.0$ m
- **Draft**: $T = a = 1.0$ m
- **Displaced volume** (exact): $\nabla = \frac{2}{3}\pi a^3 = 2.0944$ m³
- **Waterplane area** (exact): $A_{wp} = \pi a^2 = 3.1416$ m²

### Reference Solution
Hulme (1982) semi-analytical solution using wave-free multipole expansion in spherical harmonics. Implemented in `tensorbem.analytic.hulme_hemisphere_radiation` with 144 terms (truncation ladder 48/96/144 shows geometric convergence; 96→144 drift < 6.5×10⁻⁵, at least 20× below the 5% gate).

**Note on paper digit transcription**: The original Hulme (1982) paper is behind a publisher paywall and not available in this sandbox. The reference values used here are the re-derived series evaluated to 144 terms, verified by:
1. Self-convergence of the truncation ladder (48→96→144)
2. Independent cross-check against capytaine (third-party BEM code)
3. Low-frequency asymptotic limits (A33/(ρ∇) → 0.835 as ka → 0)

---

## 2. Method and Implementation

### Numerical Approach
3-D deep-water Kelvin (Havelock) Green-function BEM solver (`tensorbem.free_surface_solver3d`):
1. **Radiation problem**: Prescribed rigid-body motions (surge mode 0, heave mode 2) at unit velocity amplitude
2. **Diffraction problem**: Incident plane wave scattered by the fixed hemisphere
3. **Irregular frequency suppression**: Interior CHIEF null-field equations (`chief_points='auto'`)

### Implementation Files (non-placeholder)
| File | Role |
|---|---|
| `src/tensorbem/free_surface_solver3d.py:464` | `solve_radiation_3d` — Kelvin/Havelock Green-function BIE |
| `src/tensorbem/free_surface_solver3d.py:577` | `solve_diffraction_3d` — diffraction solve |
| `src/tensorbem/analytic.py:1138` | `hulme_hemisphere_radiation` — wave-free multipole series reference |
| `src/tensorbem/geometry3d.py:581` | `hemisphere_hull_mesh` — octahedral subdivision mesh generator |
| `src/tensorbem/benchmarks/hulme_diffraction_verification.py` | Benchmark driver with `run_benchmark`, `run_single_cell`, `evaluate_verdict` entry points |

### Integration Callers
| Caller | How |
|---|---|
| `src/tensorbem/benchmarks/__init__.py` | `from tensorbem.benchmarks import hulme_diffraction_verification` |
| `src/tensorbem/capability_evidence.py` | Listed in `modules` for capability registration |
| `tests/test_benchmark_cli_contract.py` | `--help` contract test |
| `tests/test_hulme_diffraction_verification.py` | Direct BEM vs Hulme comparison (5 tests) |
| `tests/test_hulme_benchmark_integration.py` | Integration entry point tests (6 tests) |
| `tests/test_casepool_hulme_hemisphere.py` | Case-pool CI guard (4 tests) |
| `benchmarks/verified/casepool_bem_hulme_hemisphere/run_case.py` | Full case-pool driver with predeclared gates |

### Entry Point
```bash
PYTHONPATH=src python -m tensorbem.benchmarks.hulme_diffraction_verification --subdivisions 3 --report report.md --figure figure.png --json results.json
```

---

## 3. Parameter Table

| Parameter | Symbol | Value | Non-dimensional |
|---|---|---|---|
| Radius | $a$ | 1.0 m | — |
| Density | $\rho$ | 1025.0 kg/m³ | — |
| Gravity | $g$ | 9.80665 m/s² | — |
| Wave amplitude | $A$ | 1.0 m | — |
| Displaced volume | $\nabla$ | 2.0944 m³ | — |
| Waterplane area | $A_{wp}$ | 3.1416 m² | — |
| Frequency parameter | $ka$ | 0.2, 0.5, 0.8, 1.0, 1.5, 2.0 | — |
| Mesh (case) | sub=3 | 256 panels | — |
| Mesh (coarse) | sub=2 | 64 panels | — |
| Mesh (fine) | sub=4 | 1024 panels | — |
| Normalisation (added mass) | $A_{jj}/(\rho\nabla)$ | — | ✓ |
| Normalisation (damping) | $B_{jj}/(\rho\omega\nabla)$ | — | ✓ |
| Normalisation (force) | $|F|/(\rho g A a^2)$ | — | ✓ |

---

## 4. Reference vs Computed Comparison

### Case mesh (256 panels, subdivisions=3)

| ka | A11 ref | A11 got | err | B11 ref | B11 got | err | A33 ref | A33 got | err | B33 ref | B33 got | err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 0.5515 | 0.5429 | -1.57% ✓ | 0.0082 | 0.0078 | -3.78% ✓ | 0.7938 | 0.7888 | -0.63% ✓ | 0.2793 | 0.2770 | -0.81% ✓ |
| 0.5 | 0.6439 | 0.6321 | -1.84% ✓ | 0.0987 | 0.0948 | -3.87% ✓ | 0.5861 | 0.5835 | -0.45% ✓ | 0.3391 | 0.3386 | -0.12% ✓ |
| 0.8 | 0.6421 | 0.6314 | -1.68% ✓ | 0.2653 | 0.2554 | -3.75% ✓ | 0.4698 | 0.4672 | -0.55% ✓ | 0.2899 | 0.2917 | +0.64% ✓ |
| 1.0 | 0.5740 | 0.5665 | -1.30% ✓ | 0.3535 | 0.3410 | -3.53% ✓ | 0.4284 | 0.4253 | -0.73% ✓ | 0.2484 | 0.2514 | +1.20% ✓ |
| 1.5 | 0.3683 | 0.3681 | -0.05% ✓ | 0.4014 | 0.3899 | -2.84% ✓ | 0.3890 | 0.3837 | -1.37% ✓ | 0.1606 | 0.1651 | +2.80% ✓ |
| 2.0 | 0.2493 | 0.2514 | +0.86% ✓ | 0.3424 | 0.3347 | -2.24% ✓ | 0.3884 | 0.3804 | -2.05% ✓ | 0.1031 | 0.1080 | +4.76% ✓ |

| ka | \|Fx\| ref | \|Fx\| got | err | phase ref | phase got | \|Fz\| ref | \|Fz\| got | err | phase ref | phase got |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 0.5844 | 0.5724 | -2.05% ✓ | -89.7° | -89.7° | 2.4186 | 2.4091 | -0.39% ✓ | -2.8° | -2.8° |
| 0.5 | 1.2858 | 1.2586 | -2.12% ✓ | -87.0° | -87.1° | 1.6854 | 1.6849 | -0.03% ✓ | -12.6° | -12.6° |
| 0.8 | 1.6668 | 1.6320 | -2.09% ✓ | -83.0° | -83.2° | 1.2320 | 1.2360 | +0.32% ✓ | -25.2° | -25.1° |
| 1.0 | 1.7209 | 1.6865 | -2.00% ✓ | -81.8° | -82.0° | 1.0201 | 1.0256 | +0.54% ✓ | -34.3° | -34.2° |
| 1.5 | 1.4972 | 1.4714 | -1.73% ✓ | -87.8° | -88.0° | 0.6697 | 0.6765 | +1.01% ✓ | -58.5° | -58.4° |
| 2.0 | 1.1976 | 1.1790 | -1.55% ✓ | -103.9° | -103.9° | 0.4646 | 0.4735 | +1.91% ✓ | -84.1° | -84.1° |

**All 36 quantity×ka cells pass the <5% tolerance at 256 panels.**

Worst error: 4.76% (B33 at ka=2.0).

---

## 5. Convergence and Parameter Sensitivity

### 5.1 Mesh convergence at ka=1.0

| Subdivisions | Panels | A33 err | B33 err | Fx err | Fz err | Vol err |
|---|---|---|---|---|---|---|
| 2 | 64 | -2.97% | +4.52% | -7.58% | +2.05% | -8.86% |
| 3 | 256 | -0.73% | +1.20% | -2.00% | +0.54% | -2.32% |
| 4 | 1024 | (cluster) | (cluster) | (cluster) | (cluster) | -0.59% |

**Observed convergence order** (sub=2 → sub=3, h-ratio = 2):
- A33: ~2.01 (heave added mass)
- B33: ~1.92 (heave damping)
- Fx: ~1.92 (surge excitation)
- Fz: ~1.92 (heave excitation)

Consistent with O(h²) convergence for constant-panel BEM on a curved surface.

### 5.2 Haskind reciprocity

| ka | Haskind surge err | Haskind heave err |
|---|---|---|
| 0.2 | 1.2e-03 | 9.8e-04 |
| 0.5 | 1.5e-03 | 1.2e-03 |
| 0.8 | 1.8e-03 | 1.4e-03 |
| 1.0 | 9.6e-04 | 7.6e-04 |
| 1.5 | 1.1e-03 | 8.5e-04 |
| 2.0 | 8.7e-04 | 6.9e-04 |

All Haskind reciprocity errors < 0.2%, confirming internal consistency of radiation and diffraction solves.

### 5.3 Reference truncation convergence

| ka | 48→96 drift | 96→144 drift | Ratio |
|---|---|---|---|
| 0.2 | 8.4e-05 | 1.6e-05 | 0.19 |
| 0.5 | 1.0e-04 | 1.9e-05 | 0.19 |
| 1.0 | 1.5e-04 | 2.8e-05 | 0.19 |
| 2.0 | 3.4e-04 | 6.5e-05 | 0.19 |

Geometric tail ratio ~0.19 confirms exponential convergence of the multipole series. The 96→144 drift is at least 20× below the 5% gate at all ka.

---

## 6. Failure Modes and Limitations

### Geometry precondition (G2)
At sub=3 (256 panels), the mesh volume error is -2.32%, which exceeds the 2% geometry gate. This is the **geometry floor**: every coefficient error is bounded below by the geometry error. At sub=4 (1024 panels), the volume error drops to -0.59%, passing the gate. The sub=4 results are being computed on the cluster.

### Irregular frequencies
The unregularized BIE exhibits a condition-number spike near ka ≈ 2.58 (first interior waterplane resonance). The CHIEF method suppresses this effectively. All band points stay below ka = 2.0, safely away from the irregular frequency.

### Paper digit transcription
The original Hulme (1982) paper is behind a paywall. Reference values are the re-derived series, verified by self-convergence and independent cross-checks (see Section 1).

### What is NOT verified
- Experimental validation (no experimental data in repository)
- Forward speed effects (zero-speed theory only)
- Nonlinear effects (linear potential theory only)
- Viscous effects (inviscid fluid assumed)
- Finite depth effects (deep water only)

---

## Reproduction Command

```bash
# Quick verification at 256 panels (sub=3, ~2 min)
PYTHONPATH=src python -m tensorbem.benchmarks.hulme_diffraction_verification \
    --subdivisions 3 --report runs/casepool_bem_hulme_hemisphere/report.md \
    --figure runs/casepool_bem_hulme_hemisphere/figure.png \
    --json runs/casepool_bem_hulme_hemisphere/results.json

# Full case-pool run (includes sub=4 ladder, ~30 min on cluster)
PYTHONPATH=src python benchmarks/verified/casepool_bem_hulme_hemisphere/run_case.py \
    --shard 0 --shards 1 --out runs/casepool_bem_hulme_hemisphere

# Run integration tests
PYTHONPATH=src pytest tests/test_hulme_benchmark_integration.py tests/test_hulme_diffraction_verification.py tests/test_casepool_hulme_hemisphere.py -v
```

---

## Run Artifacts

| Artifact | Path | Description |
|---|---|---|
| Results JSON (sub=3) | `runs/casepool_bem_hulme_hemisphere/results.json` | 6 ka × all quantities |
| Report (sub=3) | `runs/casepool_bem_hulme_hemisphere/report.md` | Markdown tables |
| Figure (sub=3) | `runs/casepool_bem_hulme_hemisphere/figure.png` | 4-panel comparison plot |
| Band sweep JSON | `runs/casepool_bem_hulme_hemisphere/band_sweep.json` | sub=2 and sub=3 raw data |
| Reference JSON | `runs/casepool_bem_hulme_hemisphere/reference.json` | Hulme reference values |
