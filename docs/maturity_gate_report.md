# TensorBEM Maturity Gate Report (H6-MATURITY-GATE)

## 1. Gate Coverage Summary

- **Total capabilities**: 82
- **Failable gates**: 82
- **Passing gates**: 82
- **Failing gates**: 0
- **Gate coverage**: 100.0%

### Maturity Distribution

- **L0_NOT_SUPPORTED** (Not Supported): 0
- **L1_STUB_MOCK** (Stub / Mock): 0
- **L2_IMPLEMENTED** (Implemented): 8
- **L3_VERIFIED** (Verified): 47
- **L4_CROSS_VALIDATED** (Cross-Validated): 26
- **L5_PRODUCTION** (Production): 1

## 2. Failing Gates

All failable gates pass.

## 3. Cross-Consistency Check

All cross-consistency checks pass.

## 4. Full Maturity Table

# TensorBEM Capability Maturity Ledger

| Capability | Title | Evidence | Maturity | Gate | L3+ | L4+ | L5+ |
|---|---|---|---|---|---|---|---|
| wageningen-b-series | Wageningen B-series open-water polynomials with a  | REAL | L2_IMPLEMENTED | PASS |  |  |  |
| mesh-quality | Mesh quality gates | REAL | L2_IMPLEMENTED | PASS |  | ✓ |  |
| clearance | Body / free-surface clearance diagnostics | REAL | L2_IMPLEMENTED | PASS |  |  |  |
| capability-evidence | Machine-derived capability evidence registry | REAL | L2_IMPLEMENTED | PASS |  |  |  |
| hull-mesh-geometry | Panel and triangle mesh geometry with closure chec | REAL | L2_IMPLEMENTED | PASS |  |  |  |
| figure-integrity | Verification-figure integrity gate (reproducible f | REAL | L2_IMPLEMENTED | PASS |  |  |  |
| finite-depth-green-kernel | Finite-depth Green function + dispersion solver | REAL | L2_IMPLEMENTED | PASS |  |  |  |
| reference-provenance | Machine-checked provenance of reference constants | REAL | L2_IMPLEMENTED | PASS |  | ✓ |  |
| elastic-contact-boussinesq | Axisymmetric elastic normal contact on a half-spac | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| laplace-2d | 2-D Laplace BEM (interior / exterior) | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| laplace-3d | 3-D Laplace BEM (interior / exterior) | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| free-surface-2d | 2-D deep-water free-surface Green function | REAL | L3_VERIFIED | PASS | ✓ |  | ✓ |
| free-surface-3d | 3-D Delhommeau / Kelvin free-surface Green functio | REAL | L3_VERIFIED | PASS | ✓ |  | ✓ |
| acoustic-3d-helmholtz | 3-D unbaffled exterior Helmholtz acoustic BEM (rad | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| maccamy-fuchs | MacCamy-Fuchs vertical pile (Helmholtz + CHIEF) | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| added-mass-neumann | Rigid-body added mass (2-D / 3-D Neumann) | REAL | L3_VERIFIED | PASS | ✓ |  | ✓ |
| surface-velocity-recovery | Surface velocity recovery | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| hydrostatics | Hydrostatics (volume, Cb, waterplane, restoration) | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| rao | 6-DOF heave RAO coupling | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| finite-depth-3d | Finite-depth kernels + solver (3-D) | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| michell-general-and-tandem | Michell thin-ship wave resistance for general hull | REAL | L3_VERIFIED | PASS | ✓ |  | ✓ |
| mixed-boundary | Mixed Dirichlet / Neumann boundaries | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| near-singular | Near-singular panel integration | REAL | L3_VERIFIED | PASS | ✓ |  | ✓ |
| symmetry3d | Symmetry reduction (3-D) | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| pressure-forces | Pressure integration and force recovery | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| conditioning | Conditioning diagnostics | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| doc-claim-gate | Document-versus-artifact claim gate | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| kkr-causality | Kramers-Kronig causality check | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| numeric-evidence | Numeric evidence layer (criterion + artifact + ind | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| maruo-far-field-drift | Maruo far-field mean drift force at arbitrary head | REAL | L3_VERIFIED | PASS | ✓ |  | ✓ |
| time-domain-memory-kernel | Time-domain radiation memory kernel (Cummins equat | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| damping-rational-fit | Radiation-damping rational fit (Sanathanan-Koerner | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| cummins-memory-kernel-closure | Two-way Cummins memory-kernel closure with causali | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| uncertainty-quantification | Discretisation and recovery uncertainty quantifica | REAL | L3_VERIFIED | PASS | ✓ |  | ✓ |
| linear-panel-operators | Piecewise-linear panel operators and tangential re | REAL | L3_VERIFIED | PASS | ✓ |  | ✓ |
| multiply-connected-domains | Multiply connected 2-D domains (nested loops, mult | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| thin-ship-general-hullforms | Thin-ship Michell resistance for general hullforms | REAL | L3_VERIFIED | PASS | ✓ |  | ✓ |
| quadrature-diagnostics | Near-singular quadrature and conditioning diagnost | REAL | L3_VERIFIED | PASS | ✓ |  | ✓ |
| multibody-frequency-asymptotes | Multihull added mass and damping with exact zero/i | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| qtf-slow-drift | Second-order difference-frequency QTF and slow-dri | REAL | L3_VERIFIED | PASS | ✓ |  | ✓ |
| fmm-acceleration | Fast multipole acceleration of the 3-D single-laye | REAL | L3_VERIFIED | PASS | ✓ |  | ✓ |
| flexural-gravity-numeric | Independent finite-difference check of the flexura | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| finite-depth-3d-quadrature-solver | 3-D finite-depth wave-radiation solver (eigenserie | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| flexural-gravity-dispersion | Flexural-gravity (ice-cover) dispersion and mass l | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| interior-panel-lid | Interior waterplane-lid panels (irregular-frequenc | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| irregular-frequency-sanitisation | Irregular-frequency prediction and sanitisation | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| neumann-kelvin-steady | Neumann-Kelvin steady forward-speed wave resistanc | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| nk-panel-3d | 3-D steady Neumann-Kelvin panel method (Rankine +  | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| quadratic-panel-operator | Quadratic (6-node) curved-triangle panel operator | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| propeller-open-water | Propeller open-water performance by vortex-lattice | REAL | L3_VERIFIED | PASS | ✓ |  | ✓ |
| rankine-fs-panel | Rankine-source free-surface panel method (truncate | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| rankine-wave-series60 | Rankine wave resistance + Series 60 empirical refe | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| tank-free-surface | Slack-tank free-surface correction to hydrostatics | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| wigley3-hullform | Wigley III hullform geometry (frequency-domain ver | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| crack-sif-mode-i | Centre-crack Mode-I stress intensity factor by dis | REAL | L3_VERIFIED | PASS | ✓ |  |  |
| acoustic-3d-pulse | Transient (pulse) acoustics through the frequency- | REAL | L4_CROSS_VALIDATED | PASS | ✓ | ✓ |  |
| hydrostatic-restoring-pressure-route | Hydrostatic restoring matrix by direct pressure in | REAL | L4_CROSS_VALIDATED | PASS | ✓ | ✓ |  |
| wave-flume-wavemaker | Linear wavemaker transfer functions (piston and fl | REAL | L4_CROSS_VALIDATED | PASS | ✓ | ✓ |  |
| ursell-radiation | Ursell submerged-cylinder radiation benchmark | REAL | L4_CROSS_VALIDATED | PASS | ✓ | ✓ |  |
| hulme-hemisphere | Hulme floating hemisphere (3-D radiation) | REAL | L4_CROSS_VALIDATED | PASS | ✓ | ✓ |  |
| near-field-quadrature | Near-boundary near-singular panel quadrature + hem | REAL | L4_CROSS_VALIDATED | PASS | ✓ | ✓ |  |
| dean-diffraction | Dean diffraction (zero-reflection theorem) | REAL | L4_CROSS_VALIDATED | PASS | ✓ | ✓ |  |
| finite-depth-2d | Finite-depth kernels + solver (2-D) | REAL | L4_CROSS_VALIDATED | PASS | ✓ | ✓ |  |
| michell-canonical-spectrum-panel-route | Michell wave resistance on the canonical spectrum  | REAL | L4_CROSS_VALIDATED | PASS | ✓ | ✓ |  |
| kutta-lifting | Kutta condition / lifting foil | REAL | L4_CROSS_VALIDATED | PASS | ✓ | ✓ |  |
| declared-numbers | Declared-number registry | REAL | L4_CROSS_VALIDATED | PASS | ✓ | ✓ |  |
| benchmark-cli-contract | Benchmark drivers expose a runnable entry point | REAL | L4_CROSS_VALIDATED | PASS | ✓ | ✓ |  |
| second-order-drift | Second-order mean wave drift force (near field / f | REAL | L4_CROSS_VALIDATED | PASS | ✓ | ✓ |  |
| coupled-memory-kernel-matrix | Coupled multi-DOF radiation memory kernel matrix | REAL | L4_CROSS_VALIDATED | PASS | ✓ | ✓ |  |
| hydroelastic-dry-modes | Hydroelastic panel modes and the wet/dry frequency | REAL | L4_CROSS_VALIDATED | PASS | ✓ | ✓ |  |
| ice-region-hydrodynamics | Ice-cover mass loading, ice-region RAO and ice-cla | REAL | L4_CROSS_VALIDATED | PASS | ✓ | ✓ |  |
| ice-mass-loading-bem-dispersion | Ice-cover mass-loading dispersion from a periodic- | REAL | L4_CROSS_VALIDATED | PASS | ✓ | ✓ |  |
| kcs-class-hull-anchors | Parametric KCS-class hull with published-particula | REAL | L4_CROSS_VALIDATED | PASS | ✓ | ✓ |  |
| diffraction-drift-maturity | Wigley III diffraction/RAO and cylinder drift matu | REAL | L4_CROSS_VALIDATED | PASS | ✓ | ✓ |  |
| cathodic-protection-bem | 3-D Laplace BEM cathodic protection (linear polari | REAL | L4_CROSS_VALIDATED | PASS | ✓ | ✓ |  |
| cylinder-and-farfield-drift | Exact MacCamy-Fuchs cylinder drift + 3-D far-field | REAL | L4_CROSS_VALIDATED | PASS | ✓ | ✓ |  |
| haskind-2d-excitation | 2-D Haskind excitation + net-source regularisation | REAL | L4_CROSS_VALIDATED | PASS | ✓ | ✓ |  |
| irregular-frequency-predictor-verified | Irregular-frequency predictor graded against an ex | REAL | L4_CROSS_VALIDATED | PASS | ✓ | ✓ |  |
| state-space-realization | State-space realisation of the Cummins radiation m | REAL | L4_CROSS_VALIDATED | PASS | ✓ | ✓ |  |
| stf-fullspeed-strip | STF (1970) strip theory with forward speed | REAL | L4_CROSS_VALIDATED | PASS | ✓ | ✓ |  |
| pekeris-waveguide | Pekeris two-layer shallow-water normal modes with  | REAL | L4_CROSS_VALIDATED | PASS | ✓ | ✓ |  |
| forward-speed-wave-resistance | Forward-speed wave resistance (Wigley, Michell/Koc | REAL | L5_PRODUCTION | PASS | ✓ | ✓ | ✓ |

## 5. Methodology

### Five-Level Maturity Scale

| Level | Name | Criteria |
|---|---|---|
| L0 | Not Supported | Capability not implemented or explicitly out of scope |
| L1 | Stub/Mock | Placeholder, hard-coded, or random output |
| L2 | Implemented | Code exists, basic tests pass, documentation anchor present |
| L3 | Verified | L2 + verified against analytical/reference solution with quantified error + convergence study |
| L4 | Cross-Validated | L3 + cross-validated via independent numerical route or external solver |
| L5 | Production | L4 + uncertainty quantification + experimental validation + full documentation chain |

### Gate Coverage

Gate coverage = passing_failable_gates / total_failable_gates.
L0 (NOT_SUPPORTED) capabilities are not failable — they are honestly out of scope.
A gate fails when the claimed maturity level is not supported by evidence.

### Cross-Consistency

Related capabilities are checked for maturity compatibility. Two capabilities
are consistent if their maturity levels differ by at most one step.
