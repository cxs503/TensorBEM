# TensorFVM 缺口锚点提议（GAP-FVM-*，本波 8 条）

> 程序化生成，勿手改。源：`docs/oss-inventory-fvm/registry.json` 的 `gaps` 段（生成时间 2026-11-11T12:23:26+00:00）。
> 每条的「现状证据」都必须是本仓可 grep 到的路径；「验收判据」必须是机器可裁的数值门；「外部锚」必须落在文献/解析解，不得用本仓自己的实现自证。

## 排序总表

| # | 锚点 | 优先级 | 参考实现（Tier） | 工作量 | 判据要点 |
|---|---|---|---|---|---|
| 1 | `GAP-FVM-AMR-QUADTREE` | P0 | AMReX(T1), Basilisk(T1), Gerris(T2) | L | Sod on a 2-level AMR hierarchy (effective 400 cells) must match the uniform 400-cell solution in L1(rho) within 5% and conserve mass to 1e-12 via the flux-register ledger |
| 2 | `GAP-FVM-IMPLICIT-DUAL-TIME` | P0 | SU2(T1), CFL3D(T2), OpenFOAM (Foundation, OpenFOAM-dev)(T1) | L | on the same Sod setup restarted to a steady channel case, implicit CFL>=10 must reach the explicit converged solution within 1e-6 (L2, rho) using >=10x fewer time steps |
| 3 | `GAP-FVM-LOWMACH-PRECOND` | P0 | SU2(T1), PeleLMeX(T2), Code_Saturne(T1) | M | low-Mach Gresho vortex (Ma=0.05) must preserve the analytic pressure/density field within 2% and reduce the acoustic iteration count by >=5x vs unpreconditioned at the same CFL |
| 4 | `GAP-FVM-MRF-SLIDING-AMI` | P1 | OpenFOAM (Foundation, OpenFOAM-dev)(T1), OpenFOAM (ESI / OpenCFD distribution)(T1) | M | 2D Taylor-Couette: torque on the inner cylinder from the AMI/MRF path vs the analytic Couette torque within 3% at two gap resolutions |
| 5 | `GAP-FVM-MPI-DECOMP` | P1 | AMReX(T1), OpenFOAM (Foundation, OpenFOAM-dev)(T1), Code_Saturne(T1) | L | lid-cavity Re=1000 split over 2 and 4 ranks must reproduce the serial Ghia-u-centreline within 1e-6 L2 and keep parallel efficiency >=0.7 at 4 ranks on the cluster |
| 6 | `GAP-FVM-AMG-KRYLOV` | P1 | PETSc (incl. hypre bindings)(T1), hypre(T1), SU2(T1) | M | 256^2 Poisson and an advection-diffusion system: AMG-PCG iteration count must grow <=1.3x from 64^2 to 256^2 and solve to 1e-10 in fewer than 30 iterations |
| 7 | `GAP-FVM-COMBUSTION-SPECIES` | P2 | PeleC(T2), PeleLMeX(T2), OpenFOAM (Foundation, OpenFOAM-dev)(T1) | L | 1D premixed CH4/air flame: laminar flame speed within 10% of the Cantera reference at two resolutions |
| 8 | `GAP-FVM-RADIATION-P1` | P2 | Truchas(T2), OpenFOAM (Foundation, OpenFOAM-dev)(T1) | M | 1D grey parallel plates: net radiative flux vs the analytic exchange factor within 2% |

## 逐条详述

### `GAP-FVM-AMR-QUADTREE` — Adaptive mesh refinement with conservative coarse-fine flux refluxing

- **优先级**：P0；**工作量**：L
- **现状（TensorFVM）**：no quadtree/octree or hanging-node machinery: uniform grids only, refinement studies are done by re-running finer uniform meshes
- **现状证据**：grep -riE 'quadtree|octree|hanging|coarsen' over src/tensorfvm returns no AMR data structure, only prose in docstrings
- **证据文件**：`src/tensorfvm/mesh3d.py`, `src/tensorfvm/native_mesh3d.py`
- **参考实现**：AMReX[T1, 831★ → 吸: refluxing (flux register) conservation across coarse-fine interfaces + regridding/clustering strategy], Basilisk[T1 → 吸: wavelet-based refinement criterion + VOF flux conservation across level boundaries], Gerris[T2, 1★ → 吸: GCI/Gerris adaptive VOF test-suite as acceptance anchors], Clawpack (incl. AMRClaw, PyClaw)[T2, 210★ → 吸: wave-propagation AMR flux registers + exact-solution test suite], PeleC[T2 → 吸: operator-split stiff chemistry integration contract]
- **验收判据（机器可裁）**：Sod on a 2-level AMR hierarchy (effective 400 cells) must match the uniform 400-cell solution in L1(rho) within 5% and conserve mass to 1e-12 via the flux-register ledger; cell count must drop by >=3x vs uniform 400
- **外部锚**：Sod 1978 exact Riemann solution (same anchor as the uniform-grid leg) + AMReX/Basilisk refluxing formulation
-- 许可提醒：BSD-3-Clause (compatible)；GPL-2.0 (read-only)；GPL-3.0 (read-only)

### `GAP-FVM-IMPLICIT-DUAL-TIME` — Implicit steady / dual-time integration (LU-SGS or Newton-Krylov)

- **优先级**：P0；**工作量**：L
- **现状（TensorFVM）**：compressible solver is explicit forward-Euler with CFL ~0.4; steady states are reached by pseudo-time marching, so high-Re and low-Mach cases are step-limited
- **现状证据**：EulerSolver advances with explicit time stepping (cfl=0.4 default in cases.sod); grep -niE 'implicit|newton|lusgs' in src/tensorfvm/compressible hits only the exact-Riemann Newton iteration
- **证据文件**：`src/tensorfvm/compressible/solver.py`, `src/tensorfvm/compressible/cases.py`
- **参考实现**：SU2[T1, 1856★ → 吸: Weiss-Smith/Turkel preconditioning matrix + LU-SGS linearisation + discrete adjoint boundary conditions], CFL3D[T2, 350★ → 吸: LU-SGS sweep ordering + CFL ramping schedule], OpenFOAM (Foundation, OpenFOAM-dev)[T1, 2302★ → 吸: MULES algebraic VOF limiter + AMI/GGI sliding interface topology + dynamicRefineFvMesh hanging-node flux correction], foam-extend[T2, 66★ → 吸: block-coupled implicit assembly pattern]
- **验收判据（机器可裁）**：on the same Sod setup restarted to a steady channel case, implicit CFL>=10 must reach the explicit converged solution within 1e-6 (L2, rho) using >=10x fewer time steps; a regression test pins the step-count ratio
- **外部锚**：Jameson & Turkel (1981) LU-SGS implicit formulation (as implemented in SU2/CFL3D) + the existing explicit converged state as the comparison state
-- 许可提醒：GPL-3.0 (read-only)；GPL-3.0: read-only, re-implement from the papers, do not copy code；LGPL-2.1 (read-only; re-implementation safest)；NASA open source agreement (permissive-ish, read-only)

### `GAP-FVM-LOWMACH-PRECOND` — Low-Mach preconditioning (Weiss-Smith / Turkel) for the compressible solver

- **优先级**：P0；**工作量**：M
- **现状（TensorFVM）**：no preconditioning matrix: acoustic stiffness at Ma<0.1 makes the explicit compressible path unusable for ship-scale flows, so those cases fall back to the incompressible solver
- **现状证据**：no low-Mach/preconditioning term anywhere in src/tensorfvm/compressible (grep 'precondition' hits only unrelated iterative-solver prose elsewhere)
- **证据文件**：`src/tensorfvm/compressible/solver.py`, `src/tensorfvm/compressible/riemann.py`
- **参考实现**：SU2[T1, 1856★ → 吸: Weiss-Smith/Turkel preconditioning matrix + LU-SGS linearisation + discrete adjoint boundary conditions], PeleLMeX[T2 → 吸: low-Mach pressure-correction loop with spectral deferred correction for chemistry], Code_Saturne[T1, 312★ → 吸: gradient/limiter treatment on highly non-orthogonal polyhedra]
- **验收判据（机器可裁）**：low-Mach Gresho vortex (Ma=0.05) must preserve the analytic pressure/density field within 2% and reduce the acoustic iteration count by >=5x vs unpreconditioned at the same CFL
- **外部锚**：Gresho & Chan 1990 vortex (analytic, incompressible limit) + SU2 preconditioning formulation
-- 许可提醒：BSD-3-Clause (compatible)；GPL-2.0 (read-only)；LGPL-2.1 (read-only; re-implementation safest)

### `GAP-FVM-MRF-SLIDING-AMI` — Rotating-frame (MRF) and arbitrary-mesh-interface (AMI) sliding boundaries

- **优先级**：P1；**工作量**：M
- **现状（TensorFVM）**：no MRF source term and no sliding/AMI interface: rotating parts (propellers, rotating cylinders) cannot be represented except by re-meshing
- **现状证据**：grep -riE 'sliding|MRF' over src/tensorfvm hits only the 'sliding ice' wall comment in ice_channel.py
- **证据文件**：`src/tensorfvm/moving_mesh.py`, `src/tensorfvm/six_dof.py`
- **参考实现**：OpenFOAM (Foundation, OpenFOAM-dev)[T1, 2302★ → 吸: MULES algebraic VOF limiter + AMI/GGI sliding interface topology + dynamicRefineFvMesh hanging-node flux correction], OpenFOAM (ESI / OpenCFD distribution)[T1 → 吸: overset interpolation weights + compressibleInterFoam pressure equation with phase-change source]
- **验收判据（机器可裁）**：2D Taylor-Couette: torque on the inner cylinder from the AMI/MRF path vs the analytic Couette torque within 3% at two gap resolutions
- **外部锚**：analytic Taylor-Couette torque (Couette 1890; Batchelor 1967, §4.9): T = 4*pi*mu*L*omega*ri^2*ro^2/(ro^2-ri^2)
-- 许可提醒：GPL-3.0 (read-only)；GPL-3.0: read-only, re-implement from the papers, do not copy code

### `GAP-FVM-MPI-DECOMP` — Distributed-memory domain decomposition with halo exchange

- **优先级**：P1；**工作量**：L
- **现状（TensorFVM）**：solves are single-process; torch.distributed is only consulted to pick a device, there is no domain decomposition or halo exchange for the FVM fields
- **现状证据**：runtime.py queries torch.distributed.is_initialized() for device selection only; no halo/exchange code in src/tensorfvm (grep 'halo' hits nothing)
- **证据文件**：`src/tensorfvm/runtime.py`
- **参考实现**：AMReX[T1, 831★ → 吸: refluxing (flux register) conservation across coarse-fine interfaces + regridding/clustering strategy], OpenFOAM (Foundation, OpenFOAM-dev)[T1, 2302★ → 吸: MULES algebraic VOF limiter + AMI/GGI sliding interface topology + dynamicRefineFvMesh hanging-node flux correction], Code_Saturne[T1, 312★ → 吸: gradient/limiter treatment on highly non-orthogonal polyhedra], preCICE[T2, 992★ → 吸: conservative interpolation (nearest-projection) + implicit coupling iteration]
- **验收判据（机器可裁）**：lid-cavity Re=1000 split over 2 and 4 ranks must reproduce the serial Ghia-u-centreline within 1e-6 L2 and keep parallel efficiency >=0.7 at 4 ranks on the cluster
- **外部锚**：Ghia et al. 1982 centreline data + serial TensorFVM run
-- 许可提醒：BSD-3-Clause (compatible)；GPL-2.0 (read-only)；GPL-3.0: read-only, re-implement from the papers, do not copy code；LGPL-3.0 (read-only)

### `GAP-FVM-AMG-KRYLOV` — Algebraic multigrid + Krylov linear solver backend

- **优先级**：P1；**工作量**：M
- **现状（TensorFVM）**：the only multigrid is the geometric V-cycle inside the immersed masked-Poisson solver; no general AMG/Krylov for the unsymmetric pressure/momentum systems
- **现状证据**：grep -riE 'multigrid|vcycle|amg|petsc|hypre' over src/tensorfvm hits only immersed/projection.py (masked geometric V-cycle) and docstrings
- **证据文件**：`src/tensorfvm/immersed/projection.py`
- **参考实现**：PETSc (incl. hypre bindings)[T1, 550★ → 吸: GMRES/BiCGStab + BoomerAMG via a thin Python binding], hypre[T1, 876★ → 吸: boomeramg parameter set validated on our Poisson cases], SU2[T1, 1856★ → 吸: Weiss-Smith/Turkel preconditioning matrix + LU-SGS linearisation + discrete adjoint boundary conditions], Code_Saturne[T1, 312★ → 吸: gradient/limiter treatment on highly non-orthogonal polyhedra], Chombo[T2 → 吸: level-by-level multigrid restriction/prolongation on AMR hierarchies]
- **验收判据（机器可裁）**：256^2 Poisson and an advection-diffusion system: AMG-PCG iteration count must grow <=1.3x from 64^2 to 256^2 and solve to 1e-10 in fewer than 30 iterations
- **外部锚**：Brandt (1977) multigrid h-independence + BoomerAMG convergence data (Baker et al. 2012)
-- 许可提醒：Apache-2.0/MIT (compatible)；BSD-2-Clause (compatible)；BSD-3-Clause (compatible)；GPL-2.0 (read-only)；LGPL-2.1 (read-only; re-implementation safest)

### `GAP-FVM-COMBUSTION-SPECIES` — Multi-species transport with stiff Arrhenius chemistry

- **优先级**：P2；**工作量**：L
- **现状（TensorFVM）**：the compressible solver carries at most two materials with a stiffened-gas EOS; there is no species transport, no reaction source and no operator splitting for stiff chemistry
- **现状证据**：grep -riE 'combust|chemistry|species' hits only the two-material 'species' bookkeeping in the compressible package
- **证据文件**：`src/tensorfvm/compressible/eos.py`, `src/tensorfvm/compressible/solver.py`
- **参考实现**：PeleC[T2 → 吸: operator-split stiff chemistry integration contract], PeleLMeX[T2 → 吸: low-Mach pressure-correction loop with spectral deferred correction for chemistry], OpenFOAM (Foundation, OpenFOAM-dev)[T1, 2302★ → 吸: MULES algebraic VOF limiter + AMI/GGI sliding interface topology + dynamicRefineFvMesh hanging-node flux correction]
- **验收判据（机器可裁）**：1D premixed CH4/air flame: laminar flame speed within 10% of the Cantera reference at two resolutions
- **外部锚**：Cantera (Goodwin et al. 2017) laminar flame speed for the same mechanism
-- 许可提醒：BSD-3-Clause (compatible)；GPL-3.0: read-only, re-implement from the papers, do not copy code

### `GAP-FVM-RADIATION-P1` — Radiative heat transfer (P1 / discrete ordinates) coupled to the energy equation

- **优先级**：P2；**工作量**：M
- **现状（TensorFVM）**：no radiation model: the energy equation has conduction and convection only
- **现状证据**：no radiation module; docs/verification-gap-audit-m7555.md already lists radiation as an unverified capability gap
- **证据文件**：`src/tensorfvm/thermal/__init__.py`, `src/tensorfvm/cht`
- **参考实现**：Truchas[T2, 11★ → 吸: P1/enclosure radiation source-term coupling into energy equation], OpenFOAM (Foundation, OpenFOAM-dev)[T1, 2302★ → 吸: MULES algebraic VOF limiter + AMI/GGI sliding interface topology + dynamicRefineFvMesh hanging-node flux correction]
- **验收判据（机器可裁）**：1D grey parallel plates: net radiative flux vs the analytic exchange factor within 2%; coupling conserves total energy to 1e-10
- **外部锚**：analytic view-factor exchange for infinite parallel plates (Incropera & DeWitt 2007, Ch. 13)
-- 许可提醒：BSD-3-Clause (verify)；GPL-3.0: read-only, re-implement from the papers, do not copy code

## 两条 P0 的落地顺序建议

1. **先 `GAP-FVM-LOWMACH-PRECOND` 后 `GAP-FVM-IMPLICIT-DUAL-TIME`**：低马赫预条件会改变方程组的谱，隐式线性化必须在它之后标定，否则 LU-SGS 的 CFL 上界要重做一次。
2. `GAP-FVM-AMG-KRYLOV` 与上面两条 P0 共享验收载体（同一个低马赫算例的线性求解次数），可以并行立项但共用一套诊断。
3. `GAP-FVM-AMR-QUADTREE` 的验收刻意复用本波的 Sod 外部锚（`artifacts/sod_literature_benchmark.json`），AMR 版与均匀 400 格解比对，避免再引入一个未验证的基准。

## 与既有账本的关系

- `docs/verification-gap-audit-m7555.md` 已把「燃烧/组分输运」「辐射换热」列为验证缺口，本波把二者升级为**能力缺口**并补上外部锚与参考实现。
- 平台 campaign_nodes 已有多条 FVM 两相/VOF/热对流节点（本波检索命中），故 VOF 两相**不**列为缺口：TensorFVM 侧已有 PLIC/CICSAM 与三条验证面。
- ALE/6DOF 与 overset 在本仓已有实现（`moving_mesh.py`/`six_dof.py`/`overset*.py`），本波不重复立项。

