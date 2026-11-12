```text
===============================================================================
=                                                                             =
=            CCCC   RRRR    EEEEE   SSSSS  TTTTT  U   U                       =
=           C       R   R   E       S        T    U   U                       =
=           C       RRRR    EEEE    SSSSS    T    U   U                       =
=           C       R  R    E           S    T    U   U                       =
=            CCCC   R   R   EEEEE   SSSSS    T     UUU                        =
=                                                                             =
=                            C R E S T U - 1 F                                =
=     (Computational Rankine-source Engine for Sea Technology, SJTU)          =
=                 3D Rankine Source BEM Hydrodynamic Solver                   =
=                                                                             =
=            Project Name : CRESTU-1F                                         =
=            Lead Authors : Yunqiang Peng, Zhentao Jiang                      =
=            Affiliation  : Shanghai Jiao Tong University                     =
=            Status       : Research Software — Active Development            =
=            Start Date   : Aug. 24, 2026                                     =
=                                                                             =
===============================================================================
```

# CRESTU-1F

> 中文说明在前，English documentation follows below.

## 中文说明

### 项目简介

CRESTU-1F 是一个基于 MATLAB 的三维 Rankine-source 边界元（BEM）水动力学研究软件。当前 release 面向科研计算、数值实验和教学使用，提供从几何与边界生成到频域势流求解、水动力系数、波浪激励和 RAO 的完整一阶工作流。

项目状态：**科研软件，持续开发中**。核心一阶流程已经可以通过 fresh clone、portable setup、canonical smoke case 和代表性 production solve；更高阶载荷、广泛几何族和完整工程级物理验证仍需独立评估。

### 主要功能

- 三维 Rankine-source BEM 与 Hess–Smith 面元计算；
- 单体/多体频域辐射和绕射；
- 自由面、海底和外边界处理；
- 附加质量、pressure-integrated radiation damping、波浪激励、静水恢复和 RAO；
- `ISX/ISY` 对称性支持；
- WAMIT reference 读取与比较工具；
- 可移植 MATLAB setup、canonical example 和 validation smoke。

### 核心约定

- 单位：SI；
- 时谐因子：`exp(+i*omega*t)`；
- 物面法向 `n_B`：从物体指向流体；
- 每个浮体的 DOF 顺序：Surge, Sway, Heave, Roll, Pitch, Yaw；
- radiation potential：单位广义位移势；
- production damping：pressure-integrated 路径；Haskind 和 control-cylinder flux 是诊断路径；
- 一阶公式、符号和坐标约定以 [FIRST_ORDER_CONVENTIONS.md](FIRST_ORDER_CONVENTIONS.md) 为准。

### 快速开始

要求：

- MATLAB R2022b 或更新版本；维护环境为 MATLAB R2023b；
- 基本 smoke case 不需要 WAMIT 或外部数据；
- WAMIT 只用于可选 reference comparison。

在 MATLAB 中从仓库根目录运行：

```matlab
setup_crestu
report = run_validation_smoke;
results = run_example_single_sphere;
```

`run_example_single_sphere` 默认运行一个频率的 canonical smoke case，完成真实 production assembly、cache 和结果输出。生成文件位于 `examples/single_sphere/`，默认不会进入 Git。

完整 16 频率 Fine 配置：

```matlab
setup_crestu
results = run_example_single_sphere('FULL');
```

FULL 模式明显比 smoke case 更耗时。

### 仓库结构

```text
Source Code/                 production MATLAB source
Case_Wave/                   维护中的频域 runtime 文件
Case_UniformFlow/            定常均匀流 benchmark
examples/single_sphere/      canonical single-sphere 输入和 smoke case
validation/                  validation guide、runner 和 fixture 说明
docs/                        结构、开发约定、已知问题和历史文档
scripts/                     maintenance helpers
AGENTS.md                    项目长期维护规则
setup_crestu.m               portable path setup
run_example_single_sphere.m  新用户入口
run_validation_smoke.m       快速 setup/geometry 检查
```

### 验证

从 [validation/VALIDATION_GUIDE.md](validation/VALIDATION_GUIDE.md) 开始。当前 release 已完成：

- fresh clone 检查；
- `setup_crestu`；
- canonical geometry smoke；
- 单频 production solver smoke。

完整 WAMIT research archive 不随 Git 仓库分发。需要 reference comparison 时，请按照 validation guide 提供外部数据目录，并记录 geometry、units、frequency conversion、reference role 和 provenance。

### 开发说明

修改 source 前先阅读 [AGENTS.md](AGENTS.md) 和 [FIRST_ORDER_CONVENTIONS.md](FIRST_ORDER_CONVENTIONS.md)。

- 新的轻量、可复现输入放在 `examples/`；
- validation runner 和小型 fixture 放在 `validation/`；
- generated MAT、cache、logs、solver output 留在 ignored output/cache 目录；
- 物理或数值修改应保留 complex force/RAO phase、source/config identity 和明确 validation status；
- 历史研究材料和大型数据集独立维护，不作为 fresh clone 的隐式依赖。

### 学术使用与许可

CRESTU-1F 是科研软件。工程应用请结合实验或权威 benchmark 做独立校核。许可信息见 [LICENSE](LICENSE)。

---

## English

### Overview

CRESTU-1F is a MATLAB research code for three-dimensional Rankine-source boundary-element hydrodynamics. The current release provides a complete first-order frequency-domain workflow from geometry and boundary generation through radiation/diffraction, hydrodynamic coefficients, wave excitation, hydrostatics, and RAOs.

It is **research software under active development**. The core first-order workflow is cloneable, portable, and exercised by a canonical MATLAB smoke case. Higher-order loads, broad geometry families, and engineering-level physical validation remain separate research topics.

### Features

- 3D Rankine-source BEM and Hess–Smith panel calculations;
- single- and multi-body frequency-domain radiation/diffraction;
- free-surface, seabed, and outer-boundary treatment;
- added mass, pressure-integrated radiation damping, wave excitation, hydrostatics, and RAOs;
- `ISX/ISY` symmetry support;
- WAMIT reference readers and comparison utilities;
- portable MATLAB setup, canonical example, and validation smoke runner.

### Quick Start

Requirements:

- MATLAB R2022b or newer; the maintained environment uses MATLAB R2023b;
- no external dataset is required for the basic smoke case;
- WAMIT is optional and only required for reference comparisons.

From the repository root in MATLAB:

```matlab
setup_crestu
report = run_validation_smoke;
results = run_example_single_sphere;
```

The default example runs one frequency through the production solver and writes generated meshes, cache, audit data, and results under `examples/single_sphere/`. These generated files are ignored by Git.

For the full 16-frequency Fine configuration:

```matlab
setup_crestu
results = run_example_single_sphere('FULL');
```

### Repository Structure

```text
Source Code/                 production numerical source
Case_Wave/                   maintained frequency-domain runtime files
Case_UniformFlow/            steady uniform-flow benchmark
examples/single_sphere/      canonical input and smoke case
validation/                  validation guidance and fixtures
docs/                        project structure and development notes
scripts/                     maintenance helpers
AGENTS.md                    project rules
setup_crestu.m               portable path setup
run_example_single_sphere.m  beginner-facing entry point
run_validation_smoke.m       fast setup/geometry check
```

### Conventions and Validation

Read [AGENTS.md](AGENTS.md) and [FIRST_ORDER_CONVENTIONS.md](FIRST_ORDER_CONVENTIONS.md) before modifying numerical code. The release uses SI units, `exp(+i*omega*t)`, body-outward normals, local six-DOF ordering, unit-displacement radiation, and pressure-integrated production damping.

Use [validation/VALIDATION_GUIDE.md](validation/VALIDATION_GUIDE.md) for smoke, full-example, and optional WAMIT reference workflows. Full WAMIT research data and large generated results are maintained separately from the Git release repository.

### Development

Keep new examples small and reproducible. Put validation runners under `validation/`, keep generated output out of tracked source, and record source/config/reference provenance for numerical changes. Historical research artifacts are maintained separately from the release repository.

### License

See [LICENSE](LICENSE).
