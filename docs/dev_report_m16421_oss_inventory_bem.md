# OSS BEM 技术盘点 wave 1（FEAT-OSS-INVENTORY-BEM / m16421）

**锚点**：`FEAT-OSS-INVENTORY-BEM`（feature_ledger, planned → 本单认领）
**产出**：`docs/oss_inventory_bem/{registry.json, census.md, registry_data.py,
collect_oss_bem.py, fetch_readmes.py, build_registry.py, raw/}`
**校验**：`tests/test_oss_inventory_bem_registry.py`（7 项证据不变量，全绿）
**registry sha256[:16]**：见 `census.md` 抬头（表与数据由哈希互锁，禁止手改）

---

## ① 案例介绍（为什么做这件事）

TensorBEM 的对标基线里存在结构性隐患：**参考值是手抄常量**。例如
`src/tensorbem/benchmarks/wigley3_complete_verification.py` 内
`_WAMIT_A33_MASS_RATIO = 0.42`、`WIGLEY_REF = {...}` 等，都是人工从文献/WAMIT 手册
抄来的数字。这类常量无法审计、无法扩频点、无法跨几何，且直接违反「零手抄」证据纪律。

本单不新增物理算例，而是先把**外部技术面**盘清：谁在公开维护 BEM/势流求解器、
每个仓能给我们什么、哪些能力我们确实没有。产出用于驱动后续「验证欠账」类节点，
避免每一波都从头检索。

**范围与口径**：只做只读盘点 + 可再生脚本；不改动 `src/` 任何求解器代码。
（容量纪律：本机 load15 已 ≥100，故本单为轻量知识/工具工作，未起算力进程。）

## ② 怎么做的（方法与工具链）

三支脚本，全部提交、可复现：

1. `collect_oss_bem.py` —— 双通道发现：10 条关键词搜索（GitHub search API，
   `sort=stars`，限速 10/min，原始 payload 落盘 `raw/search_*.json`）+ 12 个领域
   种子仓直查（API 校验；404 者显式记录为排除，不臆造）。共 **82** 个候选仓。
2. `fetch_readmes.py` —— 对入围仓取 README 原文（`raw/readme/`，附 sha256[:16] 指纹），
   使「这个仓能干什么」可回溯到文本，而非我的记忆。
   注：本机到 `raw.githubusercontent.com` 的 TCP 被重置，改用 REST
   `repos/{r}/readme` 取同一份字节（已在脚本注释记录）。
3. `build_registry.py` —— 合并实时元数据（stars/license/pushed_at/archived）与人工判断
   （`registry_data.py` 的 tier/可迁移项/缺口映射），并**执行**缺口探针：
   对 `src/`+`tests/` 跑设计好的正则，把「本仓缺 X」变成可复核的命中数。
   校验不过（<20 仓、<6 锚点、悬空引用、量化目标无数字）则构建失败。

## ③ 计算结果（数字表）

| 指标 | 值 |
|---|---|
| 候选扫描仓 | **82**（搜索 10 轴 + 种子 12 个，其中 5 个 404 排除） |
| 登记并定级 | **22**（A=5，B=17） |
| 筛查剔除（C 档，逐条写理由） | **5** |
| README 指纹捕获 | **14** 仓 |
| GAP-BEM-* 锚点提议 | **8** |
| 证据不变量测试 | 7 项，全绿 |

A 档（可直接做对标基准/算法移植源）：

| 仓 | ★ | 许可证 | 迁给我们什么 |
|---|---|---|---|
| `capytaine/capytaine` | 235 | Apache-2.0 | Nemoh 谱系现代重写；独立对标实现 + CI 测试面 |
| `BEMRosetta/BEMRosetta` | 123 | GPL-3.0 | 权威系数格式全集（WAMIT/AQWA/Hydrostar/Nemoh/BEMIO）；跨程序对比流程 |
| `LHEEA/Nemoh` | 65 | Apache-2.0 | 该谱系原点；`Nemoh.cal`/`Mesh`/`tec` 事实标准（README 自述已迁 GitLab） |
| `JuliaOceanWaves/MarineHydro.jl` | 33 | MIT | 全可微 BEM；Wu 与 Delhommeau 双 Green 函数并列 |
| `PengYunqiang/CRESTU` | 1 | GPL-3.0 | 独立编写的 3D Rankine 源求解器（同族差异化对标体） |

GAP 锚点与执行探针（探针命中 = `src/`+`tests/` 正则命中文件数）：

| 锚点 | 优先级 | 探针命中 | 状态 | 量化验收 |
|---|---|---|---|---|
| `GAP-BEM-COEFF-IO` | P0 | **0** | verified_absent | I/O <1e-12；外部解析 <1e-6；现有验证劣化 <1% |
| `GAP-BEM-ADJOINT-GEOM-GRAD` | P1 | 3（均算子伴随语境） | 能力缺席 | 梯度 <1e-3；方向余弦 >0.999；成本 <3× 前向 |
| `GAP-BEM-FMM-HELMHOLTZ-KERNEL` | P1 | 12 | 部分具备（ka≤10） | ka≥25 仍 <1e-6 且加速 ≥2× |
| `GAP-BEM-CROSS-SOLVER-REGRESSION` | P1 | 2 | 部分具备 | 三体×8 频点 max<3%；CI <90 s |
| `GAP-BEM-ASSEMBLY-SCALING` | P2 | **0** | verified_absent | N=8192 墙钟 ≥3×；系数偏差 <1e-10；内存 ≤+10% |
| `GAP-BEM-TD-MOORING-COUPLING` | P2 | 2（硬编码常量） | 部分具备 | RAO <5%；自由体极限 <1e-6 |
| `GAP-BEM-GREEN-FAMILY-CROSSCHECK` | P2 | 4（一行文献并列） | 部分具备 | 20 点偏差 <1e-4 |
| `GAP-BEM-REGFILE-ROUNDTRIP` | P3 | 0 | verified_absent | 几何自检 <1e-8；tec 列比对 <1e-6 |

## ④ 结果图

**本单无图，理由**：任务是只读技术盘点与工具构建，未产生任何场量/系数场计算，
不存在可绘制的物理结果；唯一的数值量是探针命中计数与仓元数据，已在 ③ 以表呈现。
为避免用装饰性图冒充证据，不生成图（下一步 `GAP-BEM-CROSS-SOLVER-REGRESSION`
落地后自然产出「逐频点误差曲线」图，那才是有信息量的图）。

## ⑤ 注意点（踩过的坑与局限）

1. **探针推翻了我自己的两条断言**（诚实修正，已写入 registry）：
   - 原写「FMM 的 Helmholtz 核 NOT_SUPPORTED → 完全没有 Helmholtz 快速算法」是**错的**：
     `fmm.py` 有 `fmm_helmholtz_matvec`，`helmholtz_supported(ka, ka_max=10.0)` 为
     fail-closed，即 **ka≤10 走 FMM，ka>10 静默退回直接求和**。锚点已重述为
     「高 ka 传递算子缺口」，验收要求把 ka>10 的回退档显式成表。
   - 原写「完全无系泊」也不准确：`benchmarks/qtf_drift_study.py:72` 有 OC4 系泊
     `stiffness: 1.0e5 N/m` 的 catenary 常量。锚点已重述为
     「只有常数项、无系泊动力学/多体耦合求解面」。
2. **`aca`/`hierarch` 这类短词会假命中**（如 `fmm.py` 的八叉树被 `hierarch` 命中）。
   已改成词边界正则 `\bACA\b`/`\bTLD\b`；`hierarch` 移出，避免把八叉树当成装配分块。
3. **两个短名单仓因取不到实时元数据被剔除**（
   `Xu-Duan/Potential-Flow-With-Complex-Green-Functions`、
   `LukeMcCulloch/adjoint-3D-linear-wave-resistance`：API 侧 404/限流）。
   按「无实时元数据不定级」的纪律直接移出登记表，宁可少 2 个也不提交未验证记录；
   代价：`GAP-BEM-ADJOINT-GEOM-GRAD` 目前只由 MarineHydro.jl 单源支撑。
4. **许可证风险**：BEMRosetta 与 Nemoh 镜像均涉 GPL 家族/迁移状态，本仓只能读
   格式规范与设计，**不得复制代码**；registry 每条都写了 `license_note`。
5. **C 档不等于无用**：`scuff-em` 的分层介质 Green 函数、`bempp-cl` 的高阶奇异积分
   仍可读，但物理不同源，不进对标清单。
6. 8 个锚点只是**提议**，未写入 `feature_ledger`（不越权改平台账本）；
   采纳与否由仲裁/机器门决定。

## ⑥ 误差判定（是否 <3%）

本单**不产生物理量**，故无「计算结果 vs 基准」误差可判。可判定的是**证据质量**，
逐项如实报告：

| 项 | 判据 | 实测 | 判定 |
|---|---|---|---|
| 仓元数据准确性 | 每条记录须有实时 API 字段 | 22/22 有 `pushed_at`+`license` 字段 | **通过** |
| README 取证覆盖 | 引用原文的仓须有指纹 | 引用原文者 14/14 有 sha256 指纹；其余显式标 `description_only` | **通过** |
| 缺口声称可复核 | 每个锚点带执行探针 | 8/8 有 `gap_probe`（含扫描文件数、逐词命中样本） | **通过** |
| 锚点验收量化 | 须含数字判据 | 8/8 `quantified_target` 含数字，`acceptance` ≥3 条 | **通过** |
| 表↔数据一致性 | census.md 与 registry.json 互锁 | 哈希互锁测试通过 | **通过** |

**局限**：本单判的是「外部技术面与我们缺什么」，不是「我们的数值有多准」；
真正的精度复核需要 `GAP-BEM-CROSS-SOLVER-REGRESSION` 落地后按逐频点误差表裁定 <3%。
---

**平台报告 id**：15492（`scenario_key=oss-inventory-bem`, mission 16421）
**校验命令**：`PYTHONPATH=src python -m pytest tests/test_oss_inventory_bem_registry.py -q` → 7 passed
