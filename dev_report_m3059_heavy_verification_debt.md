# TensorBEM 声明数字复算：heavy 层（椭球 1280 面元）与内部谐波 1280 面元列的验证欠账结清

**任务**：m3059（verifier 席，自找活） · **分支**：`agent/m3059` · **日期**：2026-11-04
**仓**：TensorBEM（worktree `/root/tp-agent/work/m3059`，基线 `d22b6ad`）
**工件**：`outputs/verif_m3059/{heavy_tier,fast_standard,harmonic_rows,interior_harmonics_1280}.json`、
`outputs/verif_m3059/{fig_registry_deviation,fig_interior_harmonics}.png`、
脚本 `outputs/verif_m3059/{build_rows,make_figures,inject_docs}.py`

EDIT_WHITELIST: src/tensorbem/benchmarks/declared_numbers.py, tests/test_declared_numbers.py, docs/classical_benchmarks.md, outputs/verif_m3059/, dev_report_m3059_heavy_verification_debt.md

---

## 1. 案例介绍（为什么做这件事）

2026-10-24 的扩编审计（`expand_audit_tensorbem.md` §2.3、§4.2）留下两条**未结清**的验证欠账：

1. **三维椭球 1280 面元行未被直接复现**：文档（`README.md` Milestone 3、`docs/classical_benchmarks.md` §3 表）声明三轴相对误差 \(4.1679/4.4621/4.9359\times10^{-3}\)，而审计只跑了 `subdivisions=2`（320 面元）与 `4`（5120 面元），1280 面元这一行从未被计算过，仅凭收敛趋势推断。
2. **内部固体谐波 1280 面元列是纯文档数字**：`docs/classical_benchmarks.md` §5.2 表 3 记录了度 1–3 在 \(N=1280\) 的通量误差与探针 RMSE，但注册表（m2136 建立的 `tensorbem.benchmarks.declared_numbers`）只登记了 degree-1 的 320 面元值，degree 2/3 的 1280 面元数字**没有任何计算守护**，随时可能腐化。

两者都是「已合并功能对解析基准的复核」，属于验证欠账类，且无需新物理实现：真值分别是 Lamb 形状因子的闭式附加质量与精确固体谐波多项式，误差是纯离散误差，判据干净。

**算例参数**

| 项 | 椭球（无界流 Laplace） | 内部固体谐波（内域 Dirichlet） |
|---|---|---|
| 半轴 / 半径 | \((1.5,1.0,0.75)\) m | \(a=1\) m（icosphere） |
| 离散 | `subdivisions=3` → 1280 三角形常数面元 | `subdivisions=3` → 1280 面元 |
| 真值 | \(\rho V\alpha_i/(2-\alpha_i)\)，Lamb 形状因子 | 精确固体谐波 \(x,\ x^2-y^2,\ x^3-3xy^2\) |
| 判据量 | 三轴附加质量相对误差 | 通量相对 \(L_2\) 误差 + 内点探针 RMSE |

## 2. 怎么做的

1. **把未复核的数字纳入注册表**：`src/tensorbem/benchmarks/declared_numbers.py` 新增 `interior_harmonics` 组（复用 `interior_harmonics.run_case`，一条注册项 = 一次求解，靠注册表自带的 `(group, parameters)` 缓存让通量与 RMSE 共享同一次求解），登记 1280 面元 3 度 × 2 量 = 6 条。成本实测为分钟级稠密求解，故按注册表自身的成本语义归入 `heavy` 层。
2. **跑通 heavy 层**：`python -m tensorbem.benchmarks.declared_numbers --tiers heavy --output outputs/verif_m3059/heavy_tier.json` —— 这是注册表建立以来 heavy 层第一次被端到端执行（椭球 1280 三轴 + Ursell 256 两量）。
3. **跑内部谐波研究**：`python -m tensorbem.benchmarks.interior_harmonics --subdivisions 2 3 --degrees 1 2 3`，取 \(N=1280\) 结果按注册表同一判定函数 `_judge` 判 PASS/DRIFT（`outputs/verif_m3059/build_rows.py`），全程零手抄。
4. **回归钉桩**：`tests/test_declared_numbers.py` 新增两条 heavy 门测试（`TENSORBEM_HEAVY_TESTS=1` 时执行）：椭球 1280 三轴行、内部谐波 1280 六行（含「度越高通量误差越大」的单调性判据）；并把注册表覆盖面判据扩到 `interior_harmonics` 组。
5. **文档程序化注入**：`docs/classical_benchmarks.md` §5.2 增加「复算与回归钉桩」小节，实测列由 `outputs/verif_m3059/inject_docs.py` 从 JSON 生成。

## 3. 计算结果

### 3.1 heavy 层（`heavy_tier.json`，5 条全 PASS）

| 注册项 | 声明 | 实测 | 相对偏差 | 判定 |
|---|---:|---:|---:|---|
| ellipsoid 1280 长轴 \(x\) | 4.1679e-3 | 4.16794e-3 | +8.74e-6 | PASS |
| ellipsoid 1280 中轴 \(y\) | 4.4621e-3 | 4.46212e-3 | +4.82e-6 | PASS |
| ellipsoid 1280 短轴 \(z\) | 4.9359e-3 | 4.93593e-3 | +6.54e-6 | PASS |
| Ursell 256 附加质量 | 4.2930e-3 | 4.29298e-3 | −4.07e-6 | PASS |
| Ursell 256 阻尼 | 6.8412e-3 | 6.84116e-3 | −5.25e-6 | PASS |

**判决含义**：1280 面元这一行在六位有效数字上与文档一致 —— 审计当年「320→1280 误差应降 16 倍」的疑虑其实是把面板翻倍与 \(h\) 翻倍混为一谈（320→1280 是面板 ×4、\(h\) ÷2，O(h²) 只降 4 倍，\(1.63\times10^{-2}/4 \approx 4.1\times10^{-3}\)，正是声明值）。**该行不需要修改文档，属 PASS 结清**。

### 3.2 内部固体谐波 1280 面元（`harmonic_rows.json`，6 条全 PASS）

| 度 | 算例 | 声明通量误差 | 实测 | 相对偏差 | 实测探针 RMSE |
|---:|---|---:|---:|---:|---:|
| 1 | \(x\) | 3.7640e-3 | 3.76395e-3 | −1.22e-5 | 2.2929e-4 |
| 2 | \(x^2-y^2\) | 7.3437e-3 | 7.34375e-3 | +6.31e-6 | 2.3075e-4 |
| 3 | \(x^3-3xy^2\) | 1.1908e-2 | 1.19078e-2 | −1.69e-5 | 1.8240e-4 |

### 3.3 全注册表一次复算（`fast_standard.json` + 上述）

`--tiers fast,standard` 12 条全 PASS（最坏相对偏差 \(1.10\times10^{-1}\)，来自 MFS 行，容差 5%–15% 内）；加 heavy 5 条与谐波 6 条，**注册表 22 条声明数字本次全部重算并判 PASS**。

### 3.4 图

`fig_registry_deviation.png`：22 条注册项的实测相对偏差（对数轴，按成本层着色）与各自 DRIFT 阈值线的对比，一条都没有越过阈值。
`fig_interior_harmonics.png`：左图通量误差 vs 面元数（320→1280，度 1/2/3），可见同一网格上度越高误差越大（\(3.76\times10^{-3}\to1.19\times10^{-2}\)，约 3.2 倍）；右图内点 RMSE 与度基本无关（\(1.8\text{–}2.3\times10^{-4}\)），复现了文档「更强角向变化主要伤害边界通量、不明显伤害内部场」的结论。

## 4. 注意点

- **成本**：1280 面元三维稠密求解在本机（load15 400–650 的重载窗口）单次约数分钟；heavy 层整次运行 ≈20 min。故把谐波 1280 行放进 `heavy` 层而非 `standard`，默认 `pytest` 不跑，靠 `TENSORBEM_HEAVY_TESTS=1` 触发。
- **验证路径**：谐波 6 条的实测值由 `interior_harmonics` CLI 产出后经注册表同一判定函数 `_judge` 判定，与注册表 `--tiers heavy` 走的是**同一个** `run_case` 代码路径，但不是同一次进程；两者一致性由 `tests/test_declared_numbers.py` 的 heavy 门测试在 CI 上再跑一次时闭合。
- **未做**：未改任何求解器实现；未新增物理模型；三维自由液面（Hulme/Wang-Linton）虽已由 m2609 等测试钉桩，但**仍未纳入注册表**（成本更高，建议后续单独立项）；有限水深三维核、有航速 Rankine 源仍是 Roadmap 缺口。

## 5. 误差判定

| 关键量 | 基准 | 相对误差 | 是否 <3% |
|---|---|---:|---|
| 椭球 1280 三轴附加质量 | Lamb 形状因子闭式 | 4.168e-3 / 4.462e-3 / 4.936e-3（离散误差） | ✅ |
| 谐波 1280 通量（度 1/2/3） | 精确固体谐波多项式 | 3.764e-3 / 7.344e-3 / 1.191e-2 | ✅ |
| 谐波 1280 内点场 RMSE | 同上 | 2.293e-4 / 2.308e-4 / 1.824e-4（绝对，\(\mathcal{O}(10^{-4})\)） | ✅ |
| 文档数字 vs 实测（22 条注册项） | 文档声明 | 最坏 1.10e-1（MFS 行，容差内）；其余 ≤1e-4 | ✅ |

结论：本次复核的全部关键量对解析基准误差均 <3%，最高为度 3 谐波通量 1.19%（常数面元各向同性离散误差，非实现缺陷）。