# 死 skip 门审计：一个从未执行过的防漂移测试，及其同类缺陷的机器化封堵

**任务**：m3487（paper 席自找活） · **分支**：`agent/m3487` · **日期**：2026-11-05
**仓**：TensorBEM（worktree `/root/tp-agent/work/m3487`，基线 `b63dfbd`）
**工件**：`docs/ursell_reproducibility/{ursell_reproducibility_study.json,ursell_reproducibility_table.md}`（新落仓）、
`docs/figures/ursell_reproducibility.png`、`docs/m3487/fig_dead_skip_gate.png`、
回归 `tests/test_skip_gates_are_tracked.py`、改动 `tests/test_ursell_reference_pinning.py`、`examples/ursell_reproducibility_study.py`

EDIT_WHITELIST: docs/, src/, tests/, examples/, outputs/

---

## 1. 案例介绍

### 1.1 欠账是哪一条

`docs/ci_red_triage_m3314.md` §3 的「诚实说明」里留了一句从未有人收口的话：

> `test_ursell_reference_pinning.py` 读的 `outputs/ursell_reproducibility_study.json` 缺失时会 `skip`，且 CI 从不生成它。

本单把它坐实并结清。这条欠账的性质比「少跑一个测试」严重：`tests/test_ursell_reference_pinning.py`
的模块 docstring 自称有两层防线，第二层是

> `test_study_artifact_matches_the_current_code` **re-derives the committed evidence artifact** so the table
> in the report cannot drift away from the code it claims to describe

而它的写法是

```python
STUDY_JSON = ROOT / "outputs" / "ursell_reproducibility_study.json"   # 未被 git 跟踪
...
if not STUDY_JSON.exists():          # pragma: no cover - artifact is committed
    pytest.skip("reproducibility study artifact not present")
```

`outputs/` 既不在版本库里（`git ls-files` 无此路径），也不被 CI 任何步骤生成（`.github/workflows/tests.yml`
只写 `artifacts/`），于是**这两个测试自建立起从未执行过一次**，注释 `# pragma: no cover - artifact is committed`
本身就是错的：工件并没有被提交。这正是 2026-10-24 扩编审计记录的那类事故（声明数字无人守护）的「测试版」。

### 1.2 物理与几何（被测对象）

Ursell(1949) 潜没圆柱深水辐射，2-D 势流，单位长度量：

| 项 | 值 |
|---|---|
| 半径 \(a\) | 1.0 m |
| 轴心潜深 \(f\) | 2.0 m（\(a/f=0.5\)） |
| 无因次频率 \(ka\) | 0.6（\(\omega=\sqrt{ka\,g/a}=2.425694\) rad/s） |
| 密度 / 重力 | 1025 kg/m³ / 9.80665 m/s² |
| 参考解 | Ursell 多极子闭式系统，截断加倍至系数变化 <1e-12 |
| 参考值 | \(A=2545.167248\) kg/m，\(B=2718.153952\) kg/(m·s) |
| 离散 | 分段常数间接 BIE，24 点 Gauss（光滑项上限 6 点） |

## 2. 怎么做的

1. **审计口径**：用 AST 扫 `tests/*.py`，找出所有「工件不存在就 `skip`」的守卫，两种写法都收：
   `if not X.exists(): pytest.skip(...)` 语句形态，与 `@pytest.mark.skipif(not X.exists())` 装饰器形态；
   再把守卫解析到的路径与 `git ls-files` 对照 —— **判据是「被 git 跟踪」，不是「文件在我的机器上存在」**。
   结果：7 个工件 skip 门 + 1 个直接断言工件，**1 个是死的**（下表）。
2. **修死门**：把 `examples/ursell_reproducibility_study.py` 的输出目录从不可跟踪的 `outputs/` 改为
   `docs/ursell_reproducibility/`，重算全表并落仓；`tests/test_ursell_reference_pinning.py` 指向新路径，
   并把 `pytest.skip` 换成**断言失败**（缺工件=缺陷，不是免跑理由）。
3. **封堵同类**：新增 `tests/test_skip_gates_are_tracked.py`，它做四件事——
   ①每个登记的工件门必须存在且被 git 跟踪；②AST 扫描出的门集合必须与登记表**逐条相等**（新增门未登记即红，登记表腐化也红）；
   ③工件必须非空且可解析为 JSON（空文件的门等于没门）；④负控：断言修复前的 `outputs/...` 路径确实不存在且未被跟踪。
4. **验证**：`pytest tests/test_skip_gates_are_tracked.py tests/test_ursell_reference_pinning.py` → **23 passed**，
   其中 `test_ursell_reference_pinning.py` 的 4 条从「skip」变成真实执行并全绿。

### 2.1 审计表（8 个工件门，程序化对照 git 索引）

| # | 测试文件 | 守卫表达式 | 工件 | 修前 | 修后 |
|---:|---|---|---|---|---|
| 1 | test_finite_depth_energy_convergence.py | `ARTIFACT` | docs/m2498/outputs/finite_depth_energy_identity/convergence.json | 活 | 活（纳管） |
| 2 | test_finite_depth_envelope.py | `ARTIFACT` | docs/finite_depth_envelope/benchmark.json | 活 | 活（纳管） |
| 3 | test_finite_depth_refinement.py | `ARTEFACT` | docs/m2952/outputs/finite_depth_refinement/benchmark.json | 活 | 活（纳管） |
| 4 | test_mixed_boundary.py | `ARTIFACT`（装饰器） | docs/mixed_boundary/outputs/study.json | 活 | 活（纳管） |
| 5 | test_near_surface_band.py | `STUDY` | docs/m3383/near_surface_band_study.json | 活 | 活（纳管） |
| 6 | test_three_dimensional_doc_pins.py | `pins.PINS_FILE` | docs/three_dimensional_pins.json | 活 | 活（纳管） |
| 7 | test_ursell_quadrature_convergence.py | `summary_path`/`raw_path` | docs/ursell_quadrature_study/{summary.json,raw/} | 活 | 活（纳管） |
| 8 | **test_ursell_reference_pinning.py** | `STUDY_JSON` | `outputs/ursell_reproducibility_study.json` | **死（永不执行）** | **活（断言）** |

## 3. 计算结果

由 `examples/ursell_reproducibility_study.py` 重算，工件 `docs/ursell_reproducibility/ursell_reproducibility_study.json`（零手抄）：

| 面元数 | A (kg/m) | A 相对误差 | B (kg/(m·s)) | B 相对误差 | 近场/远场阻尼比 | 与 m2232 旧工件偏差 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 2587.7627 | 1.6736e-02 | 2787.2368 | 2.5415e-02 | 1.010535710980 | 0.00e+00 |
| 128 | 2566.8343 | 8.5130e-03 | 2754.4701 | 1.3361e-02 | 1.005341374019 | 0.00e+00 |
| 256 | 2556.0936 | 4.2930e-03 | 2736.7493 | 6.8412e-03 | 1.002689125285 | 0.00e+00 |

观测收敛阶（对参考解的绝对误差按 \(\log_2\) 比）：\(p_A=0.975,\ 0.988\)；\(p_B=0.928,\ 0.966\) —— 一阶，
与 `docs/ursell_validation_note.md` 的结论一致。近/远场阻尼比 → 1.0027（256 面元），辐射条件恒等式被满足。

**关键判读**：与 m2232 落仓的旧工件逐行**完全相同（偏差 0.00e+00）**，说明这条欠账的性质是**覆盖缺失**
（有正确工件、但没人在看），不是**数值漂移**。修复后守卫第一次真正生效 —— 此后任何内核/求积改动若动了表中数字，
`test_study_artifact_matches_the_current_code` 会在 1e-6 相对容差上红。

## 4. 图

`docs/m3487/fig_dead_skip_gate.png`：
左图——三条曲线（A 误差、B 误差、m2232 旧工件 A 误差）在对数轴上重合，且 64→256 面元近似斜率 \(-1\)，佐证「工件正确、只是无人守护」；
右图——8 个工件 skip 门的修复前状态（绿色=能真正执行，红色=永不执行），修复前第 8 条为红。

无场量云图：本单是测试基础设施/验证欠账类工作，没有新的场解算例；被守护的物理量本身是 2-D 辐射水动力系数，
用上图的两条误差曲线与其收敛阶即可完整表达结论，故不出云图。

## 5. 注意点

- **判据必须钉在 git 索引上**：只看 `Path.exists()` 会在开发者本地全绿、在 CI 静默 skip —— 这正是本条欠账能活这么久的原因。
  新测试因此对每个工件跑 `git ls-files`，并把「未被跟踪」直接判红。
- **装饰器形态容易漏**：`@pytest.mark.skipif(not ARTIFACT.exists(), ...)` 不是 `If` 语句，只按语句扫会漏掉
  `test_mixed_boundary.py`；本单两种形态都扫（首版扫描器就漏了它，被「登记表 vs 扫描结果」的相等判据抓出来）。
- **直接断言型工件单独登记**：修复后 `test_ursell_reference_pinning.py` 不再 skip，故它不属于「skip 门」；
  工件本身仍纳入存在性/可解析性检查（`DIRECT_ARTIFACTS`），否则修好之后又会重新变成无人看管的数字。
- **成本**：重算全表约 3 min（64/128/256 面元 + 3 次 128 面元节点扫描 + 参考解）；本机 load15 ≈ 126/128 核，
  只跑单线程 2-D 求解，未上集群。CI 不必重算 —— 工件落仓、守卫比对现算值即可。
- **未做**：没有改任何求解器实现；`docs/ursell_validation_note.md` §8 里对 `outputs/ursell_legacy_snapshot_878834a.json`
  的引用是旧版脚本的遗留（当前脚本已不产出该文件），本单只把路径段改为新目录，未重建该快照。

## 6. 误差判定

| 关键量 | 基准 | 相对误差 | 是否 <3% |
|---|---|---:|---|
| A @64 / 128 / 256 面元 | Ursell 多极子闭式 | 1.674e-2 / 8.513e-3 / 4.293e-3 | ✅ 全部 <3%（1.67%/0.85%/0.43%） |
| B @64 / 128 / 256 面元 | 同上 | 2.542e-2 / 1.336e-2 / 6.841e-3 | ✅ 全部 <3%（2.54%/1.34%/0.68%） |
| 近场/远场阻尼比 | 精确 1 | 1.05e-2 / 5.3e-3 / 2.7e-3（表值 −1） | ✅ <3% |
| 新旧工件一致性 | m2232 落仓工件 | 0.00e+00（三行逐位相同） | ✅ 远优于 3% |
| 死门审计结论 | `git ls-files` | 8 个门中 1 个被判死（定性判定，非数值） | ✅（判定无需容差，`git ls-files` 为二元真值） |

结论：修复后防漂移守卫首次真正执行并全绿；被守护的物理量对解析基准误差最高 2.54%，全部 <3%。
本单发现的缺陷属**验证覆盖缺陷**（测试结构），不涉及求解器数值误差，故不给「求解器精度」加分。