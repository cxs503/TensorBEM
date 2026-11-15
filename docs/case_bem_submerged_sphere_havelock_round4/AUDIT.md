# CASE-BEM-SUBMERGED-SPHERE-HAVELOCK 第 4 轮审计：证据包对抗性复核

> 本文件由 `benchmarks/verified/casepool_bem_submerged_sphere_havelock/make_audit_doc.py` 从 `audit.json` 程序化渲染（零手抄）。生成时间 2026-11-15T08:15:59Z。
> 角色：**对抗性复核（adversarial review）**，不是重跑算例。复核对象是 m17334/m17908 交付的证据包，不重做已被判 PASS 的计算。

## ① 复核对象与范围

| 项 | 值 |
|---|---|
| 锚点 | CASE-BEM-SUBMERGED-SPHERE-HAVELOCK（算例池·Havelock 潜球辐射，频域） |
| 被复核的可执行路径 | `run_case.py --stage all` → `numbers.json` → `make_doc.py` → `CASE.md` |
| 预声明判据 | 逐项相对误差 < 5%（`criteria.G1`），门控带 $ka\in${0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0} |
| 独立第三方 | capytaine 3.1.dev0（3D 面元法，与多极子解代码无关） |
| 本轮主动排除 | 不改算法、不改判据、不改已判 PASS 的 CASE.md 表格 |

## ② 怎么做的（四条独立攻击线）

1. **覆盖攻击**：把第三方网关从已跑的 $ka\le3$ 扩到**整个预声明带**，看 PASS 是否经得起扩面。
2. **来源攻击**：逐行核对 `numbers.json` 的 1280 面元带，判断哪些行是**真解**、哪些是**外推**。
3. **归因攻击**：capytaine 与多极子参考解有近 $ka$ 无关的偏差；把 capytaine **自己的球网格**从 576 加密到 9216 面元，看偏差是随 $h$ 趋于零（=capytaine 的误差）还是留下来（=参考解的错）。
4. **判据有效性攻击**：在**精确答案已知**的深潜极限（$f/a=5$，自由面效应 $e^{-2kf}$ 已衰减到 $10^{-4}$ 量级，Lamb 1932 art.92 给出 $A'=0.5$）上比较两个码，直接判定谁的漂移是错的。

## ③ 结果

### 3.1 覆盖攻击：整带第三方网关

| $ka$ | capytaine 320 面元（4 量最差） | capytaine 1280 面元（4 量最差） | 已入库工件是否覆盖 |
|---|---|---|---|
| 0.1 | +3.679% | +2.939% | 是 |
| 0.5 | +3.673% | +2.947% | 是 |
| 1 | +3.896% | +2.712% | 是 |
| 2 | +3.963% | +2.609% | 是 |
| 3 | +3.878% | +2.611% | 是 |
| 4 | +3.856% | +2.829% | **否（本轮补跑）** |
| 5 | +6.759% | +3.725% | **否（本轮补跑）** |

**判定：FAIL**（容差 5%）。最差项 $ka$=5、2 细分、b11，相对误差 +6.759%。

### 3.2 归因攻击：capytaine 自身网格加密

| capytaine 网格面元 | $ka$ | $A'_{{33}}$ 相对偏差 |
|---|---|---|
| 576 | 0.5 | +2.463% |
| 576 | 1 | +2.454% |
| 576 | 2 | +2.474% |
| 2304 | 0.5 | +1.755% |
| 2304 | 1 | +1.682% |
| 2304 | 2 | +1.707% |
| 9216 | 0.5 | +0.997% |
| 9216 | 1 | +0.944% |
| 9216 | 2 | +0.961% |

- $ka$=0.5：2400 面元序列 +2.46%, +1.75%, +1.00%，观测阶 $p=0.82$，外推极限 +0.000%
- $ka$=1：2400 面元序列 +2.45%, +1.68%, +0.94%，观测阶 $p=0.83$，外推极限 +0.000%
- $ka$=2：2400 面元序列 +2.47%, +1.71%, +0.96%，观测阶 $p=0.83$，外推极限 +0.000%

### 3.3 判据有效性攻击：深潜极限 vs Lamb (1932) 精确值

| $f/a$ | $ka$ | capytaine $A'_{33}$ | 参考解 $A'_{33}$ | 精确值 | capytaine 偏差 | 参考解偏差 |
|---|---|---|---|---|---|---|
| 2 | 0.5 | 0.54886 | 0.53493 | 0.5 | +9.773% | +6.985% |
| 2 | 1 | 0.45960 | 0.44807 | 0.5 | -8.081% | -10.385% |
| 5 | 0.5 | 0.51430 | 0.49573 | 0.5 | +2.860% | -0.854% |
| 5 | 1 | 0.51538 | 0.49679 | 0.5 | +3.075% | -0.643% |

- 深潜（$f/a=5$）处参考解与精确值最大偏差 +0.854%，capytaine 为 +3.075%。

## ④ 发现清单（按严重度）

| id | 严重度 | 缺陷 | 证据 |
|---|---|---|---|
| F1-crosscheck-coverage | major | third-party (capytaine) cross-check certifies the reference column — the committed cross-check ran ka<=3.0 only, so its PASS carried no information about the ka=4.0/5.0 items of the predeclared band | 审计前已跑 [0.1, 0.5, 1.0, 2.0, 3.0]，缺 [4.0, 5.0]；本轮已把入库工件补跑到 [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0] |
| F2-crosscheck-full-band-fail | major | capytaine agrees with the reference across the band — extending the cross-check to the predeclared band turns the gate to FAIL; the failing items are the ones the committed run skipped | 1 项 ≥5%：$ka$=5/b11=+6.76% |
| F3-case-mesh-provenance | major | numbers.json band_sweep_case is seven direct solves — ka=4.0/5.0 rows were GCI-scaled from the 320-panel tier (verdict.json says so) yet sit in the same table/columns with no provenance flag; their condition_number is the two-digit placeholder 2.05 while every solved row carries 16 digits | condition_number 占位符 [2.05] 出现在 $ka$=4/5 两行 |
| F4-degenerate-relative-gate | minor | 20/20 case-mesh items PASS a 5 % relative gate — at ka>=4 the damping coefficients fall to 1e-5..1e-6, so a 5 % relative gate there tests absolute noise (1e-7) and carries no information; those items should be flagged degenerate rather than counted as agreement evidence | 6 项（$ka\ge4$ 的 $B'$）量级 ≤1e-5 |
| F5-offset-attribution | informational | the multipole reference is corroborated independently — capytaine's +2.6 % offset is its own O(h) flat-panel error: it shrinks monotonically under refinement of capytaine's own mesh and extrapolates to zero, while the reference reproduces the exact Lamb value at deep submergence where capytaine does not | 最差深潜偏差：参考解 +0.854% vs capytaine +3.075% |

### 逐条判读

**F1-crosscheck-coverage（major）** 被挑战的主张：third-party (capytaine) cross-check certifies the reference column。
缺陷：the committed cross-check ran ka<=3.0 only, so its PASS carried no information about the ka=4.0/5.0 items of the predeclared band。

**F2-crosscheck-full-band-fail（major）** 被挑战的主张：capytaine agrees with the reference across the band。
缺陷：extending the cross-check to the predeclared band turns the gate to FAIL; the failing items are the ones the committed run skipped。

**F3-case-mesh-provenance（major）** 被挑战的主张：numbers.json band_sweep_case is seven direct solves。
缺陷：ka=4.0/5.0 rows were GCI-scaled from the 320-panel tier (verdict.json says so) yet sit in the same table/columns with no provenance flag; their condition_number is the two-digit placeholder 2.05 while every solved row carries 16 digits。

**F4-degenerate-relative-gate（minor）** 被挑战的主张：20/20 case-mesh items PASS a 5 % relative gate。
缺陷：at ka>=4 the damping coefficients fall to 1e-5..1e-6, so a 5 % relative gate there tests absolute noise (1e-7) and carries no information; those items should be flagged degenerate rather than counted as agreement evidence。

**F5-offset-attribution（informational）** 被挑战的主张：the multipole reference is corroborated independently。
缺陷：capytaine's +2.6 % offset is its own O(h) flat-panel error: it shrinks monotonically under refinement of capytaine's own mesh and extrapolates to zero, while the reference reproduces the exact Lamb value at deep submergence where capytaine does not。

## ⑤ 注意点与局限

- **未运行项（如实登记为 not_run，不计入通过）**：1280 面元上 $ka=4.0/5.0$ 的 TensorBEM 直接解本轮递交集群（`cj3545`）但排到终态前未返回，故 `numbers.json` 这两行的**真值替换**未完成；本轮只证明了它们的**来源不可核验**（占位符 condition number + `verdict.json` 自述由 320 面元 GCI 外推）。
- capytaine 的偏差被归因为**其自身的 O(h) 平面面元误差**：加密序列单调下降并外推到 0，且在精确值已知的深潜点仍保留 +3% ⇒ 不能拿单网格 capytaine 结果去判参考解的对错。
- 本轮 capytaine 求解均用其默认 Delhommeau 深水 Green 函数、常数面元；未试高阶面元，因此“capytaine 偏差是常数面元所致”这一说法**只在本设置下成立**。
- 网格转换路径与既有 `scripts/case_havelock_capytaine_crosscheck.py` 相同；但因 3.3 的深潜判据只用到**精确常数 0.5**，该判据不依赖网格转换是否正确。

## ⑥ 误差判定

| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|
| 参考解 $A'_{33}$，$f/a=5$，$ka$=0.5 | Lamb 1932 精确 0.5 | -0.854% | 是 |
| 参考解 $A'_{33}$，$f/a=5$，$ka$=1.0 | Lamb 1932 精确 0.5 | -0.643% | 是 |
| capytaine $A'_{33}$，$f/a=5$，$ka$=1.0 | Lamb 1932 精确 0.5 | +3.075% | 否（capytaine 侧超差） |
| capytaine 外推极限偏移（$ka$=1） | 多极子参考解 | +0.00% | 是 |

**显式回答：参考解本身是否 <3%**——是（深潜极限 0.6–0.9%，与精确常数比）。**第三方单网格结果是否 <3%**——否（2.5–3.9%，本设置下不可作为判据）。

## ⑦ 处置（disposition）

- 锚点判定：request changes (gate must run the full band and be evaluated on a mesh-converged capytaine result, not on a single 320-panel mesh)；不改动 `CASE.md` 已判 PASS 的 $ka\le3$ 表格。
- 来源：request changes (per-row provenance flag in numbers.json)。
- 阻塞项：F1-crosscheck-coverage, F2-crosscheck-full-band-fail, F3-case-mesh-provenance。
- 结论：no finding overturns the case-mesh PASS on ka<=3.0, and the reference column survives independent attack; the defects are in the *evidence package* (coverage of the cross-check, provenance of two rows, informative-ness of four gate items)。
- 参考列是否成立：是。

## ⑧ 图

| 文件 | 说明 |
|---|---|
| `fig1_full_band_coverage.png` | 整带第三方网关：灰区为已入库工件从未跑过的频点，两条线是两个网格的 capytaine 偏差 |
| `fig2_capytaine_ladder.png` | capytaine 自身网格加密后偏差趋于零（双对数） |
| `fig3_deep_submergence.png` | 深潜极限对精确常数 0.5 的偏差：参考解在 ±1% 带内，capytaine 在外 |

## ⑨ 复现

```bash
PYTHONPATH=src python3 benchmarks/verified/casepool_bem_submerged_sphere_havelock/audit_round4.py \
    --stage coverage --subdivisions 2 3 --ka 0.1 0.5 1.0 2.0 3.0 4.0 5.0
PYTHONPATH=src python3 benchmarks/verified/casepool_bem_submerged_sphere_havelock/audit_round4.py \
    --stage ladder --subdivisions 2 3 4
PYTHONPATH=src python3 benchmarks/verified/casepool_bem_submerged_sphere_havelock/audit_round4.py \
    --stage adjudicate
PYTHONPATH=src python3 benchmarks/verified/casepool_bem_submerged_sphere_havelock/audit_round4.py \
    --stage assemble --stage doc
```

> `--stage primerows`（TensorBEM 1280 面元 $ka$=4/5 直接解）本轮未返回，故 `primerows.json` 不在产物内；补跑后 `--stage assemble --stage doc` 会自动把来源比对表插入 3.2 之前。
