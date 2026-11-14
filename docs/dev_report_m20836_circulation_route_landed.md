# CASE-BEM-LAPLACE-CYLINDER：把滞留分支上的环量路线（route G）并回 main 并重新取证

任务 m20836（selfstart-2，generalist 席）。锚点 `CASE-BEM-LAPLACE-CYLINDER`（算例池·圆柱绕势流速度场，
容差 `<3%`，ref=解析复势）。本单不重做物理，补的是**验证欠账：一条已判 passed 的基准行在 main 上不可达**。

## 1. 起点：账上说 passed，main 上没有代码

`tensor_benchmark_cases` 有该锚两行，两行都 `status='passed'`：

| id | 场景 | passed_at |
| --- | --- | --- |
| 130 | 无环量均匀来流（route D/N，N=32/64/128） | 2026-11-14 08:53 |
| 139 | **给定环量 Γ=3Ua 旋转圆柱（route G，Magnus/Kutta-Joukowski）** | 2026-11-14 14:29 |

但干净检出（HEAD=`0f691687`）里第二行**没有任何落地物**：

```
$ grep -c "circulation\|Gamma" src/tensorbem/benchmarks/case_cylinder_stream.py   -> 0
$ ls src/tensorbem/lifting_cylinder.py                                            -> No such file
$ git merge-base --is-ancestor 7c85c390 HEAD                                      -> NOT IN MAIN
$ git branch -a --contains 7c85c390                                               -> agent/m20511
```

即第二行的证据只活在未合并分支 `agent/m20511`（commit `7c85c390`）上——正是 m20424 审计给本锚点
钉下的失败模式「分支未合并即视为已交付」，这次复发在**环量路线**上。`feature_ledger` 该锚仍为 `open`。

## 2. 做了什么（只做合并搬运 + 重新取证，不重写物理）

1. `git merge agent/m20511 --no-commit`，6 处冲突全部是**两侧各自新增**（不是同一行的语义冲突）：
   - 代码：HEAD 侧是第 4 轮加的远场多极矩常量/判据/`plot_multipole`（route M）+ `multipole2d` 能力清单，
     m20511 侧是环量常量/判据/`plot_circulation`（route G）+ `lifting_cylinder` 能力清单 →
     **两侧全留**（`_multipole` 与 `_circulation` 判据互不重叠，共 38 条）；
   - 生成物（`results.json` / `case.json` / `report.md` / `docs/CASE-BEM-LAPLACE-CYLINDER.md` / `*.png`）：
     不手工合并，一律**用合并后的入口点重跑重生**（这是本算例既有纪律：文档必须逐字等于 `render_markdown(results.json)`）。
2. `tests/test_case_cylinder_stream.py` 的「工件自持」门从 4 张图扩到 5 张（补 `plot_circulation`），
   注册表 `figures` 也随之成为 5 张——否则该门会因注册表/入口点集合不等而红。
3. 重跑入口点（递交计算门 cj2986，n330，单核，wall 144.6 s），重生 `results.json`/`case.json`/`report.md`/`docs`。

## 3. 计算结果（预算 N=128，Γ=3Ua，ρ=1）

| 关键量 | 本次重生值 | 判据 | 是否 <3% |
| --- | --- | --- | --- |
| 升力 $L'$ vs Kutta-Joukowski $-ρUΓ=-3$ | $-3.001204$（相对 4.014e-04） | `<0.03` | 是 |
| 物面速度 vs $v_θ=Γ/(2πa)-2U\sinθ$ | 2.131e-04 | `<0.03` | 是 |
| $Cp$ vs $1-(v_θ/U)^2$ | 5.587e-04 | `<0.03` | 是 |
| 外部速度场 vs $dW/dz$ | 1.481e-04 | `<0.03` | 是 |
| 环量（物面速度线积分）vs Γ | 2.008e-04 | `<0.01` | 是 |
| 阻力 $C_D$（d'Alembert 应为 0） | $7.98\times10^{-17}$ | `<0.01` | 是 |

判据总数 38（原 29 + 环量 9），`failed_checks = []`，`overall = PASS`。与 `tensor_benchmark_cases` id=139
登记的数字逐项一致（4.0141e-04 / 2.1308e-04 / 5.5867e-04 / 1.4808e-04 / 2.0085e-04 / 8.0e-17），
说明合并与重生没有改变结论，只是让它**在 main 上可达**。

## 4. 验证

- `pytest tests/test_lifting_cylinder.py -q`（不含整例门）→ **14 passed**，8.6 s；
- `pytest tests/test_case_cylinder_stream.py::test_committed_case_artifact_matches_the_committed_document`
  → **1 passed**，15.6 s（该门逐字比对 `report.md` 与 `docs/` 是否等于 `render_markdown(results.json)`、
  5 张图能否从 `results.json` 重建且被正文链接、注册表与入口点图集合相等）；
- 入口点重跑（cj2986）本身是端到端复现：`overall=PASS`，38/38 判据 ok。

## 5. 注意点（审稿式自查，诚实标注）

- **环量判据的「非空性」偏弱**：内接正多边形面板中点上的单位涡通量 $v_{unit}\cdot n$ 恒为
  $O(10^{-16})$（测试 `test_vortex_flux_is_zero_on_inscribed_polygon_midpoints` 固定了这一点），所以
  route G 的第二类响应解**恒为零**，环量是解析给定后重装出来的。于是
  「环量线积分 = Γ」这条判据主要是重装算术的自检，**不构成对求解器产生环量能力的检验**；
  真正有数值内容的是双重解（doublet）那一路——它同时被外部速度场 vs $dW/dz$、物面速度、$Cp$、
  升力积分四条判据交叉约束。模块 docstring 已如实写明「a rotating cylinder tests: no Kutta condition
  is solved here」，本单沿用该口径。
- 升力符号约定（逆时针 Γ 产生向下升力 $L'=-ρUΓ$）是易错点，`lifting_cylinder` docstring 与
  `lifting_kutta_closure.md` 一致；
- 图仍不进 git（推送门丢弃 `docs/` 外的 `*.png`），可复现性由「从 `results.json` 重建」的门保证，不靠二进制；
- 本次只做搬运+重新取证，未改任何求解器/核函数；`results.json` 中多极矩（route M）与近壁窗两节由重跑重新生成，
  数值与合并前一致（同一代码路径）。

## 6. 复现命令

```
PYTHONPATH=src python3 -m tensorbem.benchmarks.case_cylinder_stream \
  --outdir benchmarks/verified/case_bem_laplace_cylinder --doc docs/CASE-BEM-LAPLACE-CYLINDER.md
PYTHONPATH=src python3 -m pytest tests/test_case_cylinder_stream.py tests/test_lifting_cylinder.py -q
```