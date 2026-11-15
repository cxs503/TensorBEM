# XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN — 验证欠账收口

**锚点**: `XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN`（capytaine 的 `finite_depth_crossval_plan` 做法）
**状态**: 计划已定义、全量 12 格交叉验证已递交集群（in-flight），本页为验证欠账的收口记录。

## 1. 案例背景（物理/几何/参数）

对 `tensorbem.finite_depth_solver3d`（有限水深辐射求解器）做**有限水深**下的
code-to-code 交叉验证，对标 capytaine 的有限水深 Green 函数（`Delhommeau(finite_depth_method=...)`）。

- **几何**: 全潜球 `a = 1 m`，球心 `z = -1.5 m`（最低点 `z = -2.5 m`）。球是唯一有
  独立半解析锚（Thorne/Wang/Linton 多极子，`tensorbem.analytic`）的光滑闭体，
  低阶面板法收敛、无不规则频率、`h/a >= 3` 时底部净空恒正。
- **网格**: `h/a in (3, 5, 10, 20)` × `k0*a in (0.5, 1.0, 2.0)` → **12 格**，
  `k0*h in {1.5 ... 40}`（从强浅水到有效深水）。
- **`nu` 回算**: 每格由 `k0 tanh(k0 h) = nu` 反解 `nu`，使 `k0*a` 在深度梯上精确
  （`tests/test_capytaine_finite_depth_crossval.py::test_plan_back_computes_nu...` 已验）。
- **参数**: `rho = 1025`, `g = 9.80665`。
- **面板梯**: TensorBEM 侧 20/80/320 面板（+4 格加 1280 确认档）；
  capytaine 侧 288/1152/4608 面板（+中档 `legacy` 公式作参考自身不确定带）。

## 2. 方法（计划核心）

计划（`tensorbem.benchmarks.finite_depth_crossval_plan`）的四条纪律：

1. **先定容差再跑**：`TOL_ADDED_MASS = 3%`、`TOL_DAMPING = 5%`、owner 上限 15%。
2. **参考自身不确定带**：capytaine `newer`/`legacy` 两公式的展布是自由、代码内带；
   TensorBEM/capytaine 差小于该带不携带信息。
3. **双码都细化、不单信一方**：两码都是低阶常强度面板法，单网格原始差被两个无关
   离散误差主导。计划在每侧跑面板梯，比较 **Richardson 外推** 值，报告观测阶与
   外推残差——把「求解器分歧」与「网格噪声」分开归因。
4. **保留必须移动的负控**：深水求解器（构造上深度盲）跑同格作负控；小 `k0*h` 时
   计划必须把两者分开，否则根本没在测深度。

## 3. 当前状态与验证欠账

**已冻结（1 格，`h/a=3.0, ka=0.5`）**：
- `benchmarks/verified/capytaine_finite_depth/capytaine_ref.json`（capytaine 3 档）
- `benchmarks/verified/capytaine_finite_depth/tensor_ladder.json`（TensorBEM 20/80 档）
- `benchmarks/verified/capytaine_finite_depth/results.json`

**欠账（本单递交，in-flight）**：全量 12 格尚未跑完。`results.json` 的
`execution_status` 为 `PARTIAL`，`test_capytaine_finite_depth_crossval.py` 的
6 个关键门（外推/带内/负控/渐近域/页面）为 `XFAIL`（"partial execution: the
TensorBEM ladder is not extrapolated yet"）。

**本单已递交**（`run_compute`，repo=TensorBEM，8 作业）：
- capytaine 参考 4 分片（`scripts/capytaine_finite_depth_ref.py --shard 0..3 --shards 4`）
- TensorBEM 梯 4 分片（`capytaine_finite_depth_crossval --shard 0..3 --shards 4 --ladder-only`）

**收口流程**（分片跑完后）：
```
# 1) 合并 capytaine 参考（拒绝丢片）
PYTHONPATH=src python scripts/capytaine_finite_depth_ref.py \
    --merge /tmp/cp_ref_shard0.json ... --out benchmarks/verified/capytaine_finite_depth/capytaine_ref.json
# 2) 合并 TensorBEM 梯（要求每格 3 档）
PYTHONPATH=src python -m tensorbem.benchmarks.capytaine_finite_depth_crossval \
    --merge-caches /tmp/tb_ladder_shard*.json --cache benchmarks/verified/capytaine_finite_depth/tensor_ladder.json \
    --out-root benchmarks/verified/capytaine_finite_depth --write-doc
# 3) 重跑门（XFAIL 转真）
python -m pytest tests/test_capytaine_finite_depth_crossval.py
```

## 4. 误差判定（当前 1 格，外推前）

| 量 | 当前最差（1 格，未外推） | 预注册带 | owner 限 | 判定 |
|---|---|---|---|---|
| heave added mass | 12.75% | 3% | 15% | 未外推，不可判 |
| heave damping | 23.85% | 5% | 15% | 未外推，不可判 |
| 负控（k0h=1.5 分离） | 7.01% | 2% | — | PASS |

**显式回答「是否 <3%」**：当前 1 格未外推，`added_mass_agreement` 12.75% 未达 3%，
**不能**判 <3%——因为单网格差被两个无关离散误差主导，必须等全量 12 格外推完成
才能给出可判定的 <3% 结论。这是本单的诚实边界（honest-FAIL 不粉饰）。

## 5. 注意点/局限

- 两码都把球离散为内接多面体，粗档面板面积亏 ~1-2% 同入两侧；外推一阶消掉，
  报告阶说明残差里多少是离散。
- capytaine `newer`/`legacy` 在此几何上到舍入一致，**不**提供独立第二意见；
  参考不确定带是面板梯本身。
- 全潜闭体：无吃水线，不涉及不规则频率处理；表面贯穿有限水深案例仍超范围
  （`docs/finite_depth_solver3d.md`）。
- 阻尼涉及辐射波幅，两侧都比附加质量收敛慢；逐格阶把它显式化而非藏在一个数后。

## 6. 下次自找活建议

- 全量 12 格外推完成后：重跑 `test_capytaine_finite_depth_crossval.py`，把 6 个
  `XFAIL` 转真，并发布带全格云图+收敛梯的论文级报告（`--write-doc` 已生成图）。
- 若外推后某格仍 >3%：按计划第 3 条归因到「求解器」或「细化不足」，不要报成
  「solver 分歧」。
