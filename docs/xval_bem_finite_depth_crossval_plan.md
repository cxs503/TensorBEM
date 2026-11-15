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

## 7. 收口进展（本轮：参考侧 12/12 格完成，判定梯级仍在集群队列）

**变更**：`benchmarks/verified/capytaine_finite_depth/capytaine_ref.json` 由 1 格扩到
**全 12 格**（`h/a ∈ {3,5,10,20}` × `k0a ∈ {0.5,1.0,2.0}`），每格 3 档
`(12,24)/(24,48)/(48,96)` = 288/1152/4608 面板 + `legacy` 公式一档。
生产者 `scripts/capytaine_finite_depth_ref.py` 以 6 分片在本机（capytaine 3.1.dev0 /
loongarch64 / numpy 2.4.6）跑完后 `--merge` 合并，合并器按计划顺序拒绝缺片/重片。
耗时 518.7 s（含 legacy 档）。第 0 格复算与新值相对差 ~6e-6，与冻结值同一代码路径。

**新增判据（可量化）**：用参考侧自身三档量出阶
`p = log|(v1-v0)/(v2-v1)| / log 2`：

| 项 | 值 |
|---|---|
| 逐格条目 | 48（12 格 × A11/A33/B11/B33） |
| **未进渐近区（`p<0.3` 或非单调）条目** | **29 / 48 = 60 %** |
| 最差阶 | −3.46（增量随面板细化反而增大） |
| 明细表 | `docs/xval_bem_finite_depth_ref12_table.md` |
| 图 | `docs/fig_xval_bem_finite_depth_ref12.png`（$A_{33}$/$B_{33}$ 三档随 $k_0h$） |

**这条数字的含义（本锚点最重要的一条）**：参考侧的 `(48,96)` 档在 60 % 的条目上
**还没有进渐近区**（增量 $v_2-v_1$ 大于 $v_1-v_0$），因此在现有分辨率下**对 12 格做
Richardson 外推是不合法的**——若强行外推，参考不确定度大于被测侧，判定会被参考侧的
离散误差主导。单格（`h/a=3, k0a=0.5`）上 m21322 已实证这一点（$A_{33}$/$B_{11}$ 只能取
最细级、不能外推）。**结论：12 格判定的欠账不只是「跑完 TensorBEM 梯级」，还包括
把 capytaine 参考梯级扩到 `(96,192)` = 18432 面板。**

**未完成（如实）**：
- TensorBEM 侧 12 格 × 3 档梯级（20/80/320 面板）已递交集群 `cj3771`（8 核，`--jobs 6`，
  `--ladder-only`），截至本单结束**排队 37 分钟仍未起跑**（在飞 73 作业）。
  故 12 格逐格判定、6 条 `xfail` 转真**均未发生**。
- 集群节点无 capytaine（`cj3338`/`cj3575` 即因此 `ModuleNotFoundError` 失败，队列却记
  `succeeded`——这是一个作业终态记账缺陷，值得单开一条缺陷单）。因此参考侧只能在本机分片跑。
- `results.json` 已按新 12 格参考重出（`execution_status` 仍为
  `PARTIAL: 1 of 12 planned cells`，梯级仍只有第 0 格的 20/80 两档，数字未变、不粉饰）。
