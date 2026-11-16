# 第 7 轮 — T1/T2 行级表补齐：已发布判决现在可从行复算（CASE-BEM-SUBMERGED-SPHERE-HAVELOCK）

锚点 `CASE-BEM-SUBMERGED-SPHERE-HAVELOCK`（算例池 `BEM-SUBMERGED-SPHERE-HAVELOCK`，
ref Havelock 1955 频域多极解，容差 **<5%**）。性质：**验证欠账**（复核已合并结论），非新功能。

## 0. 缺口（为什么这轮值得做）

`benchmarks/verified/casepool_bem_submerged_sphere_havelock/measure_correction_study.py`
（球面测度修正研究，第 5 轮 m21580 引入）发布了 T1 / T2 的**判决**，但它自己的
`not_run` 字段写明：

> "per-ka rows of the study: the run was interrupted before write_artifacts"

也就是说，支撑 T1（320 面元，$ka=0.1\ldots5.0$）与 T2（80 面元，$ka=0.1\ldots3.0$）
判决的**逐 $ka$ 行从未落盘**。没有行的判决不能被第三方复算，只能被相信——而本锚点的
验收判据（5%）正是按行/按项判的。第 6 轮补的是 T3（案定网格 1280）的实测缺口，
**T1/T2 的行级欠账无人收**。本轮就收这一笔。

## 1. 做法

| 件 | 内容 |
|---|---|
| `scripts/havelock_measure_t1t2_rows.py` | **import 已合并的研究模块**，复用它自己的 `Tier` / `_band` / `_verdict`（5% 门槛、频带、三种测度全部原样，不改不放宽）；重算 T1/T2 行并落盘；把重算判决与已提交 `measure_correction/verdicts.json` 做对照（控制 **R2**） |
| 分片执行 | 本机只允许 <10 min/次 的短测试，故按 `--tier/--kas` 分片跑（6 个分片），`--merge` 合并后**统一重算判决**；分片里出现的重复 `(ka, measure)` 单元作为**重复性控制**（两次独立重算） |
| 预声明 | 行表内容 = 研究模块自声明的内容；本脚本不引入任何新参数、新频带、新门槛 |
| 判决归属 | 已提交的 `verdicts.json` **不被改写**（其 `not_run` 原文保留）；本轮只在**新**工件里给出可复算的行与重算判决，并在测试里断言旧工件未被"补妆" |
| `scripts/havelock_round7_figures.py` | 从行表程序直出两张图（不给 T3 造图，T3 无实测） |
| `tests/test_casepool_havelock_t1t2_rows.py` | 10 项守卫：频带/网格覆盖、参考值**重推**（解析级数 $_1e-10$）、判决从行**独立重算**、R2 与已提交判决逐项一致、旧工件不可改写、以及"flat 测度的 FAIL 不得被软化" |

## 2. 计算结果（程序直出，工件 `measure_correction/numbers_t1t2_rows.json`）

判据：5% 相对误差，逐项（每档 $ka$ × 4 个量 = 4×7=28 项 / 4×5=20 项），不改不放宽。

| 档 | 测度 | 超门槛项 / 总项 | 带宽最大误差 | 最大处 | 判决 |
|---|---|---|---|---|---|
| T1 320 面元 | flat（平板弦面积） | 7 / 28 | 5.5911% | $ka=0.5$, $B'_{33}$ | **FAIL** |
| T1 320 面元 | patch（精确球面片面积） | 0 / 28 | 2.2528% | $ka=5.0$, $B'_{11}$ | PASS |
| T1 320 面元 | uniform（flat × 标量 $k$，对照 C1） | 0 / 28 | 1.9060% | $ka=5.0$, $B'_{11}$ | PASS |
| T2 80 面元 | flat | 20 / 20 | 20.2446% | $ka=0.5$, $B'_{33}$ | **FAIL** |
| T2 80 面元 | patch | 9 / 20 | 5.5916% | $ka=0.5$, $B'_{33}$ | **FAIL** |
| T2 80 面元 | uniform | 9 / 20 | 5.5893% | $ka=0.5$, $B'_{33}$ | **FAIL** |

**控制 R2（本轮的核心证据）**：重算判决与已提交 `verdicts.json` 的 T1/T2 条目**逐项一致**
——`n_items`（28 / 20）、`n_items_over_gate`（7 / 0 / 0 与 20 / 9 / 9）、判决
（FAIL / PASS / PASS 与 FAIL / FAIL / FAIL）、`band_max_at` 的 $(ka, 量)$，
以及带宽最大误差本身（相对差 <$10^{-6}$，实测逐位相同：0.05591129869557249、
0.022528036459896397、0.01906015208950514、0.20244581771165263、0.055915740581781725、
0.05589308845841253）。→ 第 5 轮发布的 T1/T2 判决**不是不可复核的断言**，它现在由盘上的行唯一决定。

## 3. 结果图

![T1/T2 相对误差随 ka（三测度，点线=5% 门槛）](case_bem_submerged_sphere_havelock_round7/fig1_t1t2_band.png)

*T1/T2 相对误差随 $ka$（三种测度，点线 = 5% 门槛）；行来自 `numbers_t1t2_rows.json`。*

![按测度与网格的带宽最大误差](case_bem_submerged_sphere_havelock_round7/fig2_cross_mesh_bandmax.png)

*带宽最大误差按测度分列：320（T1）与 80（T2）面元。1280 面元（T3）本轮无实测，未画入。*

## 4. 注意点 / 局限

- **T3（1280 面元，pool gate 真正判的那一档）本轮仍无实测**：第 6 轮登记的预测
  （精确测度下带宽最大误差 ≈0.193%）仍处 `prediction_registered_pending_measurement`。
  本轮递交的集群作业 `cj4350`（7 分片 × 8 核，1280 面元 × 7 $ka$ × 3 测度 = 21 次求解）
  在本单结束时仍在排队（递交时位次 46），未回件 → 一律记 `not_run`。
- 本轮的行是由**同一实现**（同一 `Tier.at` 路径、同一求解器）在分片里重算的，
  因此 R2 是"同源可复算"，**不是**第二套求解器/独立实现的验证；真正的独立锚点仍是
  Havelock 多极解析解（其自证见 `submerged_sphere_havelock_integrity`）与 capytaine 交叉核对。
- **T2 上精确测度并不救场**：patch 仍有 9/20 项 >5%（最大 5.5916%，就在 $ka=0.5$ 的 $B'_{33}$）。
  即"球面片测度去偏置"在 80 面元这一档只把误差从 20.2% 降到 5.59%，**没有**跨过 5% 门槛。
  这与第 5 轮 `verdicts.json` 的 T2 判决一致（并非本轮新发现），但它是"测度修正足够"这一
  说法的反例，必须与 T1 的 PASS 并列读。
- `uniform` 对照在 T1 上（1.906%）甚至略优于 `patch`（2.253%），而两者的最大处相同
  （$ka=5.0$, $B'_{11}$）——这说明 T1 精度带的最后一点点由高频阻尼项主导，与测度的全局尺度重标定无法区分；
  这一点与第 6 轮 `P3`（uniform 不应复现 patch 收益）的判定范围不同（那是 1280 面元、仍无实测），不可混用。
- 本机计算纪律：分片按 `--kas` 拆到 <10 min 的短测试内跑，未起长算力；因此单次分片可能
  在时限内被收割，本轮靠"分片 + 合并 + 重复单元控制"收口，**没有**手工转抄任何数字。

## 5. 误差判定（对基准逐项，是否 <3%）

关键量 = 各档带宽最大相对误差（对照解析多极解）：

| 档 / 测度 | 带宽最大误差 | 是否 <5%（算例容差） | 是否 <3%（更严） |
|---|---|---|---|
| T1 / patch | 2.2528% | 是 | **是** |
| T1 / flat | 5.5911% | 否（honest FAIL） | 否 |
| T2 / patch | 5.5916% | 否（honest FAIL） | 否 |
| T2 / flat | 20.2446% | 否 | 否 |
| T3 / 任意测度 | **无实测** | — | — |

结论：**"<3%" 只在 320 面元的精确测度上成立**（2.25%）；案定网格 1280 的 <3% 断言本轮
**不能回答**（T3 未回件，见 §4），沿用第 3 轮已判的 PASS（1280 flat 测度 28/28 <5%、
最大 1.431%）也不等于回答 3%（1.431% < 3% 是成立的，但那不是本轮的测量，出处是
`numbers.json::band_sweep_case`，本轮未重测）。

## 6. 复现

```
# 分片（每个 < 10 min）
python3 scripts/havelock_measure_t1t2_rows.py --tier T1 --kas 0.1,0.5,1.0 --shard-name _t1a.json --outdir /tmp/t1t2
python3 scripts/havelock_measure_t1t2_rows.py --tier T1 --kas 1.0,2.0     --shard-name _t1d.json --outdir /tmp/t1t2
python3 scripts/havelock_measure_t1t2_rows.py --tier T1 --kas 3.0         --shard-name _t1f.json --outdir /tmp/t1t2
python3 scripts/havelock_measure_t1t2_rows.py --tier T1 --kas 4.0         --shard-name _t1g.json --outdir /tmp/t1t2
python3 scripts/havelock_measure_t1t2_rows.py --tier T1 --kas 5.0,0.1,0.5 --shard-name _t1c.json --outdir /tmp/t1t2
python3 scripts/havelock_measure_t1t2_rows.py --tier T2                   --shard-name _t2.json  --outdir /tmp/t1t2
# 合并 + 判决重算 + R2 对照
python3 scripts/havelock_measure_t1t2_rows.py --merge /tmp/t1t2/_t1{a,c,d,f,g}.json /tmp/t1t2/_t2.json
# 图 + 守卫
python3 scripts/havelock_round7_figures.py --t1t2 docs/case_bem_submerged_sphere_havelock/measure_correction/numbers_t1t2_rows.json --outdir docs/case_bem_submerged_sphere_havelock_round7
python3 -m pytest tests/test_casepool_havelock_t1t2_rows.py -q      # 10 passed
```

## 7. 处置

**approve with notes**：T1/T2 的行级欠账已收口（行 + 重算判决 + R2 逐项一致 + 10 项守卫），
这是本轮实质增量；**T3（1280）仍未实测**，如实登记 `not_run`，第 6 轮的登记预测继续挂账，
集群作业 `cj4350` 回件后按其自身判据判决。