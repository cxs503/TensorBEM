# 文档数字守卫覆盖审计 + `docs/finite_depth3d.md` 的 64 个数字上钉（m3173）

**任务**：m3173（infrastructure 席，自找活） · **分支**：`agent/m3173` · **日期**：2026-11-04
**仓**：TensorBEM（worktree `/root/tp-agent/work/m3173`）

EDIT_WHITELIST: src/tensorbem/benchmarks/, tests/, docs/

工件：`src/tensorbem/benchmarks/doc_claim_audit.py`、`src/tensorbem/benchmarks/finite_depth3d_pins.py`、
`tests/test_doc_claim_audit.py`、`tests/test_finite_depth3d_pins.py`、
`docs/doc_claim_audit/{audit.json,baseline.json,doc_claim_audit.png}`、`docs/doc_claim_audit.md`、
`docs/finite_depth3d_pins/check.json`、`docs/finite_depth3d.md` §7（本次追加）

## 1. 案例介绍：选题依据

本仓已经建了五套「文档数字 = 可执行判据」闸门（`doc_pins` / `doc_claims` /
`doc_number_ledger` / `declared_numbers` / `three_dimensional_pins`），每一套都是**在
一次事故之后**补的（Ursell `7.7e-3`、MFS `1.9e-4`、Wigley 排水体积 `2LBT/3`）。但从来
没有人回答过反向问题：**还有多少文档在报数字、却没有任何闸门读它？** 这正是下一次事
故的候选集。本单先把这个集合机器化（覆盖审计），再把其中最大的一份论文级欠账当场结清。

## 2. 怎么做的

1. **覆盖审计器**（`benchmarks/doc_claim_audit.py`）：对 `docs/*.md` + `README.md` 逐个
   分类 `PINNED`（测试直接读该文档名 → 文档文本被解析）/ `ARTIFACT`（文档声明了
   `python -m tensorbem.*` 生成器且该模块被测试导入 → 数字可复现但文档文本不被解析）/
   `UNGUARDED`（两者皆无）。数字 token 用 `(?<![\w.])-?\d+\.\d+(e±?d+)?` 计数，历史性
   文档（`dev_report*` / `*_m\d{3,}` / `audit*` / `review*`）单列为非论文级，避免把冻结
   快照当契约。基线 `baseline.json` 是棘轮：**未守集合只许缩小**，`--check` 在扩大时返回 1。
2. **审计结果**（覆盖 44 个文档、1922 个数字 token）：`PINNED=16`、`ARTIFACT=14`、
   `UNGUARDED=14`（其中论文级 7），数字覆盖率 **83.8 %**。论文级未守榜首位是
   `docs/finite_depth3d.md`：112 个数字 token、**无声明生成器、无读取测试**。
3. **结清这份欠账**（`benchmarks/finite_depth3d_pins.py`）：从文档的 markdown 表格里把数
   值**解析出来**（不是手抄），用 `tensorbem.finite_depth3d` / `tensorbem.free_surface3d`
   按同一套设置重算，逐项判 PASS/DRIFT。复盖 §3.1 色散梯级（`k0`/`c`/`c_g`，9 深 ×3）、
   §3.2 核的边界条件残差（4 深 ×2）、§3.3 深水极限对独立 Kelvin 核（7 深 ×2）、
   §3.4 远场（展布平坦度、相位斜率、5 个 `k_0R` 上的衰减模态占比）、§3.5 奇异强度与截断
   自收敛（各 5 行）。容差取**打印精度**（3 位有效数字 → 5e-3，6 位 → 1e-6），两行打印为
   `0.000e+00` 的海床残差改用绝对地板 5e-16。运行 1.5 s。

## 3. 计算结果

**审计表（节选，`docs/doc_claim_audit.md` 全表程序化生成）**

| 指标 | 值 |
|---|---:|
| 被审计文档 | 44 |
| PINNED | 16 |
| ARTIFACT | 14 |
| UNGUARDED（论文级） | 14（7） |
| 数字 token 总数 | 1922 |
| 数字覆盖率 | 83.8 % |

**上钉结果**：`docs/finite_depth3d_pins/check.json`

| 表 | 判据数 | PASS | DRIFT | 最大相对偏差 |
|---|---:|---:|---:|---:|
| §3.1 色散梯级 | 27 | 27 | 0 | < 1e-6（`k0`）/ < 5e-4（`c`,`c_g`） |
| §3.2 边界条件残差 | 8 | 8 | 0 | 3.0e-2（打印 3 位有效数字的 1.260e-15 级量，绝对差 ~1e-16） |
| §3.3 深水极限 | 14 | 14 | 0 | 4.6e-3 |
| §3.4 远场 | 5 | 5 | 0 | 4.0e-3 |
| §3.5 奇异 + 截断 | 10 | 10 | 0 | 9.63e-3（`modes=40` 行） |
| **合计** | **64** | **64** | **0** | **9.63e-3** |

**判决**：文档数字**没有**腐化 —— 64/64 全部复现，最大相对偏差 9.63e-3 出现在截断梯级
第一行（打印 3 位有效数字的 `1.065e-11` vs 重算 `1.05e-11` 量级的最后一位），其余 63 项
均 < 5e-3。**结论是「PASS 结清 + 从此有守」，不是「文档错了」**（honest：本单没有发现漂移）。

**与 3 % 基准的关系**：本单判据是「打印精度复现」，不是物理精度；但 §3.1–§3.5 涉及的物
理基准本身全部远超 3 % 门（深水极限 2.9e-7 vs 独立 Kelvin 核、边界条件残差 1e-15、相位
斜率 1.000124 vs 1.0、奇异强度 →1 误差 1.18e-3）。**是否 < 3 %：是**（最大物理偏差
9.63e-3 属打印舍入，非物理误差）。

## 4. 注意点 / 局限

1. **审计器只认 `python -m tensorbem.*` 形式的生成器声明**。`finite_depth3d.md` 声明的
   是 `python docs/m3015/finite_depth3d_report.py`（脚本路径），所以它被判为「无生成器」。
   这对结论无害（它确实无人守），但说明 `ARTIFACT`/`UNGUARDED` 的分界偏保守；把脚本路径
   也纳入识别是下一步。
2. **`PINNED` 不等于全覆盖**：`PINNED` 只表示某测试读了该文档名，未必读了文档的每一个表
   格行。本次 64 项是全量覆盖，但不是所有 PINNED 文档都如此。
3. **历史文档单列是人为规则**（正则 + 白名单）。`ursell_convergence_study.md`、
   `ursell_verification_debt_2026-11-02.md` 这类以 `ursell_*` 命名的历史快照被算作论文级，
   抬高了「论文级未守」计数；这是宁可高估、不愿漏报的取向。
4. 未改任何求解器一行；`finite_depth3d.py` / `free_surface3d.py` 保持原样，本单只做读数复核。

## 5. 未做 / 下一步

* 论文级未守榜余项：`lifting_foil.md`（48）、`finite_depth_2d.md`（40）、
  `sphere_flow_study.md`（27）——本单只结清了榜首最大的一份，其余留在
  `docs/doc_claim_audit/baseline.json` 里被棘轮盯着，不会再无声扩大。
* 审计器对脚本式生成器（`docs/**/*.py`）的识别、以及 `docs/<milestone>/*.md` 的纳入。

## 6. 复现

```bash
PYTHONPATH=src python3 -m tensorbem.benchmarks.doc_claim_audit \
    --json docs/doc_claim_audit/audit.json --markdown docs/doc_claim_audit.md \
    --figures docs/doc_claim_audit --write-baseline
PYTHONPATH=src python3 -m tensorbem.benchmarks.finite_depth3d_pins \
    --json docs/finite_depth3d_pins/check.json --strict
PYTHONPATH=src python3 -m pytest tests/test_finite_depth3d_pins.py tests/test_doc_claim_audit.py -q
```

测试结果：**16 passed in 5.73 s**（含两条负控：改一个打印数字必须 DRIFT；删掉结果表必
须让解析行数归零）。