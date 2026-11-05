# CI 修红报告：PR #125（TensorBEM，分支 `agent/m3173`）→ m3808

## 1. 现象

GitHub Actions `tests` 对 PR #125（`agent/m3173`：doc-claim 覆盖审计 + `docs/finite_depth3d.md`
64 个数字上钉）报 `pytest=failure`；分支自身 `tests/test_doc_claim_audit.py` /
`tests/test_finite_depth3d_pins.py` 全绿（16 passed），因此失败只可能来自 **main 漂移**。

## 2. 复现

`pull_request` 事件下 Actions 检查的是 `refs/pull/125/merge`（分支与 main 的合并结果）。
本地按同样语义复现（分支 落后 main 54 个提交）：

```bash
git checkout -B agent/m3808 agent/m3173
git merge origin/main                # 无冲突，63 个新测试文件随之进入
PYTHONPATH=src python -m pytest tests/test_doc_claim_audit.py tests/test_finite_depth3d_pins.py -q
# -> 5 failed, 11 passed
```

失败项：

| 测试 | 失败内容 |
|---|---|
| `test_doc_claim_audit.py::test_ratchet_against_baseline` | `new unguarded document: finite_depth3d_regime.md` 等 2 篇论文级文档 |
| `test_finite_depth3d_pins.py::test_every_number_reproduces` | 3 行 `DRIFT`：§3.2 自由面残差 1.26e-15 → 1.48e-15（相对 17%） |
| `test_worst_deviation_is_small` | 最坏相对偏差 1.72e-1 > 1e-2 |
| `test_mutated_document_is_caught` / `test_cli_artifact_roundtrip` | 依赖前者绿灯 |

## 3. 归因（两类，均为 main 漂移 + 测试陈旧 pin，不是产品缺陷）

### 3.1 自由面残差 DRIFT：**测试判据陈旧**（对机器零用了相对判据）

main 的 m3409（`finite_depth3d.py`）把传播模态 `cosh(k0*(z+h))` 改写为对数缩放指数因子
（同一个量的稳定写法，用于 `k0*h >> 1` 时避免 `cosh`/`N_0` 溢出）。改写是数学中性的，
但它改变了求和顺序，使 O(1) 核的**相消噪声**从 1.26e-15 变到 1.48e-15——差 0.98 eps。

程序化核对 `docs/m3808/roundoff_crosscheck.py`（逐行对比合并前后
`docs/finite_depth3d_pins/check.json` 的 `recomputed`，64 行）：

| 类别 | 行数 | 判读 |
|---|---:|---|
| 逐位不变 | 49 | double 完全相同 |
| 末位漂移 | 7 | 只动 1–4 bit，相对 ≤ 4.9e-16 |
| 放大舍入 | 5 | 深水极限（两核相减）把 eps 扰动放大到相对 ≤ 2.8e-9 |
| 相消噪声 | 3 | 自由面残差，绝对漂移 ≤ 1.13 eps |
| 物理改变 | 0 | 无 |

结论：m3409 数值等价；`|declared| ~ 1e-15` 的行上「相对偏差 17%」是 1 eps，判据本身不成立。

### 3.2 棘轮红：**main 漂移**（新增论文级未守文档）

合并进来的 main 提交带了两篇新论文级文档，它们**在 main 上就是 UNGUARDED**：
`finite_depth3d_regime.md`（76 数字 token）、`hydrostatic_restoring_couplings.md`（80）。
同一次合并里另有 `audit_baseline.md`、`finite_depth_2d.md` 被新测试读上（离开未守集合），
所以未守文档数仍是 14，论文级 7 → 8。

## 4. 修复（最小改动）

1. `src/tensorbem/benchmarks/finite_depth3d_pins.py`
   - 新增 `NOISE_FLOOR = 1e-12`，**只对 `*residual*` 类行**生效：`|declared| <= NOISE_FLOOR`
     时改按绝对地板判定；`0.000e+00` 的海床残差仍用更严的 `ZERO_FLOOR = 5e-16`；
     小物理量（`evanescent_share = 4.6e-20`）**不豁免**，仍按相对判据。
   - `Row.is_roundoff_zero` / `abs_deviation`，`summary()` 增加 `roundoff_zero_rows`、
     `noise_floor`，最坏相对偏差统计排除噪声行，DRIFT 打印区分相对/绝对判据。
2. `src/tensorbem/benchmarks/doc_claim_audit.py`
   - baseline 写出 `note` 溯源字段（`--baseline-note`），棘轮失败信息附上溯源与重生成命令。
3. `docs/doc_claim_audit/baseline.json` 重生成（含 2 篇 main 新增文档的溯源说明）、
   `docs/doc_claim_audit.md` / `audit.json` / `doc_claim_audit.png`、`docs/finite_depth3d_pins/check.json`
   全部程序化重生成；`docs/finite_depth3d.md` 追加 §8（m3409 等价性核对，数字由脚本注入）。
4. 新增/加强测试（5 条）：
   - `test_roundoff_rows_are_identified`：只有 3 行自由面残差属于噪声行。
   - `test_roundoff_floor_absorbs_ulp_noise_only`：**负控**——残差 4e-10 → `DRIFT`；
     `0.0` vs `1e-15` 海床残差 → `DRIFT`；`evanescent_share` 翻倍 → `DRIFT`；普通行仍按打印精度。
   - `test_worst_deviation_ignores_roundoff_rows`：最坏偏差是物理行。
   - `test_baseline_records_provenance`：baseline 必须写明棘轮语义。

## 5. 验证

```bash
PYTHONPATH=src python -m tensorbem.benchmarks.finite_depth3d_pins --strict   # 64/64 PASS, 0 DRIFT
PYTHONPATH=src python -m pytest tests/test_doc_claim_audit.py tests/test_finite_depth3d_pins.py -q   # 20 passed
PYTHONPATH=src python -m pytest tests -q                                    # 见 §6
```

## 6. 残留与后续

- 两篇 main 新增论文级文档仍未守（写进 baseline 溯源，棘轮只允许收缩）；给它们上 pin 属各自任务。
- 自由面残差可考虑在文档里改印 `<= 1e-15 (machine zero)` 而非具体数值，从源头消除无意义 pin。