# m23091 — CASE-BEM-CUMMINS-WIGLEY 说明书的 DOF 归属错误（PR#1081 重开第 3 轮）

## 0. 本单结论（一句话）

`docs/casepool_bem_cummins_wigley/README.md` 由 `tools/render_casepool_bem_cummins_wigley_readme.py` 渲染，
但渲染器里有两处手写索引/切片错误，把 **yaw** 的阻尼最小值印发成 **pitch**，并把真正的最坏值
（pitch = −1.463，整份 n315 扫描里唯一的量级级负阻尼）印成一个没有 DOF 的裸数字；同时提交件一侧的
`min diag B` 被 `np.diag()` 的副对角线零压成「恰好 0」。三处都已按「从数据取 DOF、切片本身即对角线」
修正，README 重新渲染，并新增两条会咬住旧实现的测试。

## 1. 背景：PR#1081 的交付物在当前 main 上的实际状态

| 交付物 | main 上的位置 | 状态 |
| --- | --- | --- |
| `docs/casepool_bem_cummins_wigley/README.md` | 同上 | 存在（由 m22876 改为渲染） |
| `evidence_cluster_n315_sweep_coarse.npz` | 同上，41 节点 × 6 DOF | 存在，可读 |
| `fig_node_vs_local.png` | 同上 | 存在；渲染器引用 |
| `src/tensorbem/benchmarks/case_bem_cummins_wigley.py`（C0 闸门 / voided 依赖） | 同仓 | 已落地 |
| `tests/test_case_bem_cummins_wigley.py` | 同仓，78 用例 | 已落地 |
| `tools/rerender_casepool_bem_cummins_wigley.py` | 同仓 | 已落地 |

即：PR#1081 的功能代码早已通过后续 mission（m21960 / m22876）进入 main，剩下能「管到底」的只有一件事
——**这份说明书自己是不是真的等于工件的渲染结果**。m22876 已经把「数字从工件注入」这条做到了文本层面
（`test_casepool_readme_is_the_render_of_the_artifacts` 比对渲染与提交文件逐字相等），但**没有**校验
「数字旁边那个 DOF 名字是不是这个数字的 DOF」。本单补的正是这一层。

## 2. 缺陷（两处，同一个表达式家族）

### 2.1 手写索引把 yaw 当 pitch

`_sweep_facts()` 旧实现：

```python
"surge_min": float(np.diag(damping[:, 0, 0]).min()),
"heave_min": float(np.diag(damping[:, 2, 2]).min()),
"pitch_min": float(np.diag(damping[:, 5, 5]).min()),   # 5 是 yaw
```

`DOF_NAMES = ("surge", "sway", "heave", "roll", "pitch", "yaw")`，所以索引 4 才是 pitch、5 是 yaw。
后果直接出现在已发布的 README 第 4 节表格里：

| 量（coarse 12x4） | 旧 README 印的 | `.npz` 里的真值 |
| --- | --- | --- |
| min B（surge / pitch） | −0.947 / **−0.000681** | −0.947 / **−1.463** |
| 其中 −0.000681 实际属于 | — | **yaw**（pitch 的不是这个数） |

同页第 2.1 节又把 −1.463 印成 `-1.463（surge -0.947）`——括号里给了 surge 的值，却没给出
−1.463 自己的 DOF，读者只能猜。也就是说：**同一个文件里「pitch」出现了两个互不相同的值**，
而真正最坏的那个值没有名字。

### 2.2 `np.diag()` 把「非负」的 DOF 压成恰好 0

```python
np.diag(damping[:, 0, 0])     # 入参是 1-D 切片 -> 返回 41x41 的「对角矩阵」
```

`np.diag` 对 1-D 输入构造的是矩阵，副对角线全是 0，于是 `.min()` 恒 ≤ 0。对 n315（有真实负值）
看不出问题（最小值本来就是负的），但对**通过 C0 的提交件**：每个 DOF 都被报成「恰好 0」。

真值（`sweep_coarse.npz`，B(ω=0.2 rad/s) 已趋于 0 但为正）：

| DOF | 旧 README | 真值 |
| --- | --- | --- |
| surge | 0（下界） | 1.986e-08 |
| pitch | 0（下界） | 2.432e-07 |
| yaw | 0（下界） | 2.701e-13（与 C0 闸门自报的 min B 2.7e-13 一致） |

「0（下界）」这个说法还与同一份 README 自己的 C0 行（min B 2.7e-13 ≥ 0）矛盾。

## 3. 修法（最小改动，只碰渲染器 + 它的说明书 + 测试）

1. `_sweep_facts()` 改为按 `DOF_NAMES` 枚举建 `per_dof_min: {name: min}`，不再手写列号；
   全局最小值附 `min_damping_dof`（谁最小）、`min_damping_omega`（带内哪里最小）、
   `negative_damping_nodes`（6 DOF×41 节点里多少为负）。
2. 对角线取 `damping[:, i, i].min()`——切片本身就是该 DOF 的对角线，`np.diag` 只在 2-D 上才有意义。
3. 由此「恰好 0」不可能再出现，删掉 `_floor()` 这个特例（它把真值遮成「0（下界）」）。
4. README 相应列改为带 DOF 名的形式，并新增一句由工件注入的说明：n315 的最小值落在带中
   ω = 5.861 rad/s 且为负（造能），提交件的最小值落在带底 ω = 0.2 rad/s 且为正——**位置本身就是判据**。
5. 测试（`tests/test_case_bem_cummins_wigley.py`）新增：
   * `test_casepool_readme_labels_every_damping_extreme_with_its_dof`：从 `.npz` 独立重算 6 个 DOF 的
     带内最小值，要求渲染事实与之逐项相等、`min_damping_dof` 等于 argmin 的 DOF、README 第 4 节那行
     的两个数分别等于 surge / pitch 自己的最小值、第 2.1 节的裸数字旁边写着它的 DOF；并要求**通过 C0 的
     提交件每个 DOF 都是严格正数**（钉住 `np.diag` 那个「恰好 0」）。
   * `test_casepool_readme_bites_on_a_mislabelled_dof`：负控——把 yaw 的值塞进 `pitch`，渲染必须随之改变。
   * `test_casepool_readme_does_not_restate_the_pre_fix_state` 里把 `fig_c0_negative_controls.png`
     一并纳入「文档点名的图必须真的在树里」的清单（旧清单漏了这一张）。

## 4. 验证

| 检查 | 结果 |
| --- | --- |
| 新测试在**旧**渲染器上 | 2 failed（KeyError: `per_dof_min` → 断言确实咬住旧实现） |
| 新测试在**新**渲染器上 | 5 passed（`-k readme`） |
| `render == docs/.../README.md` 逐字相等 | passed（README 用 `--write` 重新渲染） |
| `tests/test_case_bem_cummins_wigley.py` 全文件 78 用例 | **78 passed**（两段本机跑：7 个重锚点用例 94.09 s + 其余 71 个 18.93 s，单进程） |
| 相邻文档门（`tests/test_doc_claim_audit.py`） | 3 failed，**在干净 main 上同样 3 failed**（`new unguarded document: AUDIT_RESTAT_CALIBRATION_20261113.md` 等），与本单无关；且无 casepool 条目 |

## 5. 注意点 / 局限

* 本单**只**修「数字与 DOF 的对应关系」这一层，没有重解任何 BEM，也没有改闸门判据；
  `case_result.json` / `sweep_*.npz` 一个字节都没动，`CASE.md` 未变（渲染源未变）。
* `min over band` 对健康工况天然是「带底小正数」（B(ω→0)→0），把它当强度指标读会误导；
  正确读法是「有没有负值 + 负值出现在带内还是带边」。README 现在把这句写进正文。
* n315 数值栈为何坏掉（`NODE-N315-NUMERICAL-VALIDITY`）仍未归因，本单不涉及。
* 说明书里的 C0/C3/C5/C5c 读数仍由 `case_result.json` 注入，其正确性由 m21960 的门负责，不在本单范围。

## 6. 验证回执

* 本机两段跑完全文件：`-k "anchor or added_mass_inf or anchored_kk or production_artifact"` → 7 passed (94.09 s)；
  `-k "not (...)"` → 71 passed (18.93 s)。合计 78 passed / 0 failed。
* 全文件一次跑超过本机 120 s 门限，故按上表分段；同一条命令也已递交集群（cj4586 为中间态快照、
  cj4587 为终态快照），递交时两份作业仍在排队（队列前 ~90 单），因此上面的结论取自本机分段执行，
  集群回执不作数、也不冒充。
* 报告已发布：`/tensor/reports` id=20723（`tools/post_report_m23091.py`，图 `docs/m23091/dof_attribution.png`）。