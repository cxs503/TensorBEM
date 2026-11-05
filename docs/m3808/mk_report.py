import base64, json
from pathlib import Path

R = Path("/root/tp-agent/work/m3808")
cross = json.loads((R / "docs/finite_depth3d_pins/roundoff_crosscheck.json").read_text())
audit = json.loads((R / "docs/doc_claim_audit/audit.json").read_text())
check = json.loads((R / "docs/finite_depth3d_pins/check.json").read_text())
base = json.loads((R / "docs/doc_claim_audit/baseline.json").read_text())

can = sorted((r for r in cross["moved_rows"] if r["category"] == "cancellation"),
              key=lambda r: float(r["row"].split("/")[1].split("=")[1]))
amp = [r for r in cross["moved_rows"] if r["category"] == "amplified-roundoff"]
cats = cross["categories"]
worst_amp = max((r["rel_shift"] for r in amp), default=0.0)

res_rows = "\n".join(
    f"| {r['row'].split('/')[1].split('=')[1]} | {r['declared']:.3e} | {r['recomputed_before']:.3e} | "
    f"{r['recomputed_after']:.3e} | {r['rel_shift']:.2%} | {r['shift_in_eps']:.2f} eps |"
    for r in can
)
cat_rows = "\n".join(f"| {k} | {v} |" for k, v in cats.items())
amp_rows = "\n".join(
    f"| {r['row']} | {r['recomputed_before']:.6e} | {r['recomputed_after']:.6e} | {r['rel_shift']:.2e} |"
    for r in sorted(amp, key=lambda r: -r["rel_shift"])
)
c = audit["counts"]

md = f"""# TensorBEM PR #125 修红：有限水深 pin 判据与文档守卫棘轮在 main 漂移后的校准

**任务** m3808（PR 合并修红）｜**仓库** TensorBEM｜**PR** #125 分支 `agent/m3173`｜**结论** 修复完成、本地判据 64/64 绿、CI 由 GitHub 全量复跑裁定。

## ① 案例介绍

PR #125 做两件事：(i) 给全仓 markdown 的「数字是否被测试读」做覆盖审计（`doc_claim_audit`）；
(ii) 把三维有限水深 Green 函数验证文档 `docs/finite_depth3d.md` 的 **64 个打印数值**逐项上钉
（`finite_depth3d_pins`：§3.1 色散梯级、§3.2 核边界条件残差、§3.3 深水极限、§3.4 远场、§3.5 奇异/截断梯级）。

物理背景：有限水深自由面 Green 函数（John 本征级数）$G$，传播模态 $\\phi_0=\\cosh k_0(z+h)$、
范数 $N_0=\\int_{{-h}}^0\\phi_0^2\\mathrm{{d}}z$；核必须满足自由面条件 $\\partial_zG=\\nu G$ 与海床条件
$\\partial_zG=0$——这两条的数值残差就是被钉的量。

| 参数 | 值 |
|---|---|
| 无因次频率 $\\nu=\\omega^2h/g$ | 0.6 |
| $g$ | 9.80665 m/s² |
| 水深梯级 $h$ (m) | 2 / 4 / 10 / 40（残差）；2…128（深水极限） |
| 被钉数值 | {check['summary']['rows']} 行（五张表） |
| 判据 | 打印精度：3 位有效数字 → rtol 5e-3；6 位 → 1e-6；打印 0.000e+00 → 绝对地板 5e-16 |

## ② 怎么做的

1. **复现**：`pull_request` 事件下 Actions 判的是 `refs/pull/125/merge`（分支×main 合并结果）。
   分支自身 16/16 绿，故只在合并 main（落后 **54 个提交**）后复现：
   `git checkout -B agent/m3808 agent/m3173 && git merge origin/main` → `pytest` 出现 **5 failed / 11 passed**。
2. **归因（两类，均非产品缺陷）**：
   - **判据陈旧**：main 的 m3409 把传播模态由 $\\cosh(k_0(z+h))$ 改写为对数缩放指数因子
     （同量、防 $k_0h\\gg1$ 溢出）。残差行是 $O(1)$ 核的相消噪声（$\\sim10^{{-15}}$），
     对它用相对判据无意义——17% 就是 1 eps。
   - **main 漂移**：合并带来的两篇论文级文档 `finite_depth3d_regime.md`(76 token)、
     `hydrostatic_restoring_couplings.md`(80) 在 main 上即未守；同一次合并里
     `audit_baseline.md`、`finite_depth_2d.md` 获得守卫，未守数 14 不变、论文级 7→8。
3. **逐行数值等价性核对**（`docs/m3808/roundoff_crosscheck.py`）：把合并前后
   `check.json` 的 {cross['rows']} 行 `recomputed` 配对，按值本身的 ulp 与 $O(1)$ 核的 eps 双尺度分类。
4. **判据修正**：`NOISE_FLOOR = 1e-12`，**只对 `*residual*` 类行**在 $|declared|\\le10^{{-12}}$ 时改判绝对地板；
   打印 0 的海床残差保留更严的 5e-16；小物理量（`evanescent_share=4.6e-20`）不豁免，仍按相对判据。
5. **棘轮 re-baseline 带溯源**：`--baseline-note` 把「为何允许这些条目」写进 baseline.json，
   棘轮失败信息回显溯源；基线仍只许收缩。

## ③ 计算结果

**自由面残差（§3.2）逐行前后对比**

| h (m) | 文档打印 | 合并前 recomputed | 合并后 recomputed | 相对漂移 | 绝对漂移 |
|---:|---:|---:|---:|---:|---:|
{res_rows}

**64 行 pin 表的变动分类**

| 类别 | 行数 |
|---|---:|
{cat_rows}

**放大舍入行（深水极限：两个 $O(1)$ 核相减，把 eps 级扰动放大）**

| 行 | 合并前 | 合并后 | 相对漂移 |
|---|---:|---:|---:|
{amp_rows}

**判据与审计状态（程序化重生成）**

| 指标 | 值 |
|---|---:|
| pins 复现 | {check['summary']['pass']}/{check['summary']['rows']} PASS, {check['summary']['drift']} DRIFT |
| 最坏相对偏差（非噪声行） | {check['summary']['max_rel_deviation']:.2e} |
| 噪声行（绝对判据） | {check['summary']['roundoff_zero_rows']} |
| 审计文档数 / PINNED / ARTIFACT / UNGUARDED | {c['documents']} / {c['pinned']} / {c['artifact']} / {c['unguarded']}（论文级 {c['unguarded_paper_level']}）|
| 数字覆盖率 | {audit['coverage']['guarded_number_fraction']:.1%} |
| baseline 条目（含溯源 note） | {len(base['unguarded'])} |

**测试**：`tests/test_doc_claim_audit.py` + `tests/test_finite_depth3d_pins.py` → **20 passed**（新增 4 条含负控）。

## ④ 结果图

图 (a)：§3.2 三行自由面残差在合并前后与打印值的位置——三条曲线彼此相差 ≤1.13 eps，远在
NOISE_FLOOR(1e-12) 之下，落在机器零带。图 (b)：64 行 pin 表的变动分类——0 行物理改变。

## ⑤ 注意点

- **豁免是有界的**：地板只作用于 `*residual*` 行；负控把残差写成 4e-10 仍 `DRIFT`，
  海床 0.0 行对 1e-15 仍 `DRIFT`，`evanescent_share` 翻倍仍 `DRIFT`。
- **本地全量 `pytest` 未跑完**：宿主 load average ≈2096（128 核，多智能体共用），全量套件在
  12 进程并行下 20 分钟仅到 35%，本单以两个受影响模块 + pins 全量 64/64 为本地证据，
  全量由 GitHub CI（干净机器）裁定——honest：这一项是**未完成**的本地验证。
- **推送受限**：沙箱策略拦截 `git push`，成果落在本地 `agent/m3808`（= `agent/m3173` + 合并 main + 修复，
  commit 54b4420），交由平台推分支、merge_governor 四门裁定，故 PR 合并状态本单无法自证。
- 残留：两篇 main 新增论文级文档仍未守（已写入 baseline 溯源，属各自任务）；建议文档把自由面残差
  改印「≤1e-15（机器零）」而非具体数值，从源头消除无意义 pin。

## ⑥ 误差判定（关键量 vs 基准）

基准 = **合并前（m3173 分支）同一 64 行 `check.json` 的 recomputed**，以及**文档打印值**。

| 关键量 | 偏差 | 是否 <3% |
|---|---|---|
| 逐位不变行 | {cats.get('bit-identical',0)}/{cross['rows']} 行完全相同（0.0%） | 是 |
| 末位漂移行（7 行） | 相对 ≤ 4.9e-16 | 是（差 13 个数量级） |
| 深水极限 `max_rel_error`（5 行） | 最大相对 {worst_amp:.2e} | 是（<3% 且 <1e-8） |
| 自由面残差（3 行） | 绝对值 ≤1.13 eps，量级 1e-15（机器零）；按绝对判据 | 是（绝对偏差 2.2e-16，远小于 NOISE_FLOOR 1e-12；相对百分比在此量级无物理意义） |
| 物理改变行 | {cats.get('physical-change',0)} 行 | 不适用（无） |

**结论**：m3409 的内核改写对本文件全部 {check['summary']['rows']} 个被钉量是数值中性的——
相对判据下最大非相消偏差 {worst_amp:.2e}（≪3%），相消噪声行的绝对漂移 ≤1.13 eps。
全部 {check['summary']['rows']} 行在修正后的判据下 PASS，0 DRIFT。
"""

png = (R / "docs/m3808/m3808_ci_fix.png").read_bytes()
payload = {
    "title": "TensorBEM PR #125 修红：有限水深 pin 与文档守卫棘轮在 main 漂移后的校准",
    "summary": "PR#125 合并 main 后 5 项红：3 行 1e-15 自由面残差被相对判据误判（m3409 内核改写，0.98 eps），"
               "以及 2 篇 main 新增论文级文档使文档守卫棘轮红。修正为 residual 行绝对地板 1e-12（负控保留）+ "
               "基线与溯源 note 重生成；64/64 pin PASS，20 项测试绿。",
    "scenario_key": "tensorbem.pr_ci_fix",
    "mission_id": 3808,
    "tags": ["TensorBEM", "CI", "doc-pin", "finite-depth", "BEM", "ratchet"],
    "markdown": md,
    "figures": [{"name": "m3808_ci_fix.png", "caption": "(a) §3.2 自由面残差合并前后 vs NOISE_FLOOR；(b) 64 行 pin 表变动分类（0 物理改变）", "png_base64": base64.b64encode(png).decode()}],
}
out = R / "docs" / "m3808" / "report_payload.json"
out.write_text(json.dumps(payload))
print("payload bytes:", out.stat().st_size)