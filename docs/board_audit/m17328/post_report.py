#!/usr/bin/env python3
"""Publish the m17328 board-audit report (tensor_reports) with the audit figure embedded."""
from __future__ import annotations

import base64
import json
import os
import pathlib
import re
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
HERE = pathlib.Path(__file__).resolve().parent
W = pathlib.Path("/root/tp-agent/work/m17277")
STUDY = json.loads((W / "docs/case_qtf_drift_closed_form/study.json").read_text())
AUD = json.loads(pathlib.Path("/tmp/audit_m17328/independent_qtf.json").read_text())
CMP = json.loads((HERE / "rerun_compare.json").read_text())
med = STUDY["levels"][1]

rows = "\n".join(
    f"| ({p['i']},{p['j']}) | {p['ka_i']:.2f} | {p['ka_j']:.2f} | {p['bem_abs']:.4e} | "
    f"{p['ref_abs']:.4e} | {p['abs_error']*100:.2f}% | {'✅' if p['abs_error'] <= 0.1 else '❌'} |"
    for p in med["pairs"])
checks = "\n".join(
    f"| {r['ka']:.1f} | {r['closed_form']:.6e} | {r['audit_near_field']:.6e} | "
    f"{r['rel_diff_vs_audit_near']:.2e} | {r['audit_far_field_maruo']:.6e} | "
    f"{r['rel_diff_vs_audit_far']:.2e} |"
    for r in AUD["diagonal_anchor_independent_check"])

MD = f"""## 1. 案例介绍（物理背景/几何/参数表）
交差审计单 m17328，对象 = m17277（CASE-BEM-QTF-DRIFT，已 succeeded）。二阶波浪力里最难验证的是**差频** QTF
`T_xx(ω_i,ω_j)`（非对角）；TensorBEM 此前只对角有绝对参考。被审交付新增**机器精度闭式**参考（固定底置圆柱、
MacCamy–Fuchs 一阶模态 + Pinkster 近场二次压力装配、解析深度积分），并据此给面板 BEM 打分。
参数：ρ=1025 kg/m³，g=9.81 m/s²，a=1.0 m，ka=(0.4,0.6,0.8)，头浪，kH=12（深水），网格 coarse 336 / medium 696 面板。
裁判：`tensor_benchmark_cases` 对该锚点原为 **0 行**（登记缺口），本审计补登记行 id=53，容差 `<10%`（算例池 id=176 + 被审单 goal）。

## 2. 怎么做的（方法/流程/关键设置/工具链）
① 取证：读 m17277 结果/报告（tensor_reports id=16291）与工作树 `docs/case_qtf_drift_closed_form/`，逐项核三件套（run_artifacts 8 件 sha256、ref-vs-got 逐对数字、repro_cmd 实跑）。
② 独立复核：**不 import `tensorbem.cylinder_drift`**，用 scipy `jv/jvp/hankel1/h1vp` 重建 MF 模态、自建 4096 点方位求积，复算一阶场/无通量 BC/方位积分 I1,I2/对角近场精确解/Maruo 远场/长波渐近系数（`docs/board_audit/m17328/audit_independent_qtf.py`）。
③ 复现：在被审 commit 上按文档命令重跑 benchmark（输出到 /tmp，不动被审工作树）。
④ 判级 L0–L5 逐级核后落表 `tensor_maturity_verdicts`，并跑 `maturity_gate.judge()` 干跑预演。

## 3. 计算结果（程序化注入，零手抄）
**最细可用网格 medium（696 面板）逐对 ref vs got（10% 判据）**

| (i,j) | ka_i | ka_j | BEM \\|T\\| [N/m²] | 闭式 \\|T\\| [N/m²] | rel err | <10% |
|---|---|---|---|---|---|---|
{rows}

- 非对角：median **{med['offdiagonal_median_error']*100:.2f}%** / max **{med['offdiagonal_max_error']*100:.2f}%** > 10% ⇒ **FAIL（honest）**
- 对角：median {med['diagonal_median_error']*100:.2f}%；coarse→medium 对角 6.59%→4.56%（降）而非对角 20.41%→20.62%（不降）
- 参考侧门 G1–G4 全过：4.25e-15 / 1.42e-15 / 0.118% / 2.45e-14

**审计独立复核（自己的 MF 级数）**

| ka | 闭式对角 | 审计近场 | rel | 审计 Maruo 远场 | rel |
|---|---|---|---|---|---|
{checks}

另：无通量 BC 相对残差 ≤1.1e-16；审计场 vs 交付模块 ψ 差 ≤7.1e-16；I1/I2 差 ≤3.7e-16；长波系数 ka=0.002 时 3.084251 vs 5π²/16=3.084251（6.7e-7）。

## 4. 结果图
图 1（左）medium 网格逐对闭式 vs BEM 幅值柱状图（标注逐对相对误差）；（右）coarse→medium 对角/非对角误差中位与 10% 判据线。
审计重跑状态：{CMP.get('status')}。

## 5. 注意点（踩过的坑/局限/适用范围）
- **非对角没有独立锚**：近场与 Maruo 两条精确路线只能验证对角；"缺口在 BEM 侧"是推断不是证明（只能查 Hermitian 8.9e-16 与 ka 连续性）。闭式非对角绝对值 **WITHHELD**，第二轮建议引入 capytaine 三方对照。
- 被审单 goal 要求 ref 从文献原表/手册查得并引原页；交付改为自算闭式并在报告 §1 如实声明（无外网），属方法偏差而非造假，但外部印刷表锚缺失 ⇒ L5 不成立。
- 复现瑕疵：文档 repro_cmd 未带 `--cache`（交付件 `solve_seconds_total`=0.235 s 实为缓存重载，审计实测真实求解 26.8 s/次）也未带 `--no-depth-sweep`（本次审计重跑 emit 超集）。
- 同侧同缺项：两侧都只含二次压力项、不含二阶势项，对比同范围成立，但绝对 QTF 只是下界。
- 吃水扫描未跑（交付件 §5 已如实标注），以 e^{{-kH}}≈6.1e-6 解析界论证；审计重跑顺带补跑该段。

## 6. 误差判定（逐项，显式回答）
| 量 | 误差 | 判据 10% | 是否 <3% |
|---|---|---|---|
| 非对角 QTF 最大（medium） | **{med['offdiagonal_max_error']*100:.2f}%** | ❌ | 否 |
| 非对角 QTF 中位（medium） | {med['offdiagonal_median_error']*100:.2f}% | ❌ | 否 |
| 对角 QTF 中位（medium） | {med['diagonal_median_error']*100:.2f}% | ✅ | 否（4.56%） |
| 闭式 G1 对角恒等 | 4.25e-15 | ✅ | 是 |
| 闭式 G2 Maruo 远场 | 1.42e-15 | ✅ | 是 |
| 闭式 G3 长波律（ka=0.05） | 0.118% | ✅ | 是 |
| 审计独立长波系数（ka=0.002） | 6.7e-7 | ✅ | 是 |

**结论：算例判据（非对角 <10%）不成立，honest FAIL；关键量 vs 基准 >3%，不粉饰。** 判级 **L3**（真实现 + 系统集成；L4 因判据 FAIL 与外部锚缺失不成立，与 m17242 先例同口径）。verdict 落表 id=530（gate_state=pending），裁判行 id=53 status=failed。
"""


def api(method, path, payload=None, token=None):
    req = urllib.request.Request(BASE + path, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    data = json.dumps(payload).encode() if payload is not None else None
    with urllib.request.urlopen(req, data=data, timeout=120) as r:
        return json.loads(r.read() or b"{}")


def main() -> None:
    tok = api("POST", "/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                                      "password": os.environ["TENSOR_AGENT_PASS"]})["access_token"]
    png = base64.b64encode((HERE / "audit_qtf_drift.png").read_bytes()).decode()
    payload = {
        "title": "交差审计 m17277·CASE-BEM-QTF-DRIFT：闭式差频 QTF 真机器精度，BEM 非对角 20.6%/32.1% 超 10% 判据（honest FAIL）→ 判级 L3",
        "summary": ("m17277 三件套取证通过（study.json/tables.md/4 PNG，sha256 入账；tables.md 与 study.json 逐字节一致）；"
                    "裁判行原缺 → 补登记 id=53（<10%，status=failed）；审计独立 MF 级数复算：无通量 BC≤1.1e-16、"
                    "对角 vs 自建近场/Maruo 双路线 ≤1.8e-15、长波系数 3.084251→5π²/16(6.7e-7)；"
                    "BEM 侧非对角 medium 696 面板 median 20.62%/max 32.07% 且加密不降 ⇒ 判据 FAIL（自报一致，无粉饰）；"
                    "非对角无独立锚 → 闭式非对角 WITHHELD；判级 L3，verdict id=530。"),
        "scenario_key": "tensorbem-qtf-drift-closed-form",
        "mission_id": 17328,
        "tags": ["board-audit", "TensorBEM", "CASE-BEM-QTF-DRIFT", "second-order-drift", "QTF", "honest-FAIL", "m17277"],
        "markdown": MD,
        "figures": [{"name": "audit_qtf_drift.png",
                     "caption": "左：CASE-BEM-QTF-DRIFT medium(696) 逐对 |T| 闭式 vs 面板 BEM（标注相对误差）；右：coarse→medium 对角/非对角误差中位与预注册 10% 判据线。",
                     "png_base64": png}],
    }
    out = api("POST", "/tensor/reports", payload, tok)
    print(json.dumps(out)[:400])


if __name__ == "__main__":
    main()