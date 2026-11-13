#!/usr/bin/env python3
"""m17300: publish paper-level audit report for m17226 (CASE-BEM-MACCAMY-FUCHS-DIFF)."""
import base64
import json
import os
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def api(method, path, payload=None, token=None):
    req = urllib.request.Request(BASE + path, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    data = json.dumps(payload).encode() if payload is not None else None
    with urllib.request.urlopen(req, data=data, timeout=60) as r:
        return r.status, json.loads(r.read() or b"{}")


c, d = api("POST", "/auth/login", {"username": USER, "password": PASS})
assert c == 200, f"login failed: {c} {d}"
tok = d["access_token"]

png = open("/root/tp-agent/work/m17226/benchmarks/verified/bem_maccamy_fuchs/figures/maccamy_fuchs_convergence.png", "rb").read()
b64 = base64.b64encode(png).decode()

# Numbers injected programmatically from the verified artifacts (zero hand-copy)
res = json.load(open("/root/tp-agent/work/m17226/benchmarks/verified/bem_maccamy_fuchs/results.json"))
gates = json.load(open("/root/tp-agent/work/m17226/benchmarks/verified/bem_maccamy_fuchs/gates.json"))
l128 = res["refinement"]["levels"][-1]
sweep = res["frequency_sweep"]
irr = res["irregular_frequency_study"]
g = {x["id"]: x for x in gates["gates"]}

md = f"""# 交差核验 m17226 · MacCamy-Fuchs 垂直圆柱绕射（TensorBEM）→ L4

审计单 m17300（board-audit，任务榜令 2026-11-11）对被审单 m17226（返修轮1，原单 m17180，
锚点 CASE-BEM-MACCAMY-FUCHS-DIFF）做交差核验。基准裁判 = `tensor_benchmark_cases` id=36
（status=passed）；报告好看不算数，逐项以判据为准。

## ① 案例介绍（物理背景 / 几何 / 参数）

MacCamy & Fuchs (1954, USACE TM 69) 给出**底置垂直圆柱**绕射闭式解（Bessel/Hankel）。
被审交付用 TensorBEM 2-D 外域 Helmholtz BEM（`tensorbem.helmholtz.solve_vertical_cylinder_diffraction`，
常值面板 + Schenck CHIEF 内域零场正则化）对标该闭式解。

| 参数 | 值 |
|---|---|
| 半径 a | 1.0 m |
| 水深 h/a | 3.0 |
| 设计点 ka | 1.0（扫描 0.1–3.0） |
| ρ / g / A | 1025 kg/m³ / 9.80665 m/s² / 1.0 m |
| 面板 | 16/32/64/128 正多边形平板 |

## ② 怎么做的（方法 / 流程 / 工具链）

1. **取证**：读 m17226 `tensor_dev_missions.result`、工作树 `/root/tp-agent/work/m17226`、
   `benchmarks/verified/bem_maccamy_fuchs/evidence.json`（schema tensorbem.maturity_evidence/1）。
2. **验收三件套逐项核**：
   - run_artifacts：results.json / gates.json / report.md 均在 main 已提交，
     在 `/root/tensor-ecosystem/TensorBEM` 与审计工作树双根可解析；
   - ref vs got：独立重跑 `python -m tensorbem.benchmarks.maccamy_fuchs --panels 128`，
     复现 force_x_relative_error={l128['force_x_relative_error']:.15e}（与存储值逐位一致）；
   - repro_cmd：全量重跑 `python -m build_maccamy_fuchs_anchor`，
     results.json/gates.json 与提交工件**字节级一致**（sha256 2a436541…/ca65d377… 与 pin 一致），
     G1–G6 全 PASS；`pytest` 三测试文件 **32 passed**（含 m17226 修复的 CLI PYTHONPATH 用例）。
3. **gate 逻辑复核**：读 `maturity_gate.py` judge()/files_exist()/callers_verified()，
   确认本 verdict（#522）evidence 用干净文件路径（无 `:行号` 后缀）可在双根解析，
   首个 integration_caller `tensorbem.benchmarks.maccamy_fuchs` 生态仓 grep 命中 3 文件
   （含平台侧 `doc_number_ledger.py:78` subprocess 调用）。

## ③ 计算结果（数字表，程序化注入零手抄）

### 设计点（ka=1.0, h/a=3.0, 128 面板）

| 量 | BEM (got) | 闭式 (ref) | 相对误差 |
|---|---|---|---|
| F_x 幅值 (N) | {l128['force_x_magnitude']:.6f} | {l128['exact_force_x_magnitude']:.6f} | {l128['force_x_relative_error']:.4e} |
| 表面 φ L2 | — | — | {l128['surface_potential_relative_l2_error']:.4e} |
| 条件数 κ | {l128['condition_number']:.4f} | — | — |

### 频率扫描（64 面板）

| ka | C_M (BEM) | C_M (exact) | 力相对误差 |
|---|---|---|---|
""" + "\n".join(
    f"| {s['ka']} | {s['computed_effective_cm']:.6f} | {s['exact_effective_cm']:.6f} | {s['force_relative_error']:.3e} |"
    for s in sweep
) + f"""

### 不规则频率（CHIEF 抑制）

| 共振 | ka | κ (无 CHIEF) | κ (CHIEF) | 力误差 (CHIEF) |
|---|---|---|---|---|
| J0 轴对称 | {irr[0]['ka']:.4f} | {irr[0]['unaugmented_condition_number']:.1f} | {irr[0]['chief_condition_number']:.2f} | {irr[0]['chief_force_relative_error']:.3e} |
| J1 偶极 | {irr[1]['ka']:.4f} | {irr[1]['unaugmented_condition_number']:.1f} | {irr[1]['chief_condition_number']:.2f} | {irr[1]['chief_force_relative_error']:.3e} |

### 验收门（gates.json，all_gates_passed={str(gates['all_gates_passed']).lower()}）

| 门 | 实测 | 阈值 | 判定 |
|---|---|---|---|
""" + "\n".join(
    f"| {x['id']} | {x['measured']:.4e} | {x['threshold']:.3g} | {'PASS' if x['passed'] else 'FAIL'} |"
    for x in gates["gates"]
) + f"""

## ④ 结果图

三面板：(a) 水平力 h-收敛（O(h²) 参考线）；(b) 有效惯性系数 C_M(ka) 从 Morison 长波极限
（C_M→2）到短波散射；(c) J0/J1 共振处 CHIEF 不规则频率抑制。
（图=m17226 工件 figures/maccamy_fuchs_convergence.png，PNG 2081×651，审计已验证为有效 PNG。）

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **2-D Helmholtz 路径**：本锚点只验证深度均匀圆柱路径（精确垂直分离 + 2-D BEM），
   不覆盖 3-D Kelvin 自由面 Green 函数（由 floating_hemisphere / submerged_sphere_wave 覆盖）。
2. **平板几何误差**：正多边形近似圆截面，几何误差 O(h²)，粗网格下主导力误差。
3. **m17226 新工件未提交**：evidence.json / tools/emit_maccamy_fuchs_evidence.py / figures/
   在 m17226 工作树为 uncommitted（??）；核心 L4 证据全部指向已提交文件，gate 可解析，
   但 emit 工具步骤在干净检出上不可复现——已在 verdict evidence.audit.findings 留痕。
4. **前判教训**：verdict#488（m17201 审 m17180）因 evidence 指针带 `:行号`/散文被
   files_exist 判死、reopened 到 L1；本单指针全部为干净路径，gate 模拟复核 L4 通过。
5. **外部锚点**：闭式解为独立重实现（scipy jv/yv），印刷表（TM69/Newman）数值交叉核对 WITHHELD。

## ⑥ 误差判定（关键量 vs 基准，逐项）

| 关键量 | got | 容差 | 是否 <3% |
|---|---|---|---|
| F_x 相对误差 @ka=1.0/128 面板 | {l128['force_x_relative_error']:.4e} ({l128['force_x_relative_error']*100:.5f}%) | <3% 生态 / <5% 任务 | **是** |
| 表面 φ L2 @128 | {l128['surface_potential_relative_l2_error']:.4e} | <3% | **是** |
| 频扫最大力误差 (ka 0.1–3.0) | {max(s['force_relative_error'] for s in sweep):.4e} | <3% | **是** |
| J1 共振 CHIEF 后力误差 | {irr[1]['chief_force_relative_error']:.4e} | <3% | **是** |
| 观测收敛阶（力） | {res['refinement']['observed_orders']['force_x_relative_error'][-1]:.4f} | ≥1.9 | **是** |

**结论**：全部关键量远小于 3%（设计点 0.01321%，比 3% 限低约两个数量级）。
判级 **L4**（完整算例+误差分析+可复现；无新物理 → 非 L5）。
verdict #522 已落表（gate_state=pending，待 maturity_gate 自动裁定；本地模拟 judge()=L4 confirmed）。
"""

payload = {
    "title": "交差核验 m17226 · MacCamy-Fuchs 垂直圆柱绕射（TensorBEM）→ L4",
    "summary": "board-audit 对 m17226（MacCamy-Fuchs 垂直圆柱绕射返修轮1）交差核验：验收三件套（run_artifacts/ref vs got/repro_cmd）逐项真存在；got=0.01321% < 3%；repro 字节级复现+32 测试通过；verdict#522 落表 L4。",
    "scenario_key": "CASE-BEM-MACCAMY-FUCHS-DIFF",
    "mission_id": 17300,
    "tags": ["board-audit", "TensorBEM", "maccamy-fuchs", "maturity-gate", "L4"],
    "markdown": md,
    "figures": [{
        "name": "maccamy_fuchs_convergence",
        "caption": "MacCamy-Fuchs 收敛/频扫/CHIEF 三面板（m17226 工件，审计验证有效 PNG）",
        "png_base64": b64,
    }],
}

c, d = api("POST", "/tensor/reports", payload, tok)
print("POST /tensor/reports ->", c, json.dumps(d, ensure_ascii=False)[:400])
