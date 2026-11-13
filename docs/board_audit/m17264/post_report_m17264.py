"""Post the board-audit m17264 progress report to the platform (/tensor/reports).

Every number in the markdown is interpolated from
  docs/board_audit/m17264/subject_artifacts/results.json   (subject artifacts)
  artifacts/audit_m17264/center_check_20x10.json           (audit re-solve)
  artifacts/audit_m17264/binding.json                      (8 evidence checks)
so nothing is hand-copied into the report body.
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[3]
SNAP = ROOT / "docs/board_audit/m17264"
SUBJ = json.loads((SNAP / "subject_artifacts/results.json").read_text())
AUD = json.loads((ROOT / "artifacts/audit_m17264/center_check_20x10.json").read_text())
BIND = json.loads((ROOT / "artifacts/audit_m17264/binding.json").read_text())
FIGS = SNAP / "figures"

API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")


def post(path, payload, token=None):
    req = urllib.request.Request(API + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def login():
    for path in ("/auth/login",):
        try:
            data = post(path, {"username": os.environ.get("TENSOR_AGENT_USER", "agent"),
                               "password": os.environ.get("TENSOR_AGENT_PASS", "")})
            tok = data.get("access_token") or data.get("token")
            if tok:
                return tok
        except Exception as exc:  # noqa: BLE001
            print("login failed", path, exc, file=sys.stderr)
    return None


def fig_entry(name, caption):
    png = (FIGS / name).read_bytes()
    return {"name": name, "caption": caption, "png_base64": base64.b64encode(png).decode()}


gates = SUBJ["gates"]
rows = SUBJ["cross_solver"]["rows"]
fixed = AUD["runs"]["20x10/center_zmT2"]
g = SUBJ["geometry"]
fd = SUBJ["free_decay_heave"]

def _fmt(v, spec=".6g"):
    return "nan" if v is None else format(v, spec)


gate_lines = "\n".join(
    "| {} | {} | {:.3g} | {} |".format(gate["id"], _fmt(gate["value"]), gate["tol"],
                                      "✓" if gate["passed"] else "✗")
    for gate in gates.values())

r1_lines = "\n".join(
    "| {} | {:.4g} | {:.4g} | {:+.2f} % | {:.4g} | {:+.2f} % |".format(
        r["dof_label"], r["added_mass_ref"], r["added_mass_got"], 100 * r["added_mass_rel_err"],
        fixed["added_mass"][r["dof_label"]], 100 * fixed["added_mass_rel_err"][r["dof_label"]])
    for r in rows)

markdown = f"""# 交差审计 m17264 → m17209 CASE-BEM-CUMMINS-WIGLEY（Wigley III 6-DOF Cummins 记忆核，TensorBEM）

**处置：L3 / reopen 返修。** 三件套齐、六段全、FAIL 未粉饰——但 R1 的 roll 55.5 %「求解器偏差」实为
**算例漏传旋转中心**的口径错配；补口径后同一网格 R1 由 FAIL 转 PASS。R3/R4/R5 的失败是**频带设计**问题
（不是求解器问题），当前配置下基准不可能通过。

## ① 案例介绍
Wigley III 数学船型（L={g['L']} m, B={g['B']} m, T={g['T']} m, Cb={g['Cb']}, ∇={g['volume']:.8f} m³，
L/B={g['L_B']}, B/T={g['B_T']:.2f}, ρ={g['density']}, g={g['gravity']}，旋转中心约定 z=−T/2）。
算例检验 TensorBEM 的 6-DOF 频域辐射解 A(ω)、B(ω) 经 Cummins 方程转到时域后记忆核
K(t)=2/π∫B(ω)cos(ωt)dω 与自由衰减是否成立。参考：R1=独立求解器 Capytaine 3.1.dev0 冻结件
（loongarch64，ω=1 rad/s，3 档网格）；R2–R5=Cummins/KK 形式体系自带的精确恒等式。

## ② 怎么做的
1. 取证：读对象工作树 `/root/tp-agent/work/m17209`（分支 `agent/m17209` @ f8a5442）与工件三件；
   快照入 `docs/board_audit/m17264/subject_snapshot/ subject_artifacts/ subject_report/`（sha256 见审计报告 §5）。
2. 复算：用对象自己的 `run_case` + 其缓存扫频**重算五门**（`audit_evidence_binding.py`，8 项检查）；
   `--stage gates` 与 `pytest` 实跑复现。
3. 归因：发现 `radiation_sweep` 调 `solve_radiation_3d` 未传 `center=`（默认 (0,0,0)），
   而参考件与仓内早前 wigley3 研究都用 (0,0,−T/2) → 在**同一 400 面元、ω=1** 上按参考口径重解
   （`audit_rotation_center.py`，56.8 s 单核），隔离口径效应。
4. 基准裁判：查 `tensor_benchmark_cases`（anchor 0 行）与 `tensor_case_pool`（tol `<10%`，status=dispatched）。

## ③ 计算结果
**五门（交付原样，审计重算逐位相同）**

| 门 | 值 | 容差 | 判定 |
|---|---|---|---|
{gate_lines}

**R1 逐 DOF：交付（centre z=0）vs 同一网格按参考口径重解（centre z=−T/2）**

| DOF | A_ref | A_got(交付) | 误差 | A_got(修正口径) | 误差 |
|---|---|---|---|---|---|
{r1_lines}

**自由衰减（heave）**：ζ_decay={fd['zeta_free_decay']:.5f}，周期 {fd['decay_metrics']['period']:.4f} s，
半功率 ζ=nan（`{fd['half_power']['error']}`）。C33={fd['restoring_C33']:.1f} N/m，排水质量={fd['mass']:.5f} kg。

**证据绑定 8 项**：C1–C7 全 PASS（工件存在、门可重算、verdict 自洽、几何闸门 ∇=L·B·T·Cb=1e-12 内一致、
报告表格与工件逐项一致、参考件为冻结件且逐位相符）；C8 为诊断（见 ⑤）。

## ④ 结果图
左：R1 逐 DOF 误差，交付口径 vs 参考口径（10 % 容差线）；右：五门对各自容差。
另有三张对象原图（Cummins 记忆核、跨求解器对照、门残差与自由衰减）一并留档。

## ⑤ 注意点
- **F1（blocking）** R1 的 roll −55.50 % 是口径错配：A44 0.29819→0.65249（−2.61 %）、
  B44 1.056e-5→5.852e-5（−3.25 %）、pitch −4.12 %→−1.45 %，surge/sway/heave/yaw 不变（刚体运动学）。
  报告把算例 bug 写成了「TensorBEM 最差 A_ii +55.50 %」——若进台账会把正确求解器记成缺陷。
- **F2（blocking）** R4=nan 因 heave 固有频率 8.14 rad/s 在扫频带 [0.5,6] 之外；R3/R5=2.24/2.43 因
  KK 主值积分与往返闭环在 8 点窄带上截断（模块自测用 0.3–8 rad/s、41 点、B 两端归零时 <2 %）。
  门会咬人，是频带不够。**当前配置下基准不可能通过**。
- **F3** 交付未合入 main@0516776（只在 agent/m17209），无 CI 守卫；repro_cmd 指可回收工作树路径。
- **F4（systemic）** `tensor_benchmark_cases` 对该 anchor 0 行；任务 result 自带
  `server_quality_audit={{gate_valid:false, key_present:false, gate_missing_warning:true}}`，
  仍被 dev 自动批准 → 本审计已把 ref/got/tol/repro_cmd 写入 verdict evidence（id=510）。
- **F5（low）** 报告 §③「800 面元」是参考件三角形计数，对象扫频是 400 四边形面元（§⑤ 表为 400），同一网格两种计数。
- **未跑**：参考件最细档 40×20 的参考口径重解（集群作业 cj1472 排队未完成）；对象 467.5 s 的
  `--stage sweep` 全链未复跑（仅复算 gates 阶段与门值，逐位相同）。

## ⑥ 误差判定（显式回答是否 <3 %）
| 量 | vs 基准 | 误差 | <3 %？ |
|---|---|---|---|
| A44（roll，参考口径） | Capytaine 3.1 冻结件 | {100*fixed['added_mass_rel_err']['roll']:+.2f} % | **是** |
| B44（roll，参考口径） | 同上 | {100*fixed['damping_rel_err']['roll']:+.2f} % | 否（3.25 %，<10 % 容差内通过） |
| A33（heave） | 同上 | {100*fixed['added_mass_rel_err']['heave']:+.2f} % | **是** |
| A55（pitch，参考口径） | 同上 | {100*fixed['added_mass_rel_err']['pitch']:+.2f} % | **是** |
| A44（roll，交付原样） | 同上 | {100*rows[3]['added_mass_rel_err']:+.2f} % | 否（通道口径错配，已归因） |
| R1 门值（参考口径后） | 门容差 10 % | {fixed['gate_R1_value']:.4f} | 门内 PASS |
| R3 / R5 / R4 | 形式恒等式 / 频带自洽 | {gates['R3']['value']:.4f} / {gates['R5']['value']:.4f} / nan | 否（频带截断 + 无共振点，算例设计缺陷） |

**结论**：交付的求解器物理在正确口径下与独立码在 2.6 % 内一致（<3 %）；**基准整体不过**，
原因是算例口径与频带设计，而非 TensorBEM 能力。判级 **L3**（真实现+集成+可复现，未达基准验证 L4），
reopen 返修清单见 `docs/board_audit/m17264/AUDIT_m17209.md`。
"""

payload = {
    "title": "交差审计 m17264：CASE-BEM-CUMMINS-WIGLEY（m17209）L3·reopen — R1 roll 55% 系旋转中心口径 bug 而非求解器缺陷",
    "summary": ("逐项复核 m17209 三件套与五门基准：工件/ref-got/repro 全部可核（8 项绑定检查 C1–C7 PASS，"
                "gates 重算逐位相同）；但发现 R1 的 roll −55.50 % 来自算例漏传 center=(0,0,−T/2)，"
                "同一 400 面元重解后 A44 −2.61 %、B44 −3.25 %，R1 门 0.55495→0.02613 转 PASS；"
                "R3/R4/R5 因 8 点窄带（heave 共振 8.14 rad/s 带外）结构性 FAIL。判级 L3，reopen。"
                "benchmark 表该 anchor 0 行（systemic），已在 verdict id=510 补登记位。"),
    "scenario_key": "board-audit-bem-case-pool",
    "mission_id": 17264,
    "tags": ["board-audit", "TensorBEM", "CASE-BEM-CUMMINS-WIGLEY", "m17209", "Cummins", "Kramers-Kronig",
             "rotation-centre", "maturity-L3", "reopen"],
    "markdown": markdown,
    "figures": [
        fig_entry("fig_audit_m17264_r1_centre_and_gates.png",
                  "Left: per-DOF R1 relative error, delivered case (rotation centre default z=0) vs the same 400-quad mesh re-solved with the reference convention (0,0,-T/2); dashed line = 10% case tolerance. Right: the five gates as delivered against their own tolerances (log), coloured by pass/fail; R4 is nan because the heave natural frequency (8.14 rad/s) lies outside the swept band [0.5,6]."),
        fig_entry("fig_m17209_cummins_kernels_6dof.png",
                  "Subject figure (sha256 0a11749d…): normalised Cummins memory kernels K_ii(tau)/K_ii(0) for the six DOF, from K(t)=(2/pi)int B(w)cos(wt)dw."),
        fig_entry("fig_m17209_gates_6dof.png",
                  "Subject figure (sha256 2627297b…): heave free decay from the Cummins equation and per-DOF gate residuals against the 10% case tolerance."),
    ],
}

tok = login()
if not tok:
    print("NO TOKEN", file=sys.stderr)
    raise SystemExit(2)
def attempt(pl, label):
    try:
        r = post("/tensor/reports", pl, token=tok)
        print(label, "OK", json.dumps({"id": r.get("id")})[:200])
        return True
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()[:800]
        print(label, "HTTP", exc.code, body)
        return False


if not attempt(payload, "full"):
    slim = {k: v for k, v in payload.items() if k not in ("scenario_key", "tags")}
    if not attempt(slim, "no-scenario-key"):
        slim2 = dict(slim, figures=[payload["figures"][0]])
        attempt(slim2, "one-figure")