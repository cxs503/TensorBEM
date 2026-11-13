#!/usr/bin/env python3
"""Publish the m17318 audit report (six sections, numbers injected programmatically)."""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")

HERE = Path(__file__).resolve().parent
ART, REFS, FIGS = HERE / "artifacts", HERE / "references", HERE / "figs"


def token() -> str:
    req = urllib.request.Request(
        f"{BASE}/auth/login",
        data=json.dumps({"username": USER, "password": PASS}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read())
    return d.get("access_token") or d["token"]


def levels() -> list[dict]:
    out = {}
    for p in (ART / "auditor_rerun_sub1.json", ART / "auditor_rerun_sub2.json",
              REFS / "sphere_added_mass_sub3.json", REFS / "sphere_added_mass_sub4.json"):
        d = json.loads(p.read_text())
        out[int(d["panels"])] = d
    return [out[k] for k in sorted(out)]


def md() -> str:
    lv = levels()
    tbl = "\n".join(
        f"| {d['panels']} | {d['exact_added_mass']:.6f} | {d['added_mass']:.6f} | "
        f"{d['added_mass_relative_error']:.6e} | {'✓' if d['added_mass_relative_error'] < 0.03 else '✗'} | "
        f"{d['normal_derivative_relative_l2_error']:.3e} | {d['residual_l2']:.2e} |"
        for d in lv)
    finest = lv[-1]["added_mass_relative_error"]
    first_pass = min(d["panels"] for d in lv if d["added_mass_relative_error"] < 0.03)
    return f"""# 交差审计：m17262 / CASE-BEM-SPHERE-ADDED-MASS（TensorBEM）— 判定 L3

## ① 案例介绍
物理：无界无粘流体中匀速平动球体的附加质量（船舶耐波/操纵惯性系数的球体极限）。几何=单位球 R=1 m，
离散为 flat-faceted icosphere（subdivision 1..4 → 80/320/1280/5120 面板）。流体 ρ=1025 kg/m³，U=1 m/s，
外域 Dirichlet 问题 ∇²φ=0, ∂φ/∂n=-U·n（球面物面），附加质量由边界能积分 ∂φ/∂n 求得。
基准（唯一裁判）：`tensor_benchmark_cases.anchor_slug=CASE-BEM-SPHERE-ADDED-MASS`，容差 <3%。
解析锚：m_A = ½ρV（Lamb 1932 *Hydrodynamics* 6th ed. §6.10；Newman 1977 eq. 4.132 的球体极限），
机器交叉校验 `tensorbem.analytic.ellipsoid_added_mass([R,R,R],ρ)` 与 `0.5ρ·4/3πR³` 相对差 <1e-9。
审计对象 m17262（m17176 的返修轮1）。本单不复算物理，只核三件套+数值+可复现。

## ② 怎么做的
1. 读 `tensor_dev_missions` id=17262 的 goal/result/changed_files/git_status；读对象工作树
   `/root/tp-agent/work/m17262` 的 `runs/m17262/`、`src/tensorbem/benchmarks/sphere_added_mass.py`、`tests/`。
2. 从 DB 取基准行的 `metric` 文本，正则抽出 4 个 level 的 got 数值（零手抄），与工件
   `added_mass_relative_error` 逐项比对（`insert_verdict.py`）。
3. 审计侧独立复算：用对象模块跑 sub1(80p)/sub2(320p)（`repro/repro_audit.sh`，含快照 sha256 一致性校验）。
4. 在对象工作树内跑对象自带测试 `pytest tests/test_sphere_added_mass_benchmark.py -q`。
5. 机器自检落表格式：`maturity_gate.judge(3, evidence, 'TensorBEM', 17318) → (3, [])`，避免守护误判为 invalid。
6. 工具链：python3 + psycopg + matplotlib；集群作业 cj1491 递交（5120 面板独立复算）当时仍排队，未采用。

## ③ 计算结果（ref vs got，逐项）
| panels | ref ½ρV [kg] | got [kg] | rel_err | <3% | ∂φ/∂n rel L₂ | 残差 ‖Ax-b‖₂ |
|---|---|---|---|---|---|---|
{tbl}

ref = {lv[-1]['exact_added_mass']:.6f} kg（机器求值）。DB 引文 4 个数与工件逐项一致，
最大 |Δ| = 4.10e-09（DB 文本仅 7 位有效数字所致）。
**首次达标 level = {first_pass} 面板（{1.738069e-02*100:.4f}%），最细 5120 面板 rel_err = {finest*100:.4f}%。**
对象自带测试 3 项全过（3 passed / 42.36 s），断言的是不变量（320p<3%、ladder 单调、
`first_passing_level==320`、解析交叉<1e-9、半径 2× 缩放=8.0 且 rel_err 不变）。

## ④ 结果图
（图 1）收敛阶梯：附加质量 rel_err 与 ∂φ/∂n 的 rel L₂ 随面板数按 ~O(N^-1) 下降，3% 容差线在 320 面板处穿越。
（图 2）ref–got 对角度：四点均落在 parity 线下方且随 N 收拢，量化了 flat-facet 离散偏差（体积比 0.873→0.998）。

## ⑤ 注意点（坑与局限）
* **F1（blocking for L4）交付 diff 为空**：对象 `git status` = ` M src/…/sphere_added_mass.py` + `?? tests/…`，
  `result.evidence.branch=null`，工件全在 `runs/`（`.gitignore:15`）——m17176 的病灶（runs/ 未入库）未被修掉，
  只是重演。分支检出无法复跑 `--ladder` 基准（main 上无该 CLI）。
* **F2（major）** `repro_cmd.sh` 声明 8 个产出，实际只有 2 个 per-level JSON + 2 个日志存在；脚本 mode 644。
* **F3（minor）** 5120 面板出现 `conditioning.py:69` 的 `invalid value encountered in matmul`（诊断路径未守卫，
  答案未受污染，残差 3.31e-14）。
* **F4（minor）** 对象行 `succeeded/accepted=True` 同时 `fail_stage='no_progress+forced_finish'`。
* **F5（minor）** 未发论文级报告（`report_id IS NULL`）。
* **机制盲点（可复用）**：`maturity_gate.judge()` 只核指针存在性（根=生态仓+审计工作树），对"是否真交付"盲：
  同一份证据 `judge(4)=（4,[]）`、`judge(5)=（5,[]）`。故本单主动下调至 L3，并建议加判据
  「impl_files 至少一个指针在生态仓 git 跟踪内」。
* **审计者自证**：本仓 `.gitignore` 同时忽略 `artifacts/`(第10行) 与 `runs/`(第15行)——审计侧复算工件首次提交
  也被静默吞掉，需 `git add -f` 才入库。F1 属仓库级陷阱而非个别疏忽；证据目录应改用 `docs/cases/<slug>/`
  或提交后校验 `git ls-files` 非空。
* 局限：审计者与对象同机同检出体系，只做入口级独立复算，未做算法级重写；sub4 独立复算（集群 cj1491）未取回。

## ⑥ 误差判定
| 关键量 | ref | got | 误差 | <3%? |
|---|---|---|---|---|
| m_A @5120 面板 | {lv[-1]['exact_added_mass']:.6f} kg | {lv[-1]['added_mass']:.6f} kg | **{finest*100:.4f}%** | **是** |
| m_A @1280 面板 | {lv[2]['exact_added_mass']:.6f} kg | {lv[2]['added_mass']:.6f} kg | {lv[2]['added_mass_relative_error']*100:.4f}% | 是 |
| m_A @320 面板 | {lv[1]['exact_added_mass']:.6f} kg | {lv[1]['added_mass']:.6f} kg | {lv[1]['added_mass_relative_error']*100:.4f}% | 是 |
| m_A @80 面板 | {lv[0]['exact_added_mass']:.6f} kg | {lv[0]['added_mass']:.6f} kg | {lv[0]['added_mass_relative_error']*100:.4f}% | 否（粗网格，预期） |

**显式回答：是，<3%**（自 320 面板起；最细 0.1130%）。但**判级不作 L4**：数值 PASS 不等于基准验证成立——
L4 要求"可复现"，而对象的实现/测试/工件均未进入交付分支。判定 **L3**，处置 request changes（返修）。

## 落表
`tensor_maturity_verdicts` id=529：repo=TensorBEM, scope='m17262 CASE-BEM-SPHERE-ADDED-MASS',
mission_id=17318, level=3, verdict_by='board-audit'；gate_state 待 maturity_gate 自动裁定。
产物：`docs/board_audit/m17318/`（README、insert_verdict.py、make_figures.py、repro/repro_audit.sh、
references/ 快照含 sha256、artifacts/ 审计复算、figs/）。
"""


def figure(name: str, caption: str) -> dict:
    p = FIGS / name
    return {"name": name, "caption": caption,
            "png_base64": base64.b64encode(p.read_bytes()).decode()}


def main() -> None:
    payload = {
        "title": "交差审计 m17318：CASE-BEM-SPHERE-ADDED-MASS（m17262/TensorBEM）判定 L3",
        "summary": "球体附加质量基准数值 PASS（最细 0.1130%，审计独立复算一致），但对象交付 diff 为空"
                   "（impl/测试未提交、工件在 gitignored runs/），可复现性不成立 → 判 L3，处置返修。",
        "scenario_key": "case-bem-sphere-added-mass-audit",
        "mission_id": 17318,
        "tags": ["TensorBEM", "board-audit", "CASE-BEM-SPHERE-ADDED-MASS", "added-mass",
                 "maturity-gate", "L3"],
        "markdown": md(),
        "figures": [
            figure("fig_convergence.png",
                   "图1 附加质量相对误差与 ∂φ/∂n 的 rel L2 误差随面板数收敛（3% 容差线、O(N^-1) 参考斜率）"),
            figure("fig_parity.png", "图2 ref（½ρV）vs got（BEM 附加质量）对角度，四点逐级收拢"),
        ],
    }
    tok = token()
    req = urllib.request.Request(f"{BASE}/tensor/reports", data=json.dumps(payload).encode(),
                                headers={"Content-Type": "application/json",
                                         "Authorization": f"Bearer {tok}"})
    with urllib.request.urlopen(req, timeout=90) as r:
        print(r.status, r.read()[:400].decode(errors="replace"))


if __name__ == "__main__":
    main()