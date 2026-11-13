"""m17340 deliverables: audit figure, platform report, maturity verdict row."""
from __future__ import annotations

import base64
import json
import os
import pathlib
import subprocess
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
CHECKS = json.loads((ROOT / "artifacts/audit_m17340/audit_checks.json").read_text())
STORED = json.loads((ROOT / "artifacts/audit_m17340/case_bem_cummins_wigley/results.json").read_text())
AUDIT_MD = (ROOT / "docs/board_audit/m17340/AUDIT_m17321.md").read_text()
DOFS = ("surge", "sway", "heave", "roll", "pitch", "yaw")

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
med = STORED["levels"]["medium"]["gates"]
rows = {r["dof_label"]: r for r in med["cross_solver"]["rows"]}
x = np.arange(len(DOFS))
w = 0.38
ax = axes[0]
ax.bar(x - w / 2, [100 * abs(rows[d]["added_mass_rel_err"]) for d in DOFS], w,
       label="added mass |ΔA|", color="#2b6cb0")
ax.bar(x + w / 2, [100 * abs(rows[d]["damping_rel_err"]) for d in DOFS], w,
       label="damping |ΔB|", color="#dd6b20")
ax.axhline(10.0, color="k", ls="--", lw=1.2, label="G1 tol 10 %")
ax.set_xticks(x); ax.set_xticklabels(DOFS); ax.set_ylabel("|ref − got| / ref  [%]")
ax.set_title("G1 cross-solver vs capytaine 3.1 (medium mesh, ω=1 rad/s)")
ax.legend(fontsize=8)

ax = axes[1]
g3 = [100 * med["detail"]["G3_kramers_kronig"]["per_dof"][d] for d in DOFS]
g5 = [100 * med["detail"]["G5_round_trip"]["per_dof"][d] for d in DOFS]
kk = CHECKS["kk_independent_route"]["medium"]
kkv = [100 * kk[d]["auditor_G3_rel_err_cummins_sign"] if d in kk else np.nan for d in DOFS]
ax.bar(x - w / 2, g3, w, label="G3 Kramers-Kronig (subject)", color="#c53030")
ax.plot(x - w / 2, kkv, "k*", ms=11, label="G3 auditor route (time-domain kernel)")
ax.bar(x + w / 2, g5, w, label="G5 B→K round trip", color="#805ad5")
ax.axhline(10.0, color="k", ls="--", lw=1.2, label="G3 tol 10 %")
ax.axhline(5.0, color="gray", ls=":", lw=1.2, label="G5 tol 5 %")
ax.set_xticks(x); ax.set_xticklabels(DOFS); ax.set_ylabel("relative residual [%]")
ax.set_title("Causality gates that fail the anchor (medium): G3 39.8–81.6 %, G5 35.3–53.4 %")
ax.legend(fontsize=7.5); ax.set_yscale("log")
fig.tight_layout()
FIG = ROOT / "docs/board_audit/m17340/fig_audit_m17340.png"
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG, dpi=150)

repr_coarse = CHECKS["reproduction"]["coarse"]
ref = CHECKS["reference_rebuild"]["levels"]

markdown = AUDIT_MD + f"""

## 8 误差判定（显式回答「是否 <3 %」）

| 量 | 参照 | got | 误差 | <3 %? |
|---|---|---|---|---|
| G1 悬架/垂荡/横摇…（6 DOF 最大值，medium, ω=1） | capytaine 3.1.dev0 冻结件（审计方重建重现 5.19e-13） | TensorBEM 同一网格解 | **3.26 %**（damping, roll） | **否**（但 < 锚点 10 % 容差，PASS） |
| G1 coarse 最大值 | 同上 | 同上 | 6.29 % | 否（<10 % PASS） |
| 几何 ∇ | 解析 0.09163125 m³ | 网格散度定理 | coarse −2.085 %, medium −0.526 %（网格离散，非解误差） | medium 是；Cb 0.543 精确 |
| 参照文件自复现 | 冻结 capytaine_ref.json | 审计方重跑 capytaine | 2.95e-12 (10×5) / 5.19e-13 (20×10) | **是** |
| G3 KK 因果（锚点核心） | Hilbert(B) | A(ω) | **39.8 %–81.6 %** | **否（超 13–27 倍）** |
| G5 往返 | B→K→(B,A) 带内 | 同 | 35.3 %–53.4 %（tol 5 %） | **否** |
| G4 自由衰减 ζ | 半功率 ζ=0.1675 | 时域衰减 ζ=0.0980 | **41.5 %**（medium 无法定义） | **否** |

> honest 结论：**锚点核心量（KK 因果）没有 <3 %，也没有 <10 %**。通过的是唯一外部锚 G1（3.26 %）。
> 本审计不粉饰：m17321 是诚实的部分交付，不是达标交付。

## 9 审计独立性与局限

- 独立性：审计方自写全部重算代码（`audit_independent.py`，10.4 s），KK 走时间域核路线而非主体的主值积分；
  capytaine 参照由审计方从头重建，不读主体中间量。
- 局限：仍用主体提交的同一批单频解（缓存 BEM 结果），未重解 BEM；G4 的 ζ 估计器（峰值/半功率）未替换，
  故 G4 的 41.5 % 只证明"两条估计器不一致"，未定位到谁的错；`wigley3_hull` 段漂移的根因（git 侧何时改的 mesher）
  未追到底。`docs/board_audit/m17340` 之外的仓状态未审。
"""

fig_b64 = base64.b64encode(FIG.read_bytes()).decode()
api = os.environ["TENSOR_API_BASE"]
login = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                    "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
req = urllib.request.Request(api + "/auth/login", data=login,
                             headers={"Content-Type": "application/json"})
tok = json.loads(urllib.request.urlopen(req, timeout=30).read())
tok = tok.get("access_token") or tok.get("token")

payload = {
    "title": ("交差审计 m17321 / CASE-BEM-CUMMINS-WIGLEY：三件套真实、外部锚可复现"
              "（capytaine 重建 5e-13），但锚点判据 KK 因果 39.8–81.6 % 未达 <10 % → 判 L3"),
    "summary": ("独立重算（自写通道，非读主体中间量）：G1 vs capytaine 3.1 冻结件 3.26 %/6.29 % PASS 且参照由审计方"
                "重建到 5.19e-13 证明参照非自产；G3 因果残差用时间域核路线复核 14.1–81.6 %，与主体主值积分差 ≤0.05 pp，"
                "反号对照 1.55–1.87 → FAIL 为真；G4 41.5 %/medium 不可定义；G5 35.3–59.1 %。锚点无 benchmark case 登记行。"
                "另发现：G2 为同义反复、报告把 G3 归因低频带截断被带截断实验否定（53.23 %→50.96 %）、"
                "参照文件 wigley3_hull 段不可复现（漂移 110 %）、medium 工件 18 点 vs results.json 16 点不同步。"
                "判 L3（系统集成），L4 拒授，锚点应保持 open 返修。"),
    "scenario_key": "CASE-BEM-CUMMINS-WIGLEY.audit",
    "mission_id": 17340,
    "tags": ["TensorBEM", "CASE-BEM-CUMMINS-WIGLEY", "board_audit", "Kramers-Kronig",
             "capytaine", "maturity-verdict", "L3", "honest-partial"],
    "markdown": markdown,
    "figures": [{"name": "fig_audit_m17340.png",
                 "caption": ("左：G1 与 capytaine 冻结件的逐自由度 |ΔA|/|ΔB|（medium, ω=1，虚线=10 % 容差）；"
                             "右：不达标的因果门 G3/G5 逐自由度残差（对数轴），黑星=审计方独立时间域核路线的 G3，"
                             "与主体主值积分路线重合 ⇒ FAIL 不是门实现问题"),
                 "png_base64": fig_b64}],
}
req = urllib.request.Request(api + "/tensor/reports", data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": "Bearer " + tok})
try:
    resp = json.loads(urllib.request.urlopen(req, timeout=180).read())
    print("REPORT:", json.dumps(resp)[:300])
except Exception as exc:  # noqa: BLE001
    print("REPORT FAILED:", exc)
    print(exc.read()[:800] if hasattr(exc, "read") else "")

# ---------------------------------------------------------------- verdict row
evidence = {
    "impl_files": ("docs/board_audit/m17340/subject_snapshot/cummins_wigley_kk_causality.py:181 geometry_gate, "
                   ":242 solve_point, :368 cross_solver_table, :421 analysis_gates, :553 heave_decay_gate, :636 gates_from_sweep"),
    "entry": "docs/board_audit/m17340/subject_snapshot/run_cummins_wigley_kk_causality.py:176 main (--mode point|sweep|gates|sanity)",
    "run_artifacts": ("artifacts/audit_m17340/case_bem_cummins_wigley/results.json; points/ 40 单频解 "
                      "(22 coarse + 18 medium, relative_residual 7.0e-5 @medium ω=1); figures/fig_sweep.png; "
                      "审计原始输出 artifacts/audit_m17340/audit_checks.json"),
    "tests": ("subject_snapshot/test_cummins_wigley_kk_causality.py 审计方实跑 9 passed in 3.67 s "
              "(含 KK 负控、旋转中心双钉、频带截断对照)"),
    "integration_callers": ("tensorbem.free_surface_solver3d.solve_radiation_3d (经 case.solve_point); "
                            "tensorbem.time_domain.radiation_memory_kernel/cummins_free_decay/half_power_damping/fourier_round_trip; "
                            "tensorbem.causality.kkr_added_mass; 平台 tensor_reports#16328"),
    "benchmark": ("G1 vs capytaine 3.1.dev0 冻结件(PASS 3.26% medium / 6.29% coarse, tol 10%; 参照由审计方重建 "
                  "worst 5.19e-13/2.95e-12 与 mesh signature 相同); G3 Kramers-Kronig 39.8%-81.6% FAIL(tol 10%, "
                  "审计方时间域核路线复核差<=0.05pp, 反号对照 1.55-1.87); G5 round trip 35.3%-59.1% FAIL(tol 5%); "
                  "G4 41.5% FAIL + medium 不可定义(IndexError); "
                  "tensor_benchmark_cases 中 anchor_slug='CASE-BEM-CUMMINS-WIGLEY' 0 行 => 判据取自 feature_ledger acceptance"),
    "repro_cmd": ("PYTHONPATH=src python3 scripts/run_cummins_wigley_kk_causality.py --mode gates --levels coarse,medium ; "
                  "PYTHONPATH=src python3 docs/board_audit/m17340/audit_independent.py"),
    "external_anchor": ("Cummins 1962 方程 + Kramers-Kronig 因果 (Kotik & Mangulis 1962; Newman 1977 ch.6); "
                        "独立代码参照 capytaine 3.1.dev0 -> benchmarks/verified/wigley3_recalled_ref/capytaine_ref.json "
                        "(生成脚本 scripts/wigley3_recalled_ref_capytaine.py, 审计方重跑复现)"),
    "audit_report": "docs/board_audit/m17340/AUDIT_m17321.md",
    "audit_findings": {
        "F1_anchor_judge_failed": "G3 39.8-81.6% / G5 35.3-59.1% / G4 41.5% vs tol 10/5/10 % -> acceptance not met",
        "F2_no_benchmark_case_row": "tensor_benchmark_cases 0 rows for this anchor (table has 51 rows)",
        "F3_G2_tautological": "K(0) equals the same trapezoid of B by construction; residual 1e-16 carries no physics",
        "F4_misattributed_root_cause": "band-cut control: yaw G3 53.23% -> 50.96% when lowest two nodes removed, so low-band truncation is not the dominant cause",
        "F5_stale_ref_section": "capytaine_ref.json 'wigley3_hull' section not reproducible (mesh signature differs, coefficients drift up to 110%); not used by this case",
        "F6_artifacts_not_synced": "medium: 18 cached points on disk vs 16 in results.json; recompute drifts G3 81.62->85.99 %, G5 53.37->59.12 % (verdict unchanged)",
        "F7_convergence_claim_weak": "no order evidence for G3/G5; coarse 53% -> medium 82% is divergence with refinement",
    },
    "level_rationale": ("L3: 真代码 + 接入主求解器(solve_radiation_3d)+时间域/因果生产线+可复现工件(粗档 5 门逐位重现)。"
                        "L4 拒授: 锚点核心判据(KK 因果<10%)FAIL 且无 benchmark case 登记行。"),
}
sql = """
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorBEM', 'm17321 CASE-BEM-CUMMINS-WIGLEY', 17340, 3, %s, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET level=EXCLUDED.level, evidence=EXCLUDED.evidence, mission_id=EXCLUDED.mission_id,
      gate_state='pending'
RETURNING id, gate_state, gate_level;
"""
ev = json.dumps(evidence, ensure_ascii=False)
p = subprocess.run(["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform", "-tA",
                    "-c", sql.replace("%s", "'" + ev.replace("'", "''") + "'")],
                   capture_output=True, text=True)
print("VERDICT:", p.stdout.strip(), p.stderr.strip()[:400])