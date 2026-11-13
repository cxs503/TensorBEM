#!/usr/bin/env python3
"""Board-audit report for mission 17245 (CASE-BEM-KP505-GCI) -> /tensor/reports.

Figure: (a) the three-density KT ladder at J=0.5/0.7/0.9 with the EFD anchor,
(b) GCI_fine per quantity vs the predeclared 3 % line, (c) anchor errors vs the
15 % platform line. All numbers are read from the delivered artefacts in
docs/board_audit/m17266/snapshot (no hand-typed values).
"""
import base64
import csv
import io
import json
import os
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SNAP = "docs/board_audit/m17266/snapshot"
OUT = "docs/board_audit/m17266/fig_audit_kp505_gci.png"


def main():
    lv = {}
    for r in csv.DictReader(open(f"{SNAP}/levels.csv")):
        lv[(float(r["J"]), r["level"])] = {q: float(r[q]) for q in ("KT", "10KQ", "eta0")}
    gci = {(float(r["J"]), r["quantity"]): r for r in csv.DictReader(open(f"{SNAP}/gci.csv"))}
    case = json.load(open(f"{SNAP}/case.json"))
    rec = json.load(open("docs/board_audit/m17266/independent_recompute.json"))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for J, c in ((0.5, "C0"), (0.7, "C3"), (0.9, "C2")):
        ys = [lv[(J, d)]["KT"] for d in ("D1", "D2", "D3")]
        ax[0].plot([1, 2, 3], ys, "o-", color=c, label=f"J={J} (KT)")
        ax[0].annotate(f"{ys[2]:.4f}", (3, ys[2]), textcoords="offset points",
                       xytext=(4, -3), fontsize=8)
    ax[0].axhline(0.193, ls="--", color="k", lw=1, label="EFD anchor KT=0.193 (secondary)")
    ax[0].set_xticks([1, 2, 3])
    ax[0].set_xticklabels(["D1 8x2x24", "D2 16x4x48", "D3 32x8x96"], fontsize=8)
    ax[0].set_ylabel("KT")
    ax[0].set_title("three-density ladder, r = 2 (delivered artefacts)")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=.3)

    qs = ["KT", "10KQ", "eta0"]
    xs = range(len(qs))
    vals = [float(gci[(0.7, q)]["gci_fine_pct"]) for q in qs]
    bars = ax[1].bar(xs, vals, color=["C0", "C1", "C2"])
    ax[1].axhline(3.0, ls="--", color="r", label="predeclared tolerance 3 %")
    for b, v, q in zip(bars, vals, qs):
        ax[1].text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.2f}%", ha="center", fontsize=9)
    ax[1].set_xticks(list(xs))
    ax[1].set_xticklabels(qs)
    ax[1].set_ylabel("GCI_fine [%]")
    ax[1].set_title("J = 0.70: GCI fails the 3 % line (FAIL)")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=.3, axis="y")

    anchors = [a for a in rec["anchor"] if a["got_kind"] == "finest_D3"]
    ax[2].bar([f"{a['quantity']}\nD3" for a in anchors],
              [a["recomputed_error_pct"] for a in anchors],
              color=["C0" if abs(a["recomputed_error_pct"]) <= 15 else "C3" for a in anchors])
    ax[2].axhline(15, ls="--", color="r", label="platform 15 % line")
    ax[2].axhline(-15, ls="--", color="r")
    ax[2].axhline(0, color="k", lw=.8)
    for i, a in enumerate(anchors):
        ax[2].text(i, a["recomputed_error_pct"] + (0.6 if a["recomputed_error_pct"] > 0 else -1.6),
                   f"{a['recomputed_error_pct']:.2f}%", ha="center", fontsize=9)
    ax[2].set_ylabel("error vs EFD anchor [%]")
    ax[2].set_title("anchor error @J=0.70 (2 of 3 over 15 %)")
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=.3, axis="y")

    fig.suptitle("m17245 CASE-BEM-KP505-GCI board audit (L2, honest FAIL reproduced)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(OUT, dpi=130)
    png = base64.b64encode(open(OUT, "rb").read()).decode()

    md = f"""# CASE-BEM-KP505-GCI 交差核验（m17245）— L2，honest FAIL 复核成立

## 1. 案例介绍
KP505 五叶定距桨敞水离散化不确定度研究（TensorBEM）。案例问：这套升力面/涡格 BEM 的敞水系数的网格离散化不确定度是否 <3%。
参数：J ∈ {{0.50, 0.70, 0.90}}（设计点 0.70），Z=5，D=0.250 m，d_h/D=0.180，t/c=0.060，r=2.0，Fs=1.25，Re(0.7R)=4.84e5；三级阶梯 D1 8×2×24 / D2 16×4×48 / D3 32×8×96（径向×弦向×尾涡步），三条轴同时加倍。参考 = Roache GCI 程序（Roache 1994 JFE 116:405；Celik 2008 JFE 130:078001），锚值 = Iftikhar & Takinaci (MARINE 2023) Table 2 p.10 实验列（**二手转录**）。

## 2. 怎么做的（核验方法）
审计不是重做算例，而是①读交付工件与报告（`tensor_reports` id=16245）；②查唯一裁判 `tensor_benchmark_cases` 与平台 L4 机器裁定；③**重跑** `repro_cmd` 165 s 并做逐字节比对；④从 `levels.csv` 独立重算 p/GCI/渐近比（不 import tensorbem）；⑤复核参考侧 PDF sha256 与抽文行；⑥按 `maturity_gate` 判据方案做机器级复核。

## 3. 计算结果
| 量 @J=0.70 | 验收线 | got GCI_fine | 观测阶 p | <3% |
|---|---|---|---|---|
| KT | ≤3% | 10.074% | −3.967 | ✘ |
| 10KQ | ≤3% | 8.474% | −1.125 | ✘ |
| eta0 | ≤3% | 11.249% | −0.774 | ✘ |

锚值对照：KT −18.01%（外推 −11.40%），10KQ −19.69%（外推 −14.25%），eta0 +2.10%（外推 +11.29%）。
复跑一致性：`gci.csv` 逐字节相同，`levels.csv` 仅 wall_seconds 不同，`case.json` 四个关键块全等，两图 md5 相同；独立重算与交付最大偏差 <1e-6。
机器裁判：`tensor_benchmark_cases` 中该 anchor 0 行；`GET /tensor/bench/l4-verdict/17245` → `CLAIMED_NOT_PROVEN`。

## 4. 结果图
见下图：(a) 三密度 KT 阶梯与锚值；(b) J=0.70 的 GCI 三量与 3% 线；(c) 锚值误差与平台 15% 线。

## 5. 注意点
- eta0 的 +2.10% 是 KT/10KQ 同向偏低在比值里抵消的结果，**不能**当精度达标证据。
- 展向面元数不是有效收敛旋钮：n_radial 8→32 扫描 KT 极差 19.32% 且非单调（p=−5.845）。
- 锚为二手转录，只能作报告项；沙箱拿不到 KRISO 原表，第四密度级（1024 面元×192 尾涡步，>6.4 GB）本容器不可承载 → 渐近区只能"准入"不能"验收"。
- 交付悬在 `agent/m17245`，生态仓 grep 0 命中，`tensor_case_pool` id=164 仍 dispatched → L3 拒。

## 6. 误差判定（逐项回答是否 <3%）
- GCI_fine（案例自身的验收量）：**否**，8.47–11.25%（最差 eta0 11.25%），超线 2.8–3.7 倍 → honest FAIL 成立（fail_stage=G3-monotone）。
- 锚值 KT / 10KQ：**否**，−18.01% / −19.69%，越过平台 15% 不可接受线。
- 锚值 eta0：是（+2.10%），但如上所述被比值抵消掩盖，不作达标证据。
- 判级：**L2**（真代码+入口+真工件+可复现，机器复核 `judge(2)->(2,[])`）；L3/L4 拒（无调用方、基准未过、机器 L4 未证），L5 拒（二手锚）。
"""
    body = {"title": "交差审计 m17245 CASE-BEM-KP505-GCI：L2 + honest FAIL 复核（GCI 8.47–11.25% > 3%）",
            "summary": "复核成立：工件真、复跑逐位一致、独立重算偏差<1e-6；算例自身 3% 判据未过（GCI_fine 8.47–11.25%，观测阶 p<0），锚值 KT −18.0%/10KQ −19.7% 越 15% 线；平台 tensor_benchmark_cases 无该锚行、L4 机器裁定 CLAIMED_NOT_PROVEN、生态仓 0 调用方 → 判 L2（机器复核一致），三项返修建议。",
            "scenario_key": "propeller_open_water_verification",
            "mission_id": 17266,
            "tags": ["TensorBEM", "CASE-BEM-KP505-GCI", "board-audit", "m17245", "GCI", "Roache", "honest-FAIL", "L2"],
            "markdown": md,
            "figures": [{"name": "fig_audit_kp505_gci.png",
                         "caption": "m17245 交差审计三联图：(a) 三密度 KT 阶梯 vs EFD 锚；(b) J=0.70 GCI 三量与 3% 预声明线；(c) 锚值误差 vs 平台 15% 线",
                         "png_base64": png}]}
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
    lreq = urllib.request.Request(base + "/auth/login",
                                  data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                                                   "password": os.environ["TENSOR_AGENT_PASS"]}).encode(),
                                  headers={"Content-Type": "application/json"})
    tok = json.load(urllib.request.urlopen(lreq, timeout=30))["access_token"]
    req = urllib.request.Request(base + "/tensor/reports", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + tok})
    d = json.load(urllib.request.urlopen(req, timeout=120))
    print("report_id", d.get("id"), "mission", d.get("mission_id"), "figures", d.get("figures"))


if __name__ == "__main__":
    main()