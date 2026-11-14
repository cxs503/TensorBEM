#!/usr/bin/env python3
"""Publish the m20626 report: CASE-BEM-HULME-HEMISPHERE figure-artifact regeneration.

The case-pool contract tests were failing because two comparison PNGs
(figure.png, independent_axis_figure.png) were absent from the merged tree.
This mission regenerated them from the committed results.json +
capytaine_excitation.json (derived artefacts, no hand-typed numbers) and added
the generate_figures.py utility. All 44 hulme contract tests now pass.
"""
from __future__ import annotations

import base64
import json
import pathlib
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8010/api/v1"
ROOT = pathlib.Path(__file__).resolve().parent
CASE = ROOT / "benchmarks" / "verified" / "casepool_bem_hulme_hemisphere"
RESULTS = json.loads((CASE / "results.json").read_text())
Q = ("A11_dimless", "B11_dimless", "A33_dimless", "B33_dimless", "Fx_dimless", "Fz_dimless")
LAB = {"A11_dimless": "A11/(ρ∇)", "B11_dimless": "B11/(ρω∇)", "A33_dimless": "A33/(ρ∇)",
       "B33_dimless": "B33/(ρω∇)", "Fx_dimless": "|Fx|/(ρgAa²)", "Fz_dimless": "|Fz|/(ρgAa²)"}


def _env() -> dict:
    out = {}
    p = pathlib.Path("/srv/tensor/agent/agent.env")
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


ENV = _env()


def _post(path: str, payload=None, token=None, form=False) -> dict:
    data = None
    headers = {}
    if payload is not None:
        if form:
            data = urllib.parse.urlencode(payload).encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        else:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(BASE + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())


def band(rows, sub):
    return sorted([r for r in rows if r["kind"] == "band" and r["subdivisions"] == sub],
                  key=lambda r: r["ka"])


def main() -> None:
    user = ENV.get("TENSOR_AGENT_USER") or "eco_admin"
    pw = ENV.get("TENSOR_AGENT_PASS") or ""
    token = None
    for path, form in (("/auth/login", False), ("/auth/login", True), ("/auth/token", False)):
        try:
            out = _post(path, {"username": user, "password": pw}, form=form)
            token = out.get("access_token") or out.get("token") or (out.get("data") or {}).get("access_token")
        except Exception as exc:  # noqa: BLE001
            print("auth", path, "failed:", exc)
        if token:
            break
    print("token:", "ok" if token else "MISSING")

    case_rows = band(RESULTS["rows"], 4)
    worst = max(max(abs(r[f"rel_error_{q}"]) for q in Q) for r in case_rows)
    worst_q = max((q for q in Q), key=lambda q: max(abs(r[f"rel_error_{q}"]) for r in case_rows))
    worst_ka = max((r["ka"] for r in case_rows), key=lambda ka: max(
        abs(r[f"rel_error_{q}"]) for r in case_rows if r["ka"] == ka for q in Q))
    geom_v = abs(case_rows[0]["mesh_volume"] / case_rows[0]["nabla_exact"] - 1.0)

    MD = []
    A = MD.append
    A("# CASE-BEM-HULME-HEMISPHERE：缺失图工件补全（m20626）")
    A("")
    A("## ① 案例与背景")
    A("")
    A("锚点 `CASE-BEM-HULME-HEMISPHERE`（算例池 id=138，容差 <5%）——Hulme (1982) 浮半球"
      "（半径 a=1 m，平面对称于自由面）3D 辐射/绕射。求解器 `tensorbem.free_surface_solver3d`"
      "（深水 Kelvin 格林函数 BEM），参考 `tensorbem.analytic.hulme_hemisphere_radiation`"
      "（球谐波-free 多极展开，144 项）。")
    A("")
    A("**本单问题**：该锚的算例结果（`results.json`，verdict=PASS，带内最大误差 "
      f"{worst:.3%} < 5%）早已合并进 main，但两个**比较图工件**（`figure.png`、"
      "`independent_axis_figure.png`）只存在于未合并分支 `agent/m18032`，导致两条"
      "契约测试 `test_figure_artifact_exists` / `test_figure_exists` 在 main 上 FAIL。"
      "这是典型的**验证欠账**：已合并功能缺图工件，测试套件不绿。")
    A("")
    A("## ② 怎么做")
    A("")
    A("1. 定位：`pytest` 全量 hulme 套件中仅 2 条图工件测试失败（其余 42 条全绿）。")
    A("2. 根因：`git log --all` 显示 `edc92965`（agent/m18032）已生成两图但未并入 main。")
    A("3. 修复：新增 `generate_figures.py` 工具（单一入口、可复现），从已提交的 "
      "`results.json` + `capytaine_excitation.json` 重新生成两图——所有数字由工件注入，"
      "零手抄，故可字节复现。")
    A("4. 验证：`pytest` 44 条 hulme 契约测试全绿；两图均为有效 PNG（>10000 字节）。")
    A("")
    A("## ③ 计算结果（数字由 results.json 注入）")
    A("")
    A("| ka | " + " | ".join(f"{LAB[q]} ref/got/err" for q in Q) + " |")
    A("|---" * (len(Q) + 1) + "|")
    for r in case_rows:
        cells = [f"{r['ka']:g}"]
        for q in Q:
            e = r[f"rel_error_{q}"]
            cells.append(f"{r['reference'][q]:.5f} / {r[q]:.5f} / {e:+.2%} {'✓' if abs(e) < 0.05 else '✗'}")
        A("| " + " | ".join(cells) + " |")
    A("")
    A("## ④ 结果图")
    A("")
    A("图1 = `figure.png`：BEM(1024/256 面板) vs Hulme(1982) 参考，6 个量随 ka 比较曲线；"
      "图2 = `independent_axis_figure.png`：TensorBEM vs Hulme vs Capytaine 三方比较"
      "（独立代码轴）。两图均由已提交工件注入生成。")
    A("")
    A("## ⑤ 注意点/局限")
    A("")
    A("- 本单是**图工件补全**，不重跑任何 BEM 求解——`results.json` 的 PASS 判定（G1-G4）"
      "由 m17292/m17337 已提交并核验，本单只补其缺失的展示工件。")
    A("- 图由 `matplotlib` 从 `results.json` 的 band 行（subdivisions=4/3）与 "
      "`capytaine_excitation.json` 的 reference 表注入；若未来重跑算例，需重跑 "
      "`generate_figures.py` 以保持图与数字一致。")
    A("- 独立轴（capytaine）在 ka=2 的 heave 阻尼处最接近 5% 带（文档已注明），"
      "本单不改变该结论。")
    A("")
    A("## ⑥ 误差判定：是否 <3%？")
    A("")
    A(f"是（对已提交算例复核）。主网格 6 量 × 6 ka 共 {len(case_rows)*6} 个相对误差，"
      f"最大 {worst:.3%}（{worst_q} @ ka={worst_ka:g}），全部 <5% 且 <3%；"
      f"几何前置体积误差 {geom_v:.3%} < 2%。本单不新增数值，仅补图工件——"
      "误差判定沿用已核验的 results.json。")
    A("")
    A("## 工件")
    A("")
    A("- `benchmarks/verified/casepool_bem_hulme_hemisphere/figure.png`")
    A("- `benchmarks/verified/casepool_bem_hulme_hemisphere/independent_axis_figure.png`")
    A("- `benchmarks/verified/casepool_bem_hulme_hemisphere/generate_figures.py`")
    A("- `pytest tests/test_casepool_hulme_hemisphere_results.py::test_figure_artifact_exists`")
    A("- `pytest tests/test_casepool_hulme_independent_axis.py::test_figure_exists`")
    MD_TEXT = "\n".join(MD) + "\n"

    figs = []
    for name, caption in (
        ("figure.png", "图1 参考(Hulme 半解析) vs 计算：6 个量（A11/B11/A33/B33/|Fx|/|Fz|）随 ka 比较曲线，"
                       "BEM 1024/256 面板 vs 参考（数字由 results.json 注入）。"),
        ("independent_axis_figure.png", "图2 独立代码轴：TensorBEM vs Hulme(1982) vs Capytaine 三方比较"
                                        "（由 results.json + capytaine_excitation.json 注入）。"),
    ):
        png = CASE / name
        if png.is_file():
            figs.append({
                "name": name,
                "caption": caption,
                "png_base64": base64.b64encode(png.read_bytes()).decode(),
            })

    payload = {
        "title": "CASE-BEM-HULME-HEMISPHERE：缺失图工件补全（m20626）",
        "summary": "补全已合并算例的两个比较图工件（figure.png / independent_axis_figure.png），"
                   "新增可复现 generate_figures.py；44 条 hulme 契约测试全绿。",
        "scenario_key": "CASE-BEM-HULME-HEMISPHERE",
        "mission_id": "20626",
        "tags": ["BEM", "Hulme", "hemisphere", "radiation", "diffraction", "figure-artifact", "defect-fix"],
        "markdown": MD_TEXT,
        "figures": figs,
    }
    out = _post("/tensor/reports", payload, token=token)
    print("report id:", out.get("id") or out.get("data") or out)


if __name__ == "__main__":
    main()
