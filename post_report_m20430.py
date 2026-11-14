#!/usr/bin/env python3
"""Post the paper-level report for the capytaine radiation-band axis (mission 20430)."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import requests

CASE = Path("/root/tp-agent/work/m20430/benchmarks/verified/casepool_bem_hulme_hemisphere")
ART = json.loads((CASE / "radiation_capytaine_band_xval.json").read_text())
BASE = os.environ["TENSOR_API_BASE"].rstrip("/")

# ---- token ---------------------------------------------------------------
tok = requests.post(
    f"{BASE}/auth/login",
    json={"username": os.environ["TENSOR_AGENT_USER"], "password": os.environ["TENSOR_AGENT_PASS"]},
    timeout=30,
).json()
token = tok.get("access_token") or tok.get("token")

d = ART
fine = max(d["constants"]["sub_levels"])
c = d["constants"]


def pct(v):
    return "n/a" if v is None else f"{100.0 * v:+.2f}%"


rows = []
for r in d["rows"]:
    a = r["attribution"]
    tbm = a["tensorbem_cell"]["values"] if a["tensorbem_cell"] else None
    rows.append(
        "| {ka} | {a11:.5f} | {c11:.5f} ({e11}) | {t11} | {a33:.5f} | {c33:.5f} ({e33}) | {t33} | {drift} | `{v}` |".format(
            ka=f"{r['ka']:.2f}",
            a11=r["reference"]["A11"], c11=r["capytaine"][str(fine)]["A11"],
            e11=pct(a["capytaine_vs_reference"]["A11"]),
            t11=("n/a" if tbm is None else f"{tbm['A11']:.5f} ({pct(a['tensorbem_vs_reference']['A11'])})"),
            a33=r["reference"]["A33"], c33=r["capytaine"][str(fine)]["A33"],
            e33=pct(a["capytaine_vs_reference"]["A33"]),
            t33=("n/a" if tbm is None else f"{tbm['A33']:.5f} ({pct(a['tensorbem_vs_reference']['A33'])})"),
            drift=f"{r['referee_self_drift_max']:.2%}",
            v=a["verdict"],
        )
    )

gates = "\n".join(
    f"| `{name}` | `{gate['verdict']}` | "
    f"{json.dumps({k: v for k, v in gate.items() if k != 'verdict'}, ensure_ascii=False)[:220]} |"
    for name, gate in d["gates"].items()
)

attributed = {k: v for k, v in d["gates"]["G3_attribution_defined"]["per_ka"].items() if v == "solver_side"}

md = f"""# CASE-BEM-HULME-HEMISPHERE：ka>2 断点的第三方归因（capytaine 独立轴）

## ① 案例介绍

浮半球（半径 a = {c['radius']} m，平面与平均自由面齐平，无限水深，ρ = {c['rho']:.0f} kg/m³，g = {c['g']} m/s²），
自由振荡淹没体，考核 6 个量：A11、B11、A33、B33 与激励力 |Fx|、|Fz|（本单聚焦四个辐射系数）。
基准 = Hulme (1982) 球谐波-free 多极级数（本仓 `tensorbem.analytic.hulme_hemisphere_radiation`，144 项）。
算例池接受带 ka ≤ 2；`band_extension/` 已如实记录 ka > 2 的 honest FAIL（最差 |A11| = 104.2% @ ka=3.5），
但该报告同时登记了两个未闭合的欠账：参考解与求解器共享 `free_surface3d.wave_function_3d`，
且 ka > 2 段**没有任何第三方轴**——因此那次 FAIL 究竟是「求解器坏」还是「基准坏」无法判定。

## ② 怎么做的

- **独立裁判**：capytaine {d['environment']['capytaine'].split()[-1]}（Nemoh 后继，Delhommeau 自由面 Green 函数 + Hess-Smith 常数面元），
  与 TensorBEM 无一行共享代码；同一组湿表面三角形（`hemisphere_hull_mesh`）。
- **网格阶梯**：4·4^sub 面元，sub = {c['sub_levels']} → {', '.join(str(4*4**s) for s in c['sub_levels'])} 面元；
  1024 面元这一级刻意保留为**与 TensorBEM 单元同网格**的对照级，4096 级作为裁判的最佳解。
  （第一轮用 256/1024，G2 判 FAIL：裁判自身漂移在 ka>3 达 30%、在 ka=2.58 达 879%——于是**抬高网格而不是放门限**。）
- **频带**：控制格 ka = 1.0、2.0；扩展带 ka = 2.2、2.58、3.0、3.5、4.0（与 `band_extension` 逐格对齐）。
- **预注册判据**（跑前写死，事后未放宽）：G1 控制格内裁判 ≤5%；G2 裁判自收敛 ≤5%；G3 每个 ka>2 格必须给出判定
  （`solver_side` / `reference_suspect` / `both_off` / `no_fail`），`referee_unresolved` 直接记 FAIL；
  G4 基准 96→144→192 项截断自收敛；G5 带内两码互校（本单因冻结单元不含 ka<2.2 而记 NOT_RUN，已在文档说明）。
- **工具链**：本机 Python {d['environment']['python']} / numpy {d['environment']['numpy']}，全部数值由
  `radiation_capytaine_band_xval.py` 程序化生成并冻结进 `radiation_capytaine_band_xval.json`；
  图中曲线全部读该 JSON 与库函数，无手抄数字。
- 说明：首个集群作业 cj2710（n315）因节点无 capytaine 失败（rc=1，日志存档），单次求解实测 1.8 s/ka@1024 面元，
  故本单在控制面按轻量规则完成全部 7×2 次求解（总 ~4 min，单核）。

## ③ 计算结果（4096 面元裁判 vs Hulme 级数，1024 面元 TensorBEM 对照）

| ka | A11 ref | capytaine 4096 (err) | TensorBEM 1024 (err) | A33 ref | capytaine 4096 (err) | TensorBEM 1024 (err) | 裁判自漂移 | 判定 |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

判据表：

| gate | verdict | 证据 |
|---|---|---|
{gates}

总体：**{d['overall']}**（G1、G4 PASS；G2、G3 FAIL，原因见 ④⑤）

## ④ 结果图

![coefficients](fig1) 辐射系数 A33/B33 对 ka：Hulme 级数（实线）、capytaine 4096 面元（方块）、
TensorBEM 1024 面元（三角）。灰底为算例池接受带 ka ≤ 2。

![error](fig2) 四个量的相对误差（对数轴）与 5% 门线：实线 capytaine、虚线 TensorBEM。

## ⑤ 注意点（局限与适用范围）

1. **ka = 2.58 是两条路线共同的不规则频率**：capytaine 在该格误差 108%（自漂移 37%），TensorBEM 的 CHIEF 也在同一格失效
   （曾记录「CHIEF on 18.5% 反比 off 6.1% 差」）。该格两码都不能作为裁判，故记 `referee_unresolved`——不是把它算成通过。
2. **裁判自身在 ka ≥ 3 尚未收敛**（自漂移 6.8%–8.6%，略超 5% 门线）：这些格上判据只能作为**上界参考**，不构成 <5% 的证书。
   这也解释了为什么 G2、G3 记 FAIL——本单的诚实结论是「归因在部分格上成功，但不是全带闭合」。
3. **归因成功的格**（{', '.join(f"ka={k}" for k in attributed) or '无'}）方向一致：capytaine 在 3.5–3.8% 内复现 Hulme 级数，
   而 TensorBEM 1024 面元偏 15.6%（ka=2.2）与 104.2%（ka=3.5）——**ka>2 的断点在求解器一侧**，先前 honest FAIL 的归因成立，未被推翻。
4. 两者都是低阶常数面元法，本轴区分的是「基准 vs 求解器」，不能替代高阶方法或贴体网格的绝对精度结论。
5. capytaine 3.x 无 `add_lid`，故裁判臂未加覆盖盖；不规则频率风险由 G2（自收敛）与逐格 `first_irregular_ka` 记录共同兜底。

## ⑥ 误差判定（是否 <3%？）

**否（整体），但归因方向明确。** 逐项：
- 控制带 ka = 1.0：裁判与基准带内最大 {d['gates']['G1_control_in_band']['per_ka']['1.00']:.2%}；ka = 2.0：{d['gates']['G1_control_in_band']['per_ka']['2.00']:.2%} → **<3% 成立**（裁判可用，归一化与时间/相位约定正确）。
- 扩展带裁判误差：ka=2.2 {d['rows'][2]['attribution']['capytaine_band_max']:.2%}、ka=3.0 {d['rows'][4]['attribution']['capytaine_band_max']:.2%}、
  ka=3.5 {d['rows'][5]['attribution']['capytaine_band_max']:.2%}、ka=4.0 {d['rows'][6]['attribution']['capytaine_band_max']:.2%}、
  ka=2.58 {d['rows'][3]['attribution']['capytaine_band_max']:.1%} → **不满足 <3%**，最好格（ka=3.5）也有 3.51%。
- 对比：同带 TensorBEM 差 10.6%–104.2%。因此「<5% 达标带止于 ka=2」的结论与归因均维持不变，
  新增的证据是：**卡住的是求解器一侧的离散误差，而不是基准级数**（除 ka=2.58 的不规则频率格两码同时失效）。

## ⑦ 复现

```
PYTHONPATH=src python benchmarks/verified/casepool_bem_hulme_hemisphere/radiation_capytaine_band_xval.py --write-doc
PYTHONPATH=src python benchmarks/verified/casepool_bem_hulme_hemisphere/make_radiation_band_figures.py
PYTHONPATH=src pytest tests/test_casepool_hulme_radiation_capytaine_band.py -q   # 17 passed
```

## ⑧ 踩过的坑

- 集群 n315 无 capytaine（`No module named 'capytaine'`），**外部 OSS 依赖必须先探针**再递交；本单改用控制面轻量求解（1.8 s/ka@1024）。
- 首轮网格阶梯取 256/1024 时 G2 大范围 FAIL，几乎把裁判变成噪声源；正确做法是抬高阶梯（→4096），而不是放宽门限。
- 归因判据必须把 `referee_unresolved` 单列并强制 FAIL，否则「裁判没收敛」会被误读成「求解器没问题」。
"""

figs = []
for name, caption in (
    ("fig_radiation_band_coefficients.png", "辐射系数 A33/B33 对 ka：Hulme 级数 vs capytaine(4096) vs TensorBEM(1024)"),
    ("fig_radiation_band_error.png", "四量相对误差 vs ka（对数轴）与 5% 门线"),
):
    p = CASE / name
    figs.append({"name": name, "caption": caption,
                 "png_base64": base64.b64encode(p.read_bytes()).decode()})

payload = {
    "title": "CASE-BEM-HULME-HEMISPHERE：ka>2 断点的第三方归因（capytaine 独立轴，honest FAIL 保留）",
    "summary": (
        "用 capytaine(4096 面元) 作为独立第三方轴覆盖 Hulme 浮半球辐射系数的 ka>2 扩展带，"
        "判定先前 honest FAIL 的归因：ka=2.2/3.5 上裁判 3.8%/3.5% 而复现基准、TensorBEM 1024 面元偏 15.6%/104.2% → 断点在求解器侧；"
        "ka=2.58 为两码共同的不规则频率格，无法裁判；裁判自身在 ka≥3 自漂移 6.8%–8.6%，故 G2/G3 如实记 FAIL。"
    ),
    "scenario_key": "bem_free_surface_verification",
    "mission_id": 20430,
    "tags": ["TensorBEM", "CASE-BEM-HULME-HEMISPHERE", "verification", "capytaine", "honest-fail", "radiation"],
    "markdown": md.replace("![coefficients](fig1)", figs[0]["name"]).replace("![error](fig2)", figs[1]["name"]),
    "figures": figs,
}

r = requests.post(f"{BASE}/tensor/reports", json=payload,
                  headers={"Authorization": f"Bearer {token}"}, timeout=120)
print(r.status_code, r.text[:600])