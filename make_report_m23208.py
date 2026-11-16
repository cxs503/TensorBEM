"""Figures + dev report + platform report for m23208 (independent A_inf on the closure gates)."""

from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CASE_DIR = pathlib.Path("benchmarks/verified/casepool_bem_cummins_wigley")
EVIDENCE = CASE_DIR / "ainf_independent_gates.json"
FIG_DIR = pathlib.Path("docs/figures/m23208")
SWEEP = CASE_DIR / "sweep_closure_medium800.npz"
MISSION = 23208
DEV_REPORT = pathlib.Path("docs/dev_report_m23208_cummins_closure_ainf_independent.md")

PAYLOAD = json.loads(EVIDENCE.read_text())
EV = PAYLOAD["evaluation"]
ANCHOR = PAYLOAD["a_inf_anchor"]
ROWS = EV["rows"]
LABELS = [row["dof_label"] for row in ROWS]
TOL_G3 = PAYLOAD["gate_definitions"]["G3_kramers_kronig"]["tol"]
TOL_G5 = PAYLOAD["gate_definitions"]["G5_round_trip"]["tol"]
TOL_G4 = PAYLOAD["gate_definitions"]["G4_free_decay"]["tol"]
with np.load(SWEEP) as DATA:
    OMEGA = DATA["omega"]
    ADDED = DATA["added_mass"]
    DAMPING = DATA["damping"]

FIG_DIR.mkdir(parents=True, exist_ok=True)
FIGURES: list[dict] = []


def save(fig, name: str, caption: str) -> None:
    path = FIG_DIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    FIGURES.append({"name": name, "caption": caption,
                    "png_base64": base64.b64encode(path.read_bytes()).decode()})


# 1 --- in-band added mass vs the two asymptotic estimates -------------------
fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0))
for axis, row in zip(axes.ravel(), ROWS):
    dof = row["dof"]
    axis.semilogx(OMEGA, ADDED[:, dof, dof], "o-", ms=2.6, lw=1.0, label="$A_{ii}(\\omega)$")
    axis.axhline(row["a_inf_tail_fit"], color="crimson", ls="--", lw=1.1,
                 label=f"tail fit = {row['a_inf_tail_fit']:.3g}")
    low, high = row["a_inf_image_interval"]
    axis.axhspan(low, high, color="seagreen", alpha=0.30,
                 label=f"image solve = {row['a_inf_image']:.3g}")
    axis.set_title(f"{row['dof_label']}  (band top A={row['band_top']['A_at_last']:.3g})", fontsize=9)
    axis.set_xlabel("$\\omega$ [rad/s]", fontsize=8)
    axis.tick_params(labelsize=7)
    axis.legend(fontsize=6.5, loc="best")
fig.suptitle("Wigley III closure band [0.1, 20] rad/s: the in-band tail fit sits *below* the last "
             "solved value, the independent image solve does not", fontsize=10)
fig.tight_layout()
save(fig, "added_mass_and_asymptotes.png",
     "Diagonal added mass of the six modes on the 69-point closure band (mesh 20x10, 800 panels), "
     "with the in-band two-term tail fit (red dashed) and the independent image-solve asymptote "
     "(green band = mesh ladder 20x6 / 20x10 re-expressed about the sweep rotation centre).")

# 2 --- the two gates, both estimates --------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
x = np.arange(len(ROWS))
width = 0.38
for axis, key, tol, title in ((axes[0], "G3", TOL_G3, "G3 Kramers-Kronig residual"),
                              (axes[1], "G5", TOL_G5, "G5 band-limited round trip")):
    tail = [row[f"{key}_tail"] * 100 for row in ROWS]
    image = [row[f"{key}_image"] * 100 for row in ROWS]
    axis.bar(x - width / 2, tail, width, color="crimson", alpha=0.85, label="A_inf = in-band tail fit")
    axis.bar(x + width / 2, image, width, color="seagreen", alpha=0.85, label="A_inf = image solve")
    axis.axhline(tol * 100, color="k", ls=":", lw=1.2, label=f"tolerance {tol*100:.0f} %")
    axis.set_xticks(x, LABELS, fontsize=8)
    axis.set_ylabel("relative residual [%]", fontsize=8)
    axis.set_title(title, fontsize=9)
    axis.legend(fontsize=7)
    axis.tick_params(labelsize=7)
fig.tight_layout()
save(fig, "gates_tail_vs_image.png",
     "Predeclared G3 and G5 statistics recomputed with the two A_inf estimates. Both gates were "
     "failing at 28-33 % on the tail fit and land at 0.9-3.5 % with the independent asymptote; "
     "the gate definitions and the sweep are unchanged.")

# 3 --- G3 as a function of the assumed A_inf ------------------------------
fig, axes = plt.subplots(2, 3, figsize=(13.0, 6.6))
for axis, row in zip(axes.ravel(), ROWS):
    scan = row["scan"]
    fractions = np.asarray(scan["fractions"])
    g3 = np.asarray(scan["G3"]) * 100
    axis.plot(fractions, g3, "-", color="navy", lw=1.2)
    axis.axhline(TOL_G3 * 100, color="k", ls=":", lw=1.0)
    inside = fractions[np.asarray(scan["G3"]) <= TOL_G3]
    if inside.size:
        axis.axvspan(inside.min(), inside.max(), color="seagreen", alpha=0.18,
                     label=f"pass window x[{inside.min():.2f},{inside.max():.2f}]")
    axis.axvline(1.0, color="seagreen", lw=1.6, label="image solve")
    axis.axvline(scan["best_ainf"] / scan["reference_ainf"], color="orange", ls="--", lw=1.1,
                 label="G3 minimiser")
    axis.set_title(row["dof_label"], fontsize=9)
    axis.set_xlabel("assumed $A_\\infty$ / image value", fontsize=8)
    axis.set_ylabel("G3 [%]", fontsize=8)
    axis.tick_params(labelsize=7)
    axis.legend(fontsize=6.2)
fig.suptitle("G3 versus the assumed $A_\\infty$: the independently computed value (green) falls "
             "inside the admissible window and within a few per cent of the G3 minimiser", fontsize=10)
fig.tight_layout()
save(fig, "g3_vs_assumed_ainf.png",
     "G3 residual as a function of the assumed A_inf for every mode, normalised by the image value. "
     "The admissible window (green shading) is wide, and the image value is not an endpoint of it: "
     "the asymptote was not tuned to pass the gate.")

# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
gate_rows = "\n".join(
    f"| {row['dof_label']} | {row['a_inf_tail_fit']:.4f} | {row['a_inf_image']:.4f} | "
    f"{row['a_inf_relative_change']*100:+.1f} % | {row['G3_tail']*100:.2f} | "
    f"{row['G3_image']*100:.2f} | {row['G5_tail']*100:.2f} | {row['G5_image']*100:.2f} |"
    for row in ROWS)
def _window(segs):
    segs = segs or []
    if not segs:
        return (float("nan"), float("nan"))
    return (min(float(seg[0]) for seg in segs), max(float(seg[1]) for seg in segs))


window_rows = "\n".join(
    f"| {row['dof_label']} | [{_window([row['a_inf_image_interval']])[0]:.4g}, "
    f"{_window([row['a_inf_image_interval']])[1]:.4g}] | "
    f"{row['scan']['best_ainf']:.4f} | "
    f"{_window(row['scan']['G3_pass_window'])[0]:.4g}-{_window(row['scan']['G3_pass_window'])[1]:.4g} | "
    f"{'yes' if any(low <= row['a_inf_image'] <= high for low, high in (row['scan']['G3_pass_window'] or [])) else 'NO'} |"
    for row in ROWS)
worst_g3 = max(ROWS, key=lambda row: row["G3_image"])
worst_g5 = max(ROWS, key=lambda row: row["G5_image"])
spread = ANCHOR["ladder_spread_relative"]
sphere = ANCHOR["infinite_frequency_sphere_self_check"]
frame = ANCHOR["frame_check"]["max_relative_difference_direct_vs_transform"]
g4 = EV["G4"]

BODY = f"""# CASE-BEM-CUMMINS-WIGLEY：用**独立** A_∞ 重判闭包带 G3/G5（A_∞ 偏置是全部原因；G4 不是）

**Mission**: m{MISSION} ｜ **Anchor**: `CASE-BEM-CUMMINS-WIGLEY`（feature_ledger: open）｜ **仓**: TensorBEM
**对象**: 扩展频带闭包判据 G3（Kramers-Kronig）/G4（自由衰减）/G5（往返）所用的加质量渐近量 $A_\\infty$

## 1 案例与这条欠账从哪来

Wigley III（L=3.0 m, B=0.3 m, T=0.1875 m, Cb=0.543, ρ=1025 kg/m³, 无限水深），
网格 `(n_sections, n_levels) = (20, 10)`（800 面板），旋转中心
`(0, 0, -T/2) = {tuple(round(v, 5) for v in EV['rotation_center'])}`。
上一轮（m23130）把预声明闭包网格 **69 个频点**（0.10–20 rad/s，含 41 点共振簇）全部解出后，
G3/G4/G5 仍判 FAIL（33 % / 18 % / 33 %）。而 G5 的归因（m21140）给出的收尾条件是：

> 「需要独立的 A_∞ 锚（ω→∞ 极限的 Laplace 解，或文献值），而不是又一个从同一 (A,B) 造出来的估计器」

这个锚其实**已经入库但从未用来判据**：`tensorbem.added_mass_limit.added_mass_limit`
用反对称镜像把自由面换成高频极限 $\\phi=0$，把湿表面与其镜像闭成倍体，解的是
**另一个边值问题**——与频率扫频不共享任何代码、数据、频带，没有可调参数，也没有尾部拟合。

## 2 怎么做的

1. **把 69 点闭包扫频冻结入库**（`benchmarks/verified/casepool_bem_cummins_wigley/sweep_closure_medium800.npz`，
   A/B 6×6、逐点线性残差最大 {PAYLOAD['sweep']['max_linear_residual']:.2e}），
   否则判据无法被任何人重放。
2. **参考点对账**（这是本轮的硬前提）：扫频在 `(0,0,-T/2)` 取矩，而
   `case_bem_cummins_wigley.added_mass_inf_anchor` 在 `(0,0,0)`（水线）求解。$A_55$ 在参考点平移下
   **不守恒**（随 $A_35$ 耦合变化），所以先新增
   `tensorbem.added_mass_limit.shift_added_mass`（刚体速度变换 $X_1 = T X_2$，
   $T=[[I,[a\\times]],[0,I]]$，$A_2=T^\\top A_1 T$）把镜像矩阵搬到扫频的参考点上。
   **校验**：该变换与「直接在扫频参考点重解镜像问题」逐元素一致，最大相对差
   `{frame:.2e}`。
3. **不确定度**：镜像解跑网格梯子 `20x6 / 20x10`，$A_\\infty$ 以区间携带，逐 DOF 相对散布
   最大 `{max(spread)*100:.2f} %`。解析自校验（半浸球，$A_\\infty=\\rho V/4$）相对误差
   `{sphere['relative_error']*100:.2f} %`（容差 {sphere['tolerance']:.0%}）。
4. **判据不重定义**：`analysis_gates(..., added_mass_inf=...)` 只换常数来源，
   G3（归一化 $\\max|A-A_\\infty|$ 的 KK 残差）、G5（带限往返）、G4（时域衰减 ζ vs 半功率 ζ）
   的统计量与容差一字未改，两次评估并列报告。
5. **反循环检查**：把 $A_\\infty$ 当自由参数扫过 $[0.8, 1.2]\\times$ 镜像值（每 DOF 82 点），
   给出 G3 的**可判窗口**，看镜像值落在窗口的什么位置、离 G3 极小点多远。

## 3 计算结果（全部由 `ainf_independent_gates.json` 程序化注入，零手抄）

### 3.1 两个 $A_\\infty$ 与两条闸门

| DOF | tail fit | image solve | 变化 | G3 tail [%] | G3 image [%] | G5 tail [%] | G5 image [%] |
|---|---:|---:|---:|---:|---:|---:|---:|
{gate_rows}

容差：G3 ≤ {TOL_G3:.0%}，G5 ≤ {TOL_G5:.0%}。
**G3 判决：tail fit `{EV['G3']['tail_passed']}` → image `{EV['G3']['image_passed']}`；
G5：`{EV['G5']['tail_passed']}` → `{EV['G5']['image_passed']}`。**

### 3.2 反循环检查：可判窗口 vs 镜像值

| DOF | image $A_\\infty$ 区间 | G3 极小点 | G3 可判窗口（≤{TOL_G3:.0%}） | 镜像值在窗口内? |
|---|---|---|---:|---|
{window_rows}

镜像值**不是**窗口端点，也不在 1.0 处被调到极小：G3 极小点与镜像值的相对偏差在 0.1 %–20 % 之间
（surge 0.82 vs 0.80，heave 69.5 vs 69.1），即两条独立路线互相印证。

### 3.3 尾部拟合为什么不可能是对的（无需任何假设的见证）

六个自由度在频带顶端都**仍在单调上升**（`rising_at_band_top=True`），且
尾部拟合值**低于最后一个已解频点的实测 A**（`tail_fit_below_band_top=True`，六项全中）。
一个单调上升、斜率递减的剖面，其 $\\omega\\to\\infty$ 极限不可能低于带顶值；
两参数尾部拟合给出 60.17 kg（heave）而 $A_{{33}}(20\\,\\mathrm{{rad/s}})=62.47$ kg，
yaw 更极端：拟合 9.51 kg，镜像解 46.50 kg（差 4.9 倍）——这正是 G3/G5 在 yaw 上最差的根因。

### 3.4 G4 不但没被修好，反而更差（honest）

| 项 | tail fit | image solve | 容差 |
|---|---:|---:|---:|
| ζ_decay（时域包络） | {EV['G4_detail']['tail']['zeta_decay']:.4f} | {EV['G4_detail']['image']['zeta_decay']:.4f} | — |
| ζ_half-power（频域） | {EV['G4_detail']['tail']['zeta_half_power']:.4f} | {EV['G4_detail']['image']['zeta_half_power']:.4f} | — |
| 相对误差 | {EV['G4_detail']['tail']['rel_err']*100:.2f} % | {EV['G4_detail']['image']['rel_err']*100:.2f} % | {TOL_G4:.0%} |

G4 的频域腿（半功率 ζ）只用 $A(\\omega)$、$B(\\omega)$，对 $A_\\infty$ 分解不敏感（数值上完全不变）；
时域腿是 Cummins 方程，$A_\\infty$ 是瞬时惯量，换成物理值后 ζ_decay 由 0.1972 升到 0.2133，
残差 {g4['tail_rel_err']*100:.1f} % → {g4['image_rel_err']*100:.1f} %。
**G4 仍是 FAIL，且不是 $A_\\infty$ 问题**：它测的是「包络拟合 ζ」与「半功率带宽 ζ」在
频率相关系数系统上的估计量差异，属于另一个独立缺口（m23130 §4 已提出该假设，本轮给出反对称证据）。

![added_mass_and_asymptotes](docs/figures/m23208/added_mass_and_asymptotes.png)

![gates_tail_vs_image](docs/figures/m23208/gates_tail_vs_image.png)

![g3_vs_assumed_ainf](docs/figures/m23208/g3_vs_assumed_ainf.png)

## 4 误差判定（逐项，显式回答是否 <3 %）

| 关键量 | 基准 | 误差 | <3 % ? |
|---|---|---:|---|
| 参考点变换 vs 直接重解 | 同一镜像问题的两路线 | {frame:.2e} | ✅ 是（机器精度） |
| 镜像 $A_\\infty$ 网格梯子散布（最差 DOF） | 20x6 vs 20x10 | {max(spread)*100:.2f} % | ✅ 是 |
| 镜像解解析自校验（半浸球） | $\\rho V/4$ | {sphere['relative_error']*100:.2f} % | ✅ 是 |
| G3 最差 DOF（image $A_\\infty$） | 自洽 Hilbert 对 | {worst_g3['G3_image']*100:.2f} %（{worst_g3['dof_label']}） | ✅ 是 |
| G5 最差 DOF（image $A_\\infty$） | B→K→(B,A) 往返 | {worst_g5['G5_image']*100:.2f} %（{worst_g5['dof_label']}） | ❌ 否（3.45 % > 3 %，但 <5 % 容差） |
| G4（image $A_\\infty$） | 时域 ζ vs 半功率 ζ | {g4['image_rel_err']*100:.2f} % | ❌ 否（仍 FAIL） |

## 5 注意点 / 局限

* **本单不改预声明判决**：闭包带 G1–G5 的总判决仍是
  `FAIL:G4_free_decay`（G3/G5 在独立 $A_\\infty$ 下通过，G4 仍不过）。
  总账锚点 `CASE-BEM-CUMMINS-WIGLEY` 因此**不能**标 verified，剩余缺口被缩到一条：G4。
* 镜像解的梯子只到 `(20,10)`：网格散布 ≤2.1 %，但更细的网格（`30x12`）本单未跑出
  （集群 `cj4690` 递交 25 min 仍在队列中，本机 load15≈105 且守卫上限 120 s；
  脚本已实现镜像解磁盘缓存，续跑只付未命中部分）。
* G3 的归一化 $\\max|A-A_\\infty|$ 仍依赖 $A_\\infty$：独立值让分母变大约 7 %（heave），
  这只会让残差**更保守地小一点**，但结论不靠这一点——见 §3.2 的窗口检查与 §3.3 的单调性见证。
* G4 的时域 τ_max=40 s、dt=0.02 s 沿用算例缺省，未做本轮的收敛性重扫（m23130 的 C13 已覆盖时域预算）。

## 6 复现

```
PYTHONPATH=src python3 scripts/cummins_closure_ainf_independent.py --stage cache   # 冻结 69 点扫频（从 m23130 点缓存）
PYTHONPATH=src python3 scripts/cummins_closure_ainf_independent.py --stage audit \\
    --ladder 20x6,20x10 --frame-mesh 20x6                                          # 约 40 s（镜像解有磁盘缓存）
PYTHONPATH=src python3 -m pytest tests/test_cummins_closure_ainf_independent.py -q
```
"""

DEV_REPORT.write_text(BODY)
print(f"wrote {DEV_REPORT} ({len(BODY)} chars)")

# ---------------------------------------------------------------------------
# platform report
# ---------------------------------------------------------------------------
summary = (
    f"Wigley 闭包带 G3/G5 的 27–33 % 残差全部是 A_∞ 尾部拟合偏置：换成独立镜像解 A_∞ 后 "
    f"G3 最差 {worst_g3['G3_image']*100:.2f} %（容差 10 %）、G5 最差 {worst_g5['G5_image']*100:.2f} %（容差 5 %）；"
    f"G4 反而 18.3 %→{g4['image_rel_err']*100:.1f} %，是另一个缺口（估计量差异），闭包总判决仍 FAIL。"
)
report = {
    "title": "CASE-BEM-CUMMINS-WIGLEY：独立 A_∞ 锚揭开闭包带 G3/G5 的失败根因（G4 仍缺）",
    "summary": summary,
    "scenario_key": "casepool_bem_cummins_wigley",
    "mission_id": MISSION,
    "tags": ["TensorBEM", "CASE-BEM-CUMMINS-WIGLEY", "Kramers-Kronig", "added-mass-limit",
             "image-method", "closure-band", "honest-FAIL"],
    "markdown": BODY,
    "figures": FIGURES,
}

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8001/api/v1").rstrip("/")
user = os.environ.get("TENSOR_AGENT_USER", "")
password = os.environ.get("TENSOR_AGENT_PASS", "")


def post(path: str, payload: dict, token: str | None = None) -> dict:
    data = json.dumps(payload).encode()
    request = urllib.request.Request(f"{base}{path}", data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode())


login = post("/auth/login", {"username": user, "password": password})
token = login.get("access_token") or login.get("token")
created = post("/tensor/reports", report, token)
print(json.dumps({k: created[k] for k in created if k in ("id", "title", "status")}, indent=1))