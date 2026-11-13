"""Build and POST the m18742 paper-style report with embedded figures."""
import base64
import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"]
U = os.environ["TENSOR_AGENT_USER"]
P = os.environ["TENSOR_AGENT_PASS"]
D = "docs/fv_order_xval"
res = json.load(open(os.path.join(D, "results.json")))
crit = res["criteria"]


def req(path, data=None, tok=None):
    body = json.dumps(data).encode() if data is not None else None
    r = urllib.request.Request(BASE + path, data=body,
                               method="POST" if data is not None else "GET")
    r.add_header("Content-Type", "application/json")
    if tok:
        r.add_header("Authorization", "Bearer " + tok)
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:800]


tok = req("/auth/login", {"username": U, "password": P})[1]["access_token"]

w = crit["C2_primary_order_15pct"]["worst"]
t = crit["C3_primary_order_3pct"]["worst"]
c7 = crit["C7_cross_implementation"]
c6 = crit["C6_asymptotic_range"]["rows"]
c8 = crit["C8_discrimination"]["rows"]
obs = res["observations"]["functional_route_blindness"]


def pct(x):
    return f"{100 * x:.4f}%"


prim_rows = "\n".join(
    f"| `{r['problem']}` | `{r['ladder']}` | {r['norm']} | {r['ref']:.3f} | {r['got']:.6f} | {pct(r['rel_err'])} |"
    for r in crit["C2_primary_order_15pct"]["rows"]
)
ctrl_rows = "\n".join(
    f"| `{r['variant']}` | {r['p_exact']:.0f} | {r['p_obs_l2']:.6f} | {pct(r['err_vs_own_target'])} | {pct(r['err_vs_primary_target'])} | {r['roache_functional']:.6f} |"
    for r in c8
)
c4_rows = "\n".join(
    f"| `{r['problem']}` | `{r['ladder']}` | {r['order_error_l2']:.6f} | {r['order_roache_functional']:.6f} | {pct(r['rel_diff'])} |"
    for r in crit["C4_estimator_agreement"]["rows"]
)
c6_rows = "\n".join(
    f"| `{r['problem']}` | {r['p_32_64_128']:.6f} | {r['p_64_128_256']:.6f} | {pct(r['rel_diff'])} | {r['asymptotic_ratio_fine']:.4f} |"
    for r in c6
)
blind_txt = "\n".join(
    f"- `{o['variant']}`: 误差型 p = {o['error_based_p']:.4f}，泛函型 p = {o['functional_p']:.4f}"
    + (" → **泛函盲区**" if o["functional_route_blind"] else " → 泛函也降级")
    for o in obs
)
errs = res["fv"]["sin_sin"]["r2_coarse"]["errors"]
levels = res["fv"]["sin_sin"]["r2_coarse"]["levels"]
err_tbl = "\n".join(
    f"| {n} | {e['l1']:.4e} | {e['l2']:.4e} | {e['linf']:.4e} |"
    for n, e in zip(levels, errs)
)

md = f"""# 单元中心有限体积「三网格 log-log 斜率」观测阶交叉验证（XVAL-FVM-THREE-GRID-LOGLOG-SLOPE-CELLCENTRE-FV-OR）

任务 m18742 · 仓库 TensorBEM · 契约判据：观测阶相对误差 <15%（owner 精度标准），同时报 <3% 线。

## ① 案例介绍

**物理背景**：这是纯验证（verification）问题——物理模型误差为 0，唯一被检验的离散量是**观测阶**
`p_obs`。被验证对象是单元中心有限体积（cell-centred FV）对 `-∇²u = f`（Ω=[0,1]²，全 Dirichlet）
的离散解。它等价于任何 FVM 求解器的“仪器标定”，因此其结果可迁移到 TensorFVM 域。

**几何/算例**：单位正方形，均匀 N×N 网格；两个制造解（manufactured solutions，解析解精确已知）：
`sin_sin`: u = sin(πx)sin(πy), f = 2π²u；`exp_sin`: u = e^x sin(πy), f = (π²−1)e^x sin(πy)。

**参数表**

| 参数 | 取值 |
|---|---|
| 网格阶梯 | r=2: 32/64/128（主）、64/128/256（渐近区间）、r=1.5: 40/60/90 |
| 格式 | `fv2`（中点源项 + 半格 Dirichlet 幽灵）→ p_exact=2 |
| 负对照 | `fv1_src`（角点矩形源项）、`fv1_bc`（边界全格距离）→ p_exact=1 |
| 独立路由 | scikit-fem 12.0.2 P1（p_exact=2）/ P2（p_exact=4）Galerkin |
| 范数 | 单元中心离散 L2（L1/L∞ 并列） |
| GCI 安全因子 | Fs = 1.25（Roache 三网格） |
| 线性求解 | 直接法（spsolve），无迭代容差 |

**基准出处**：解析解本身（误差 ref = 0）+ 理论阶 p_exact（Ferziger & Perić 2002 §3.3/§4.3-4.4；
Versteeg & Malalasekera 2007 §4.2-4.3/§5）+ 三网格观测阶公式（Roache 1994 JFE 116:405-413；
Roache 1998；Celik et al. 2008 JFE 130:078001 / ASME V&V 20-2009）。

## ② 怎么做的

**调研（第 2 轮要求）**：
- 本机实读 `/root/oss-cache`：只有 3 个仓（`pylbm` 194py、`Maarten-vd-Sande_lbm` 7py、`hschwane` 0py），全是 LBM；
  **没有** cell-centred FV Poisson 求解器，也没有 GCI/观测阶工具链。`pylbm` 的验证范式是“与解析
  Poiseuille 剖面对比”（`demo/2D/Poiseuille_vec.py` 文件头给出解析解），只有单网格 → 给误差但给不出阶。
  这正是本单补的一层：把“对比解析解”升级成“阶梯上的 log-log 斜率”。
- 仓内冻结语料 `docs/rci-pilot200-r2/frozen/corpus/TensorFVM/docs/oss-inventory-fvm/gap-proposals.md`
  列了 FVM 生态 8 条缺口与参考实现，同样**无**“观测阶验证”条目。
- 集群 `~/oss-cache` 的 SQL 检索作业 cj2009 递交后 12 min 仍在队列（提交时队列 139），未等，如实记录。
- 唯一有现成方法可直接学结构的 OSS：scikit-fem 12.0.2（已装），其 P1/P2 组装与按点插值被直接调用。

**实现**：`src/tensorbem/benchmarks/fv_order_xval.py`。面通量与源项记账：
Σ_f h(∂u/∂n)_f = f(x_P)h²，内部面 (u_N−u_P)/h，边界面 (u_b−u_P)/(h/2)。
三个估计器：`loglog_slope`（最小二乘 log e–log h）、`pairwise_slopes`、`three_grid_slope` /
`tensorbem.uq.grid_convergence_index`（泛函型 Roache 阶，复用既有实现）。
入口 `scripts/m18742_fv_order_xval.py run|verify`，预声明 10 条判据，`verify` 只读工件重判。

## ③ 计算结果（程序化注入，零手抄）

`fv2` 主格式，`sin_sin`，32/64/128：

| N | L1 | L2 | L∞ |
|---|---|---|---|
{err_tbl}

## ④ 结果图

图 1 误差–网格 log-log 曲线与拟合斜率；图 2 各路由观测阶 vs 理论阶（±15% 带）；图 3 解场与误差云图。

## ⑤ 注意点（踩过的坑）

- **泛函型 GCI 的盲区与算例有关**：{blind_txt}
  矩形求积的首阶误差有符号，在对称算例上区域平均时抵消 → 泛函超收敛；换非对称算例就露出来。
  所以“只在一个对称算例上看某个工程量收敛”的验证链可能对一阶缺陷完全无感。
- 三网格只有一次多余自由度，r² 天然≈1，不能单独作为渐近区间证据 → 加 64/128/256 与 Roache 渐近比值。
- 未覆盖：三维/非结构或非正交网格（TensorFVM 域，本仓无实现）；对流占优格式；只验 Poisson。
- `oss_solutions` 表在本租户 API 面（1324 条路由）无可写端点，方法选型沉淀为仓内
  `docs/fv_order_xval/oss_solutions.json`（带 feature_slug）。

## ⑥ 误差判定（逐项，显式回答）

**观测阶 vs 理论阶 p_exact = 2**（离散 L2）：

| 算例 | 阶梯 | 范数 | ref | got | 相对误差 |
|---|---|---|---|---|---|
{prim_rows}

- worst L2 单元：{pct(w["rel_err"])} → **<15%：{'是' if crit['C2_primary_order_15pct']['pass'] else '否'}**；
  **<3%：{'是' if crit['C3_primary_order_3pct']['pass'] else '否'}**（同 worst 单元 {pct(t['rel_err'])}）。
- 跨实现 FV↔P1：FV p = {c7['fv_p']:.6f}（vs 理论 {pct(c7['fv_vs_theory'])}），
  P1 p = {c7['skfem_p']:.6f}（vs 理论 {pct(c7['skfem_vs_theory'])}），互差 {pct(c7['rel_diff'])} → <15% 达标。
- 仪器的仪器：合成幂律 e=3.0h^1.7 上估计器回到 1.7，绝对误差 {res['procedure']['p_loglog_abs_err']:.1e}（容差 1e-10）。

**估计器一致性（误差型 vs 泛函型）**：

| 算例 | 阶梯 | 误差型 p | 泛函型 p | 相对差 |
|---|---|---|---|---|
{c4_rows}

**渐近区间 / 两分辨率敏感性**：

| 算例 | p(32/64/128) | p(64/128/256) | 相对差 | 渐近比值 |
|---|---|---|---|---|
{c6_rows}

**判别力（负对照必须返回 p=1）**：

| 变体 | ref | got | vs 自身 | vs 主目标 2 | 泛函型 p |
|---|---|---|---|---|---|
{ctrl_rows}

**10 条预声明判据全部通过**：{res['verdict']['n_pass']}/{res['verdict']['n_criteria']} —
finish 的 `checks` 中逐条列出；工件 `docs/fv_order_xval/results.json`
（sha256 {res['results_sha256'][:16]}…，两次独立运行 byte 相同）、
`benchmarks/verified/fv_cc_three_grid_orr/case.json`。
复现：`PYTHONPATH=src python3 scripts/m18742_fv_order_xval.py run --outdir docs/fv_order_xval`（本地约 2 min，无需集群）；
机器复核：`... verify --outdir docs/fv_order_xval`（退出码 0 = 全通过）。
"""

figs = []
for name, cap in [
    ("fv_order_loglog.png", "误差–网格 log-log 曲线；图例给出各路由拟合斜率 p_obs"),
    ("fv_observed_order.png", "各路由观测阶 vs 理论阶（黑线），灰带为 ±15% owner 容差"),
    ("fv_field_error.png", "N=128 单元中心 FV 解场与 log10|u_h−u_exact| 误差云图"),
]:
    with open(os.path.join(D, name), "rb") as fh:
        figs.append({"name": name, "caption": cap,
                     "png_base64": base64.b64encode(fh.read()).decode()})

payload = {
    "title": "单元中心 FV 三网格 log-log 斜率观测阶交叉验证（XVAL-FVM-THREE-GRID-LOGLOG-SLOPE-CELLCENTRE-FV-OR）",
    "summary": f"cell-centred FV 观测阶 p_obs={c7['fv_p']:.6f} vs 理论 2（误差 {pct(w['rel_err'])}，<15% 与 <3% 均达标）；"
               f"跨实现 scikit-fem P1 给出 {c7['skfem_p']:.6f}；两个一阶负对照回到 p≈1，"
               f"并暴露泛函型 GCI 在对称算例上的盲区；10/10 预声明判据通过",
    "scenario_key": "verification",
    "mission_id": 18742,
    "tags": ["TensorBEM", "FVM", "cell-centred", "observed-order", "log-log-slope",
             "GCI", "Roache", "XVAL", "verification", "manufactured-solution"],
    "markdown": md,
    "figures": figs,
}
status, out = req("/tensor/reports", payload, tok=tok)
print("POST status", status)
print(json.dumps(out, ensure_ascii=False)[:1200] if not isinstance(out, str) else out)