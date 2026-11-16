"""Publish the m22587 report: figure from committed artifacts + API post."""
import base64, json, os, urllib.request, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path("/srv/tensor/agent/work/m22587")
CASE = ROOT / "benchmarks/verified/casepool_bem_cummins_wigley"
case_json = json.loads((CASE / "case_result.json").read_text())
probe = json.loads((CASE / "extended_probe_report.json").read_text())
cost = json.loads((CASE / "extended_probe_cost.json").read_text())

c3 = case_json["gates"]["C3"]["values"]["per_dof"]
dof = [r["dof_name"] for r in c3]
p12 = [r["damping_tail_exponent"] for r in c3]
p20c = {r["dof_name"]: r["exponent_at_20_coarse"] for r in probe["per_dof"]}
p20m = {r["dof_name"]: r["exponent_at_20_medium"] for r in probe["per_dof"]}

fig, ax = plt.subplots(figsize=(7.2, 4.0))
x = np.arange(len(dof)); w = 0.26
ax.bar(x - w, p12, w, label="p at 12 rad/s (committed, certifiable test)", color="#444")
ax.bar(x, [p20c[d] for d in dof], w, label="p at 20 rad/s (probe, coarse 192 pan.)", color="#7aa6c2")
ax.bar(x + w, [p20m[d] for d in dof], w, label="p at 20 rad/s (probe, medium 480 pan.)", color="#c28b7a")
ax.axhline(2.05, color="crimson", lw=1.2, ls="--")
ax.text(len(dof) - 0.5, 2.2, "algebraic-decay bar p>2.05", color="crimson", ha="right", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(dof); ax.set_ylabel("damping tail exponent p")
ax.set_title("C9 question: does the omitted band stay bounded? (12 vs 20 rad/s)")
ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=.3, axis="y")
fig.tight_layout(); png = ROOT / "docs/case_bem_cummins_wigley_extended_band_gap_m22587.png"
fig.savefig(png, dpi=150)

digest = {}
for r in c3:
    digest[r["dof_name"]] = dict(p12=r["damping_tail_exponent"],
                                 admissible=r["truncation_budget_admissible"],
                                 kk=100.0 * r["max_relative_residual"])
c3a = case_json["gates"]["C3a"]["values"]
worst_kk = 100.0 * case_json["gates"]["C3"]["values"]["worst_relative_residual"]
verdict = case_json["verdict"]

rows12 = "\n".join(
    f"| {d} | {digest[d]['p12']:.2f} | {'yes' if digest[d]['admissible'] else 'no'} | {digest[d]['kk']:.2f}% |"
    for d in dof)
probe_rows = "\n".join(
    f"| {r['dof_name']} | {r['exponent_at_12']:.2f} | {r['exponent_at_20_coarse']:.2f} | "
    f"{r['exponent_at_20_medium']:.2f} | {r['probe_verdict'].split(':')[0]} |"
    for r in probe["per_dof"])

md = f"""# CASE-BEM-CUMMINS-WIGLEY：扩展带装配链路缺陷修复与 C8/C9 证据呈现

**Mission m22587 ｜ 仓 TensorBEM ｜ 锚点 `CASE-BEM-CUMMINS-WIGLEY`（算例池·Wigley 船 Cummins 记忆核 6DOF）**

## 1 案例介绍

Wigley 数学船型 $y=(B/2)(1-(2x/L)^2)(1-(z/T)^2)$，$L=3.0$ m、$B=0.3$ m、$T=0.1875$ m，
$C_b=4/9$、排水体积 $0.075\\ \\mathrm{{m^3}}$，$\\rho=1025\\ \\mathrm{{kg/m^3}}$，$g=9.80665$。
算例判据：Cummins 时域卷积 + Kramers–Kronig 因果对，容差 10%（`tensor_case_pool id=145`）。
已提交判据 **{verdict['gates_passed']}/{verdict['gates_total']}**，失效列表 `{verdict['failed_list']}`。

已知欠账（前序 round 9 记录）：12 rad/s 频带上 surge/sway/yaw 的阻尼尾部指数 0.28–1.62
未进入代数衰减，截断预算 $\\int^\\infty B(s)\\,ds/(s^2-\\omega^2)$ 不收敛，三个自由度的 C3a
残差只是经验一致而非带限数据蕴含。

## 2 怎么做的

本单不去重跑物理，而是核对「把扩展带结果接回判据」的那一端，结果发现它**静默失效**：

1. `scripts/assemble_cummins_extended.py` 读 `data.get("residual", 0.0)`，而求解器写的键是
   `solver_relative_residual` → 装配出的 sweep 残差全 0.0，闸门 C0（判据 `max residual ≤ 1%`）
   会把**未收敛**的 sweep 判为收敛。仓内 m20021 事故正是同一类伪证。
2. 同一脚本把 `gates_from_artifact(<扩展 sweep>)` 当作重判入口 → C8 需要「已提交 12 rad/s 频带
   vs 扩展频带」两条曲线，自比自没有意义；`C9.band_at_12` 会被写成 20。
3. `existing["gates"] = result["gates"]` 整体覆盖 → 已发布的 C5b/C5c/C10 被静默删除。
4. 不校验分片网格一致性、按文件名而非频率排序。
5. `render_markdown` 无 C8/C9/C3a_extended 的数值呈现：即便闸门挂上，`CASE.md` 只有一行 PASS。

改动：重写装配/重判（读真残差键、拒绝混网格、按 $\\omega$ 排序、重判对象改为 `sweep_medium.npz`
并显式 `extended_path=`、只替换 gates/verdict/extended 三个字段并记 `gate_replay` 出处块、
`CASE.md` 仍由 `render_markdown(JSON)` 生成）；`render_markdown` 新增「## 4b Extended band」
逐自由度表（两端带顶斜率、预算可判性、可判样本数、锚定残差），**无工件时不出现**，
故已提交 `CASE.md` 逐字节不变。

工具链：`PYTHONPATH=src python3 -m pytest tests/test_case_bem_cummins_wigley.py`（73 passed, 109 s）。

## 3 计算结果（程序化注入，非手抄）

已提交 12 rad/s 频带的尾部指数与截断预算可判性（来源 `case_result.json` → `gates.C3`）：

| DOF | p @12 rad/s | 预算可判 | KK 残差 |
| --- | --- | --- | --- |
{rows12}

C3 最差自由度残差 **{worst_kk:.2f}%**（判据 10%）；C3a（零自由参数、镜像锚点）最差
**{c3a['worst_dof']} {100*c3a['worst_relative_residual']:.2f}%**。

扩展带探针（`extended_probe_report.json`，192/480 面板，非 C2 合规，故只作界不作判据）：

| DOF | p@12 | p@20 粗 | p@20 中 | 探针裁决 |
| --- | --- | --- | --- | --- |
{probe_rows}

成本（`extended_probe_cost.json` 实测拟合 $t\\sim N^{{1.93}}$）：60×16（3840 面板）单频点
≈ {cost['certified_wall_s_per_node']:.0f} s，55 点 ≈ {cost['certified_cpu_hours_for_band']:.2f} CPU·h。

## 4 结果图

图中灰柱是已提交频带的实测尾部指数，蓝/棕柱是探针在 20 rad/s 的两种网格读数，红线是
代数衰减判据 $p>2.05$。它把「为什么需要扩展带」与「为什么探针不能代替判据」画在一起：
heave/pitch/yaw 的两块网格读数跨在判据线两侧，正是 round 9 判它们 inconclusive 的原因。

![gap](fig)

## 5 注意点 / 局限（honest）

* **扩展带工件仍未落盘**。本单递交的 `cj4241`（55 分片 × 3840 面板，0.2–20 rad/s）与
  `cj4242`（44×14 / 60×16 镜像锚点）在收单时仍 `queued`（age 15–17 min），未产出任何分片，
  因此 `C8/C3a_extended/C9` 状态**仍是 not_run**，`CASE.md` 仍是 {verdict['gates_passed']}/{verdict['gates_total']}。
  本单不宣称任何扩展带结论；凡「扩展带已通过」的表述在工件落盘前都不成立。
* 因此本单**没有新的物理数字**：表中所有数值均来自已提交工件，本单交付的是「让真数据能接上判据」
  的那段代码 + 4 项封住静默失效的测试。
* 探针网格违反 C2（$k\\sqrt{{A_p}}>1$），其 20 rad/s 读数只是物理上界，不构成判据——这正是
  round 9 已经写明、本单未推翻的界限。
* 装配脚本仍会在缺分片时报缺失数并退出，不做部分装配：55 点频带上漏点会让 C3 的分段线性
  插值积分偏离，这个误差没有声明过。

## 6 误差判定

| 关键量 | 值 | 基准 | 误差 | <3%？ |
| --- | --- | --- | --- | --- |
| C3 KK 闭合（最差 DOF） | {worst_kk:.2f}% | 算例容差 10% | 余量 {10-worst_kk:.2f} pp | — （判据是 10%，非 3%） |
| C3a 零自由参数闭合（最差 {c3a['worst_dof']}） | {100*c3a['worst_relative_residual']:.2f}% | 算例容差 10% | 余量 {10-100*c3a['worst_relative_residual']:.2f} pp | — |
| 本单新增物理量 | 无 | — | — | **不适用**：本轮未计算任何新物理量（工件排队中），故无误差可判 |

诚实结论：本单是**缺陷修复 + 可读性补全**，判据集与已提交数字一字未改；
扩展带三个闸门仍待集群工件到期后由 `scripts/assemble_cummins_extended.py` 一次性挂上。
"""
(ROOT / "docs/report_m22587.json").write_text(json.dumps({
    "title": "CASE-BEM-CUMMINS-WIGLEY：扩展带装配链路缺陷修复与 C8/C9 证据呈现（m22587）",
    "summary": "修复装配端点 4 处静默失效（残差键误读致 C0 伪证、自比自重判、闸门整体覆盖、混网格/乱序），render_markdown 新增逐自由度扩展带节；73 测试通过；扩展带工件仍排队，C8/C9 维持 not_run 不虚报。",
    "scenario_key": "CASE-BEM-CUMMINS-WIGLEY",
    "mission_id": 22587,
    "tags": ["TensorBEM", "seakeeping", "Cummins", "Kramers-Kronig", "case-pool", "defect-fix"],
    "markdown": md,
    "figures": [{
        "name": "extended_band_gap_m22587.png",
        "caption": "已提交 12 rad/s 频带的阻尼尾部指数 vs 探针在 20 rad/s 的两种网格读数；红线为代数衰减判据 p>2.05。",
        "png_base64": base64.b64encode(png.read_bytes()).decode(),
    }],
}, ensure_ascii=False))

base = os.environ.get("TENSOR_API_BASE", "http://localhost:8001/api/v1")
def post(url, payload, token=None):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read().decode()
    except Exception as exc:  # noqa: BLE001
        return "ERR", str(exc)[:400]

status, body = post(f"{base}/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                                           "password": os.environ["TENSOR_AGENT_PASS"]})
print("login", status, body[:200])
token = None
try:
    blob = json.loads(body)
    token = blob.get("access_token") or blob.get("token") or blob.get("data", {}).get("access_token")
except Exception:
    pass
if token:
    payload = json.loads((ROOT / "docs/report_m22587.json").read_text())
    print("report", *post(f"{base}/tensor/reports", payload, token))
else:
    print("no token")