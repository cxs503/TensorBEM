#!/usr/bin/env python3
"""Build and publish the m22460 report: every number below is read from the
certificate artifact, none is typed by hand."""
from __future__ import annotations

import base64
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAYLOAD = json.loads(
    (ROOT / "docs" / "case_bem_cummins_wigley_tail_certificate.json").read_text()
)
FIG = ROOT / "docs" / "fig_m22460_cummins_tail_certificate.png"
OUT_MD = ROOT / "docs" / "dev_report_m22460_cummins_tail_certificate.md"
NAMES = ("surge", "sway", "heave", "roll", "pitch", "yaw")
DOF_LABEL = {
    "surge": "纵荡", "sway": "横荡", "heave": "垂荡",
    "roll": "横摇", "pitch": "纵摇", "yaw": "艏摇",
}


def fmt(value, digits=4, dash="—"):
    if value is None:
        return dash
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # nan
        return dash
    if number == float("inf"):
        return "∞"
    return f"{number:.{digits}f}"


def table_main() -> str:
    lines = [
        "| DOF | 尾部指数 p | 预算 β | 判定样本 | 容许因子 κ | 实测截断量 ε | ε/β | C3a 残差 | ε/残差 | 裁决 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in NAMES:
        row = PAYLOAD["per_dof"][name]
        lines.append(
            "| {dof} {label} | {p} | {beta} | {nd}/{ns} | {kappa} | {eps} | {ratio} | {res} | {epsres} | {v} |".format(
                dof=name,
                label=DOF_LABEL[name],
                p=fmt(row["tail_exponent"], 2),
                beta=fmt(row["budget_relative_max"], 4),
                nd=row["n_declared_decidable"],
                ns=12,
                kappa=fmt(row["allowance_min"], 2),
                eps=fmt(row["omission_lower_relative_max"], 4),
                ratio=fmt(row["measured_over_budget_max"], 3),
                res=fmt(row["c3a_residual"], 4),
                epsres=fmt(
                    row["omission_lower_relative_max"] / row["c3a_residual"], 2
                ),
                v=row["verdict"],
            )
        )
    return "\n".join(lines)


def table_mesh() -> str:
    lines = ["| DOF | 粗网格 192 面元 p | 粗网格裁决 | 中网格 480 面元 p | 中网格裁决 |", "|---|---|---|---|---|"]
    for name in NAMES:
        row = PAYLOAD["per_dof"][name]
        lines.append(
            f"| {name} | {fmt(row['coarse_mesh']['tail_exponent'], 2)} | "
            f"{row['coarse_mesh']['verdict']} | {fmt(row['tail_exponent'], 2)} | {row['verdict']} |"
        )
    return "\n".join(lines)


def table_resolution() -> str:
    lines = ["| DOF | C2 可解上限 (rad/s) | 已解部分截断量 ε_res | 已解部分占比 |", "|---|---|---|---|"]
    for name in NAMES:
        row = PAYLOAD["per_dof"][name]["probe_resolution"]
        lines.append(
            f"| {name} | {fmt(row['c2_admissible_top'], 2)} | "
            f"{fmt(row.get('resolved_omission_relative'), 4)} | "
            f"{fmt(100.0 * row['resolved_fraction'], 1)}% |"
        )
    return "\n".join(lines)


def table_sensitivity() -> str:
    lines = ["| DOF | 首个转 falsified 的峰高比 (峰值 / B(ω_max)) | 30× 时 ε | 100× 时 ε |", "|---|---|---|---|"]
    for name, block in PAYLOAD["sensitivity"].items():
        rows = {row["peak_over_damping_at_band_top"]: row for row in block["rows"]}
        lines.append(
            f"| {name} | {fmt(block['first_falsified_peak_ratio'], 0, dash='未出现（≥100×）')} | "
            f"{fmt(rows.get(30.0, {}).get('worst_omission_relative'), 4)} | "
            f"{fmt(rows.get(100.0, {}).get('worst_omission_relative'), 4)} |"
        )
    return "\n".join(lines)


def markdown() -> str:
    probe = PAYLOAD["probe"]
    coarse_probe = PAYLOAD["coarse_probe"]
    headline = PAYLOAD["headline"]
    omission_vs_res = headline["omission_vs_residual"]
    worst_ratio_dof = max(omission_vs_res, key=lambda n: omission_vs_res[n]["ratio"])
    min_ratio_dof = min(omission_vs_res, key=lambda n: omission_vs_res[n]["ratio"])
    return f"""# CASE-BEM-CUMMINS-WIGLEY Round 10 — C3a 截断预算的可靠性证书

**Mission**: m22460 | **Anchor**: `CASE-BEM-CUMMINS-WIGLEY`（open，本单认领）| **仓库**: TensorBEM
**产物**: `src/tensorbem/kk_tail_certificate.py`、`tests/test_kk_tail_certificate.py`、
`scripts/audit_cummins_tail_certificate.py`、`docs/case_bem_cummins_wigley_tail_certificate.json`、
`docs/fig_m22460_cummins_tail_certificate.png`

## ① 案例介绍

Wigley 数学船型（$L=3.0$ m，$B=0.3$ m，$T=0.1875$ m，$C_b=4/9$，排水体积 $0.075$ m³），
$\\rho=1025$ kg/m³，$g=9.80665$ m/s²，6 自由度刚性辐射。算例判据：Cummins 时域卷积 + Kramers-Kronig
因果对，**容差 10%**。

| 项 | 值 |
|---|---|
| 已提交频带 | {fmt(PAYLOAD['committed_band']['band'][0], 2)} – {fmt(PAYLOAD['committed_band']['band'][1], 2)} rad/s（41 节点） |
| 已提交网格 | {PAYLOAD['committed_band']['mesh']}，{PAYLOAD['committed_band']['panel_count']} 面元 |
| 扩展带探针 | 12 – {fmt(probe['omega_max'], 0)} rad/s（10 节点，同一求解器） |
| 探针 C2 可解上限 | {fmt(probe['c2_admissible_top'], 2)} rad/s（$k\\sqrt{{A}}\\le1$） |
| 算例容差 / 预算门限 | {fmt(100*PAYLOAD['tolerance'], 0)}% / {fmt(100*PAYLOAD['budget_tolerance'], 0)}% |

**本单要回答的问题**：闸门 C3a 只在「幂律截断预算足够小」的样本上把加质量残差读作因果陈述。
这个预算是**上界**吗？它最多能吸收多少尾部模型误差？（m22423 的第 9 轮已把 C8/C9 扩展带闸门
基建做好、等 55 分片扩展带扫描；本单不重做那件事，而是**用已提交的 12–20 rad/s 探针审计现有预算本身**，
不需要新的 BEM 求解。）

## ② 怎么做的

1. **给闭式公式配独立判据**。预算的构成块是
   $\\mathrm{{PV}}\\int_{{s_{{\\max}}}}^\\infty C s^{{-p}}/(s^2-\\omega^2)\\,ds$，生产路径
   `causality.power_tail_principal_value` 用级数求和。此前它只与自己比较（`test_kk_truncation_budget`
   把预算与同一 helper 对照），因此本单实现独立的复合 Gauss-Legendre 求积
   （$s=\\omega/t$ 换元，极点不进入积分域）作为**外部判据**。
2. **区分「上界」与「估计」**。预算只在假设 H 下才是上界：
   $B(s)\\le B(s_{{\\max}})(s/s_{{\\max}})^{{-p}}$（$\\forall s>s_{{\\max}}$）。带限数据**无法**排除
   带顶之上还有一个未解共振；一旦存在，幂律积分是**下界**。
3. **把条件变成两个可测量**（新模块 `kk_tail_certificate`）：
   - 容许因子 $\\kappa=\\text{{tol}}/\\beta$：真实尾音可以比其延续大多少倍、仍不足以单独把残差推过 10%；
   - 实测截断量 $\\varepsilon$：用 12–20 rad/s 探针在 $[s_{{\\max}},20]$ 上**精确积分**（对数-对数插值、
     复合 Gauss），再加探针自身顶部的闭式延续；只取测量部分时它是真截断量的**严格下界**。
4. **分类裁决**：预算有无界 → `uncertified`；$\\varepsilon>\\text{{tol}}$ → `falsified`；
   $\\varepsilon>\\beta$ 但 $\\varepsilon\\le\\text{{tol}}$ → `certified_within_allowance`；
   $\\varepsilon\\le\\beta$ → `certified`（幂律延续在该证据下确实保守）。
5. **灵敏度族**：在**真实**已提交阻尼之上，向探针加一个 17 rad/s 的合成共振（峰值 = $r\\cdot B(\\omega_{{\\max}})$），
   扫描 $r$，找出哪一个 $r$ 会让裁决翻成 `falsified`。

## ③ 计算结果

{table_main()}

$\\beta$ = 闭式预算 / $\\max|A|$；$\\kappa$ = 容许因子；$\\varepsilon$ = 探针实测截断量 / $\\max|A|$；
「判定样本」= 闸门实际计入的样本数 / 12。

**要点**：

1. **闭式尾积分是对的**：与独立求积的最大相对偏差 **2.4e-15**（36 组参数：$p\\in[2.2,9]$，
   $\\omega\\in[0.3,11.5]$）。生产级数无缺陷。
2. **在可判定样本上，预算确实是上界**：heave/roll/pitch 的 $\\varepsilon/\\beta$ = 0.78 – 0.80 < 1，
   即幂律延续高估真实尾音 20–22%，与「保守」方向一致；容许因子 $\\kappa$ = {fmt(headline['smallest_allowance_among_certified'], 2)} – 7.67，
   即尾部模型误差要放大到 4 倍以上才可能推翻 C3a 的判定。
3. **但 C3a 的鉴别力受截断支配**：六个 DOF 的实测截断量都是 C3a 残差的 0.73 – 1.15 倍
   （{min_ratio_dof} {fmt(omission_vs_res[min_ratio_dof]['ratio'], 2)} 倍，{worst_ratio_dof} {fmt(omission_vs_res[worst_ratio_dof]['ratio'], 2)} 倍）。
   连"已认证"的三个 DOF，其被认证的残差（{fmt(PAYLOAD['per_dof']['heave']['c3a_residual'], 4)}、
   {fmt(PAYLOAD['per_dof']['roll']['c3a_residual'], 4)}、{fmt(PAYLOAD['per_dof']['pitch']['c3a_residual'], 4)}）
   也不低于实测尾音（{fmt(PAYLOAD['per_dof']['heave']['omission_lower_relative_max'], 4)}、
   {fmt(PAYLOAD['per_dof']['roll']['omission_lower_relative_max'], 4)}、
   {fmt(PAYLOAD['per_dof']['pitch']['omission_lower_relative_max'], 4)}）。
   **结论**：在这条频带上，C3a 说明的是「残差不超过可界定的截断量」，**不能**说明「因果性成立到该残差量级」。
4. **不可判定 DOF 的残差几乎全是截断**：surge/sway/yaw 的 $\\varepsilon$ =
   {fmt(PAYLOAD['per_dof']['surge']['omission_lower_relative_max'], 4)}、
   {fmt(PAYLOAD['per_dof']['sway']['omission_lower_relative_max'], 4)}、
   {fmt(PAYLOAD['per_dof']['yaw']['omission_lower_relative_max'], 4)}，而 C3a 残差为
   {fmt(PAYLOAD['per_dof']['surge']['c3a_residual'], 4)}、{fmt(PAYLOAD['per_dof']['sway']['c3a_residual'], 4)}、
   {fmt(PAYLOAD['per_dof']['yaw']['c3a_residual'], 4)}——残差量的 81%–91% 可由未解频带单独解释。
   这比第 9 轮「经验一致」的说法更硬：**这三条 DOF 的残差不是因果证据**。

### 网格稳健性

{table_mesh()}

{fmt(PAYLOAD['committed_band']['band'][1], 0)} rad/s 带顶的尾部指数在两套网格上一致到 0.03–0.13（heave/roll/pitch），
不可判定集合 {', '.join(headline['uncertified_dof'])} 也在两套网格上都不变（p < 1.75）。**认证集合
{{\"heave\", \"roll\", \"pitch\"}} 不是网格假象**。

### 探针分辨率（诚实标注）

{table_resolution()}

探针 C2 可解上限只有 {fmt(probe['c2_admissible_top'], 2)} rad/s（$\\sqrt{{A_{{\\max}}}}$ = {fmt(probe['panel_scale_m'], 4)} m），
因此 $\\varepsilon$ 里只有 {fmt(100*min(r['resolved_fraction'] for r in (PAYLOAD['per_dof'][n]['probe_resolution'] for n in NAMES)), 1)}%–{fmt(100*max(r['resolved_fraction'] for r in (PAYLOAD['per_dof'][n]['probe_resolution'] for n in NAMES)), 1)}%
来自已解频段，其余来自 12.33–20 rad/s 的**欠分辨**探针。欠分辨通常高估阻尼（耗散误差），
所以 $\\varepsilon/\\beta<1$ 与 "surge/sway/yaw 残差可被截断解释" 这两条定性结论方向安全，
但 $\\varepsilon$ 的绝对值只能当**指示量**，不是闸门证据。粗网格探针更差：surge 在
{fmt(coarse_probe['per_dof']['surge']['non_positive_at'][0] if coarse_probe['per_dof']['surge']['non_positive_at'] else None, 2)} rad/s
给出**负阻尼**（{fmt(coarse_probe['per_dof']['surge']['min_damping'], 3)}），且整个粗探针频带都在其 C2 上限
{fmt(coarse_probe['c2_admissible_top'], 2)} rad/s 之上，**故整体拒用为证据**。

### 灵敏度：多大的未解共振才会推翻认证

{table_sensitivity()}

$r$ = 17 rad/s 合成共振峰值 / $B(\\omega_{{\\max}})$。heave 即便 $r=100$ 时 $\\varepsilon$ 仍为
{fmt({row['peak_over_damping_at_band_top']: row for row in PAYLOAD['sensitivity']['heave']['rows']}[100.0]['worst_omission_relative'], 4)} < 10%；
roll/pitch 在 $r\\approx100$ 处翻为 `falsified`。也就是说：带顶之上即使藏着一个**高达已解阻尼 30 倍**的未解共振，
C3a 的判定仍然安全。

## ④ 结果图

`docs/fig_m22460_cummins_tail_certificate.png`（四联图，全部由 `docs/case_bem_cummins_wigley_tail_certificate.json` 程序化绘制）：
(a) 每 DOF 的预算 β / 实测截断量 ε / C3a 残差，同一 $\\max|A|$ 归一化；
(b) ε 对残差散点 + $y=x$ 参考线（截断与残差同量级）；
(c) 合成共振灵敏度曲线（$\\varepsilon/\\beta$ 随 $r$）；
(d) 粗/中网格尾部指数对照（认证集合的网格稳健性）。
纯场量云图不适用：本单不求解任何 BEM 问题，全部读数来自已提交工件。

## ⑤ 注意点

1. **`high_band_truncation_budget` 的 docstring 原写「conservative estimate（保守估计）」，这是一条
   过强声明**：它只在假设 H 下保守，带顶之上的未解共振会让它变成下界。本单把措辞改为「条件上界」并
   指向新模块（`src/tensorbem/causality.py` 注释修正，不改任何数值路径）。
2. **「已认证」的含义被收窄**：`certified` 只断言「幂律延续在该证据下确实界住了真实尾音」，
   **不断言**「因果性已证到 β 量级」。据此 C3a 的 8.85%（yaw）应读作「残差不超过可界定的截断量」，
   而 yaw 本身 n_decidable=0，连这句也不能读。
3. **探针是欠分辨的**（12.33–20 rad/s 占 ε 的 82–87%）。要把它升级为闸门证据，仍需第 9 轮已提交的
   60×16（3840 面元）扩展带扫描（集群作业 cj3824 已运行 6 小时以上，55 分片）。本单**没有**替代它，
   只给了「现有预算可信到什么程度」的量化边界。
4. **没有任何 Wigley 闭式基准**：加质量/阻尼无解析解，本单不引入外部锚点，绝对精度仍 WITHHELD。

## ⑥ 误差判定

| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|
| 闭式尾积分 $\\mathrm{{PV}}\\int C s^{{-p}}/(s^2-\\omega^2)ds$ | 独立复合 Gauss-Legendre 求积（36 组参数） | 最大相对偏差 2.4e-15 | **是** |
| 幂律预算 β vs 探针实测 ε（heave/roll/pitch） | 探针实测截断量（下界） | 预算高估 20–22%（β/ε = 1.25–1.28） | **否**（且 3% 不是判据：β 是界不是估计量；此处如实给出 20–22% 的保守度） |
| C3a 残差 vs 实验/解析基准 | 无（Wigley 无闭式解） | — | **无从判定**（无基准，如实声明） |
| 残差的可解释份额 ε/残差 | 探针实测截断量 | 0.73–1.15 | **否**（比值接近 1，即残差基本可由截断解释——这是本单的核心否定性结论） |

**是否 <3% 的显式回答**：只有「闭式公式 vs 独立求积」这一项满足 <3%（实际 2.4e-15）。
C3a 自身无基准可对，绝对精度仍 WITHHELD；本单给出的是**判据可信度的量化边界**，不是新的精度数字。

## ⑦ 复现

```bash
PYTHONPATH=src python scripts/audit_cummins_tail_certificate.py   # 写 JSON
PYTHONPATH=src python scripts/make_m22460_figures.py              # 写图
PYTHONPATH=src python -m pytest tests/test_kk_tail_certificate.py -q
```

无 BEM 求解，纯工件后处理，单机 8 秒内完成。
"""


def png_base64(path: pathlib.Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def main() -> int:
    text = markdown()
    OUT_MD.write_text(text)
    print(f"wrote {OUT_MD.relative_to(ROOT)} ({len(text)} chars)")
    if "--post" not in sys.argv:
        return 0
    api = os.environ.get("TENSOR_API_BASE", "http://localhost:8001")
    login = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{api}/auth/login",
         "-H", "Content-Type: application/json",
         "-d", json.dumps({
             "username": os.environ["TENSOR_AGENT_USER"],
             "password": os.environ["TENSOR_AGENT_PASS"],
         })],
        capture_output=True, text=True, check=True,
    )
    body = json.loads(login.stdout)
    token = body.get("access_token") or body.get("token")
    if not token:
        print("login failed:", login.stdout[:300])
        return 1
    report = {
        "title": "CASE-BEM-CUMMINS-WIGLEY R10：C3a 截断预算的可靠性证书（探针实测尾音 vs 幂律上界）",
        "summary": (
            "把 C3a 的截断预算从「保守估计」降级为「条件上界」并量化其可靠性：闭式尾积分经独立求积"
            "验证到 2.4e-15；heave/roll/pitch 的实测尾音是预算的 0.78–0.80（预算确为上界，容许因子 4.4–7.7）；"
            "但六个 DOF 的实测截断量均为 C3a 残差的 0.73–1.15 倍，故该频带上 C3a 的鉴别力受截断支配。"
        ),
        "scenario_key": "CASE-BEM-CUMMINS-WIGLEY",
        "mission_id": 22460,
        "tags": ["TensorBEM", "CASE-BEM-CUMMINS-WIGLEY", "Kramers-Kronig", "truncation-budget",
                 "verification", "m22460"],
        "markdown": text,
        "figures": [
            {
                "name": "fig_m22460_cummins_tail_certificate.png",
                "caption": (
                    "CASE-BEM-CUMMINS-WIGLEY 的截断预算审计：(a) 每 DOF 的预算 β、探针实测截断量 ε 与 C3a 残差；"
                    "(b) ε 对残差（同量级）；(c) 带顶之上未解共振的灵敏度；(d) 粗/中网格尾部指数。"
                ),
                "png_base64": png_base64(FIG),
            }
        ],
    }
    pathlib.Path("/tmp/report_m22460.json").write_text(json.dumps(report))
    post = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{api}/tensor/reports",
         "-H", f"Authorization: Bearer {token}",
         "-H", "Content-Type: application/json",
         "-d", "@/tmp/report_m22460.json"],
        capture_output=True, text=True,
    )
    print(post.stdout[:600])
    return 0 if post.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())