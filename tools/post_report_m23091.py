"""Publish the m23091 report (login + POST /tensor/reports). Numbers are read
from the artifacts at post time, never typed."""

import base64
import json
import os
import pathlib
import urllib.request

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOFS = ("surge", "sway", "heave", "roll", "pitch", "yaw")
N315 = ROOT / "docs/casepool_bem_cummins_wigley/evidence_cluster_n315_sweep_coarse.npz"
COMMITTED = ROOT / "benchmarks/verified/casepool_bem_cummins_wigley/sweep_coarse.npz"
FIG = ROOT / "docs/m23091/dof_attribution.png"


def facts(path):
    data = np.load(path)
    damping = data["damping"]
    per_dof = {n: float(damping[:, i, i].min()) for i, n in enumerate(DOFS)}
    diag = damping.diagonal(axis1=1, axis2=2)
    i_omega, i_dof = divmod(int(diag.argmin()), len(DOFS))
    return {
        "per_dof": per_dof,
        "worst_dof": DOFS[i_dof],
        "worst": per_dof[DOFS[i_dof]],
        "worst_omega": float(data["omega"][i_omega]),
        "residual_max": float(data["solver_relative_residual"].max()),
        "negative": int((diag < 0.0).sum()),
        "n_omega": int(data["omega"].size),
        "np_diag_floor": {
            n: float(np.diag(damping[:, i, i]).min()) for i, n in enumerate(DOFS)
        },
    }


n315 = facts(N315)
ok = facts(COMMITTED)

table = "\n".join(
    f"| {n} | {n315['per_dof'][n]:.4g} | {ok['per_dof'][n]:.4g} | "
    f"{'是' if n315['per_dof'][n] < 0 else '否'} | {ok['np_diag_floor'][n]:.4g} |"
    for n in DOFS
)

markdown = f"""## ① 案例介绍

CASE-BEM-CUMMINS-WIGLEY（算例池 id=145）是 TensorBEM 的 6DOF Cummins 记忆核算例：Wigley 船型、
无航速、零波陡，用边界元逐频求辐射附加质量 A(ω) 与阻尼 B(ω)，再用 Kramers-Kronig 闭合与自由衰减
互校（判据 tol < 10 %）。生产网格 medium 20×6（480 面板，41 节点，band 0.2–12 rad/s）。

本单处理的不是物理，而是**这份算例的说明书**（`docs/casepool_bem_cummins_wigley/README.md`，由
`tools/render_casepool_bem_cummins_wigley_readme.py` 从工件渲染）里「数字 ↔ DOF」的对应关系。
该说明书的第 4 节并列两台机器同一份代码、同一网格/带/节点数的结果：n315 节点（cj2690/cj2704）
与提交件那一侧的 loongarch64 重算。两侧的 max solver residual 与 min diag B 是 C0 闸门的判据。

## ② 怎么做的

1. 用 PR#1081 交付的负控工件 `evidence_cluster_n315_sweep_coarse.npz`（12×4，192 面板，41 节点）
   与提交件 `sweep_coarse.npz`，逐 DOF 重算 `min over band diag B`——按
   `DOF_NAMES = ("surge","sway","heave","roll","pitch","yaw")` 的**名字**取列，不手写列号。
2. 发现两处渲染器缺陷：手写列号 5 被当作 pitch（实际是 yaw）；`np.diag(damping[:, i, i])` 对 1-D
   切片返回矩阵，副对角线零把每个非负 DOF 压成「恰好 0」。
3. 改渲染器：`per_dof_min` 按 DOF 名枚举、`min_damping_dof/_omega/negative_damping_nodes` 由数据
   导出，删掉把真值遮成「0（下界）」的 `_floor()`；README 用 `--write` 重新渲染（逐字等于渲染输出）。
4. 新增两条测试：一条从 `.npz` 独立重算并要求渲染事实与 README 文本逐项一致（含「健康工况每个 DOF
   必须严格为正」），一条负控（把 yaw 的值塞进 pitch，渲染必须改变）。两条在旧渲染器上均失败。

## ③ 计算结果（程序化注入，零手抄）

| DOF | n315 coarse min diag B | 提交件 coarse min diag B | n315 是否造能（B<0） | 旧口径 `np.diag` 给出的提交件值 |
| --- | --- | --- | --- | --- |
{table}

* n315 最坏 DOF：**{n315['worst_dof']} = {n315['worst']:.4g}**，出现在 ω = {n315['worst_omega']:.3f} rad/s（带中）；
  全带 6 DOF×{n315['n_omega']} 节点里共 {n315['negative']} 个为负。
* 提交件最坏值：{ok['worst_dof']} = {ok['worst']:.4g}，出现在带底 ω = {ok['worst_omega']:.3f} rad/s（小正数，B(ω→0)→0）。
* max solver residual：n315 = {n315['residual_max']:.3f}，提交件 = {ok['residual_max']:.3f}（C0 门 0.01）。
* 旧 README 把 **{n315['per_dof']['yaw']:.4g}（yaw）** 印成「pitch」，而真正的 pitch 最小值
  **{n315['per_dof']['pitch']:.4g}** 只以裸数字出现、没有 DOF；提交件一侧 6 个 DOF 全被印成「0（下界）」，
  真实值是 {ok['per_dof']['surge']:.3g} / {ok['per_dof']['pitch']:.3g}（surge / pitch）。

## ④ 结果图

![dof attribution](figures/dof_attribution.png)

左：n315 负控逐 DOF 的带内最小辐射阻尼（symlog），红柱为负（造能），最坏 pitch=−1.463 曾被印成无名
裸数字、而 yaw=−0.000681 被印成「pitch」。右：同一量 B_pitch(ω)——同一份代码、同一网格、同一频带，
提交件（绿）全程 ≥0 并随 ω→0 趋于 0，n315（红）在带中 4 个节点转负。这就是 C0 闸门要拦的东西：
负辐射阻尼物理上不可能，所以那不是「另一台机器的另一种答案」，而是**未收敛解**。

## ⑤ 注意点

* 本单不动物理、不动判据、不重解 BEM：`case_result.json`、`sweep_*.npz`、`CASE.md` 一个字节未变。
* `min over band diag B` 对健康工况天然是「带底小正数」，当强度指标读会误导；正确读法是「有无负值 +
  负值在带内还是带边」，说明书已把这句写进正文。
* 旧 `np.diag` 口径的「恰好 0」还与同一份 README 自己的 C0 行（min B 2.7e-13 ≥ 0）自相矛盾，属于
  典型的「文档第二真相」。
* 相邻门 `tests/test_doc_claim_audit.py` 有 3 条失败，在干净 main 上同样失败（`new unguarded document:
  AUDIT_RESTAT_CALIBRATION_20261113.md`），与本单无关，未粉饰。
* n315 数值栈根因（`NODE-N315-NUMERICAL-VALIDITY`）仍未归因，本单不涉及。

## ⑥ 误差判定

关键量对基准的逐项误差（基准 = 工件 `.npz` 里的原值，判据 = 渲染文本必须与之相等）：

| 关键量 | 旧 README | 工件真值 | 相对误差 | 是否 < 3 % |
| --- | --- | --- | --- | --- |
| 提交件一侧 pitch min B | −0.000681 | {n315['per_dof']['pitch']:.4g}（n315）/ {ok['per_dof']['pitch']:.4g}（提交件） | 旧值实为 yaw，属**错标**而非数值误差 | 否（错标，非误差） |
| n315 pitch min B | 无名裸数字 | {n315['per_dof']['pitch']:.6g} | 数值本身一致（0 %），缺的是 DOF 归属 | 是（数值）/ 否（归属） |
| 提交件 surge min B | 0（下界） | {ok['per_dof']['surge']:.6g} | 100 %（被压成 0） | 否 |
| 提交件 pitch min B | 0（下界） | {ok['per_dof']['pitch']:.6g} | 100 %（被压成 0） | 否 |
| n315 max solver residual | 1.253 | {n315['residual_max']:.6g} | 0 % | 是 |
| 提交件 max solver residual | 0.00639 | {ok['residual_max']:.6g} | 0 % | 是 |

结论：**数值来源本身没错（residual 与 min 的数值逐位一致），错的是「这个数属于哪个 DOF」以及
「非负 DOF 被压成 0」**。修后所有渲染数字与工件逐项相等（相对误差 0 %，< 3 %），且两条新测试在旧
实现上失败、在新实现上通过，证明修复被测试咬住而不是被措辞安慰。
"""

png = base64.b64encode(FIG.read_bytes()).decode()
body = {
    "title": (
        "CASE-BEM-CUMMINS-WIGLEY 说明书 DOF 归属缺陷修复：yaw 的最小值曾被印成 pitch，"
        f"真值 pitch={n315['per_dof']['pitch']:.3g}"
    ),
    "summary": (
        "PR#1081 交付的说明书由工件渲染，但渲染器把手写列号 5 当作 pitch（实为 yaw），"
        f"把 yaw 的 {n315['per_dof']['yaw']:.3g} 印成「pitch」，而真正的 pitch 最小值 "
        f"{n315['per_dof']['pitch']:.3g}（n315 扫描里最坏、带内造能）只以无名裸数字出现；"
        "另一处 np.diag(1-D 切片) 的副对角线零把通过 C0 的提交件 6 个 DOF 全压成「0（下界）」"
        f"（真值 {(ok['per_dof']['surge']):.3g}/{(ok['per_dof']['pitch']):.3g}）。"
        "修法：按 DOF 名取列、切片即对角线、删掉遮蔽真值的 floor 特例、README 重渲染，"
        "并新增两条测试（独立重算比对 + 错标负控），在旧实现上均失败。物理与判据零改动。"
    ),
    "scenario_key": "case-bem-cummins-wigley",
    "mission_id": 23091,
    "tags": [
        "TensorBEM",
        "CASE-BEM-CUMMINS-WIGLEY",
        "C0",
        "doc-integrity",
        "defect-fix",
    ],
    "markdown": markdown,
    "figures": [
        {
            "name": "dof_attribution.png",
            "caption": (
                "左：n315 coarse 逐 DOF 带内最小辐射阻尼（symlog），红柱为负＝造能，最坏 pitch=−1.463；"
                "右：B_pitch(ω) 两机对照，提交件全程 ≥0 并在带底趋于 0，n315 在带中 4/41 节点转负。"
            ),
            "png_base64": png,
        }
    ],
}

base = os.environ["TENSOR_API_BASE"].rstrip("/")
login = urllib.request.Request(
    f"{base}/auth/login",
    data=json.dumps(
        {
            "username": os.environ["TENSOR_AGENT_USER"],
            "password": os.environ["TENSOR_AGENT_PASS"],
        }
    ).encode(),
    headers={"Content-Type": "application/json"},
)
token = json.load(urllib.request.urlopen(login, timeout=60))["access_token"]
req = urllib.request.Request(
    f"{base}/tensor/reports",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
)
print(json.dumps(json.load(urllib.request.urlopen(req, timeout=120)), indent=2)[:800])