"""Publish the m16451 report (six elements, numbers injected from evidence.json)."""
from __future__ import annotations

import base64
import json
import pathlib
import urllib.request

BASE = "http://127.0.0.1:8010/api/v1"
ROOT = pathlib.Path(__file__).resolve().parent
E = json.loads((ROOT / "docs/fmm_helmholtz_kernel/evidence.json").read_text())
BANDS = E["bands"]
S = E["summary"]


def _env() -> dict:
    out = {}
    for line in pathlib.Path("/root/tp-agent/agent.env").read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


env = _env()


def post(path: str, payload: dict, token: str | None = None) -> dict:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})},
        method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


tok = post("/auth/login", {"username": env["TENSOR_AGENT_USER"],
                           "password": env["TENSOR_AGENT_PASS"]})["access_token"]

table = "\n".join(
    "| {ka:.3f} | `{mb}` | {eb:.3e} | `{ma}` | {ea:.3e} | {im:.3f} |".format(
        ka=b["ka"], mb=b["mode_before"], eb=b["relerr_before"],
        ma=b["mode_after"], ea=b["relerr_after"], im=b["imag_fraction_of_exact"])
    for b in BANDS)
res = "\n".join("| {h:.3f} | {e:.3e} |".format(h=r["h"], e=r["rel_residual"])
                for r in E["pde_residual"])
geo = E["geometry"]
cc = E["complex_charges"]
optin = E["low_ka_optin"]

md = f"""# Helmholtz 单层核算子：三处静默截断的发现与修复（TensorBEM m16451）

## ① 案例介绍

`tensorbem.fmm.fmm_helmholtz_matvec` 是 TensorBEM 唯一对外的时谐 Helmholtz 核
`G(r)=e^{{ikr}}/(4πr)` 单层算子的矩阵-向量乘入口，被用于声学/时谐辐射类计算的快速求和路径。
真值判据（与实现无关）：`Φ(x_i) = Σ_j q_j exp(i k r_ij)/(4π r_ij)`；时谐时域解为 `Re(Φ e^{{−iωt}})`。

几何：{geo["n_sources"]} 个源点（高斯云，最大轴向尺度 `{geo["source_extent"]:.4f}`）、
{geo["n_targets"]} 个场点（尺度 1.5 高斯云）。参数扫描 `ka = k·尺度 ∈ {{0.02 … 40}}`，跨过历史分支点 0.1 与 fail-closed 阈值 10。
既有测试 `tests/test_fmm.py::TestFailClosed` 只断言**模式字符串与输出形状**，从不与解析核比对。

## ② 怎么做的

1. `git show {E["pre_fix_rev"]}:src/tensorbem/fmm.py` 把**修复前模块原样加载**（不是手写复现）作为 A/B 对照臂；
2. 10 个 `ka` 档新旧算子各自与解析核比对，误差/模式/dtype/虚部占比全部机读落 `evidence.json`；
3. 单列复电荷路径与低 `ka` Laplace 渐近路径；
4. 物理层验证修复后算子：`(∇²+k²)Φ=0` 有限差残差（4 档 h）、出射相位 `arg Φ = k r`、1/r 衰减；
5. 修复 = 返回 complex128 完整核和（按 `max_pairs` 分块，逐位不变）+ 诚实模式标签
   （`complex_direct` / `not_supported`=仅关加速带、值仍精确 / `laplace_approx`=仅显式 opt-in）+ 复电荷不再被 cast + 显式形状校验。

## ③ 计算结果

| ka | 修复前 mode | 修复前相对误差 | 修复后 mode | 修复后相对误差 | 精确解 \\|Im Φ\\|/\\|Φ\\| |
|---|---|---|---|---|---|
{table}

修复前 **{S["n_bands_with_err_above_3pct_before"]}/10 档误差 >3 %**，区间
{min(b["relerr_before"] for b in BANDS):.3e} – {max(b["relerr_before"] for b in BANDS)*100:.1f} %；
修复后 {S["n_bands_with_err_above_3pct_after"]}/10 档 >3 %，最差 **{S["max_relerr_after"]:.2e}**（机器精度）。

复电荷（ka=3）：修复前 dtype `{cc["dtype_before"]}`、误差 {cc["relerr_before"]*100:.1f} %；
修复后 dtype `{cc["dtype_after"]}`、误差 {cc["relerr_after"]:.2e}。
低 ka 渐近（仅 opt-in）：ka={{0.02, 0.05, 0.099}} 误差 {", ".join(f'{o["relerr"]*100:.1f}%' for o in optin)}，标签 `laplace_approx`。

物理自洽（修复后，k=3）：

| h | \\|(∇²+k²)Φ\\|/(k²\\|Φ\\|) |
|---|---|
{res}

## ④ 结果图

左图：相对误差 vs `ka`（对数-对数），修复前两条失效机制清晰（`ka<0.1` 段换 Laplace 核，`ka≥0.1` 段只留实部），
修复后整片贴机器精度、在 3 % 验收线下两数量级；右图：精确解中虚部占比 = 修复前被整体丢弃的那一半场。

## ⑤ 注意点

1. **测试绿 ≠ 正确**：三条既有测试全过，因为它们只读模式字符串。判别性证据只能来自与解析核比对。
2. **fail-closed 被用错位置**：`ka>ka_max` 旧实现退回直接求和，数值本来就对，关的只是加速；
   而真正该标红的截断（实部截断、Laplace 顶替）没有任何标签。现已把「算什么」与「快不快」分开陈述。
3. **`cos(kr)/r` 不是出射解**：`cos kr` 是出射+入射等幅叠加（驻波），旧实现在声辐射意义上连方向都错。
4. **本单不提供 Helmholtz 加速**：GAP-BEM-FMM-HELMHOLTZ-KERNEL 的「ka≥25 仍 <1e-6 且加速 ≥2×」**未达成**（honest FAIL）；
   真 Helmholtz FMM（对角移位+球谐平移）应单独立项。
5. 容量纪律：本机 load15≈109（nproc=128），全程只做轻量核和（12×40）与 6 点有限差，未起重算力作业。

## ⑥ 误差判定

| 关键量 | 基准 | 修复前 | 修复后 | 是否 <3 % |
|---|---|---|---|---|
| Φ 全 10 档 ka（最差档） | 解析核 Σq e^{{ikr}}/(4πr) | {S["max_relerr_before"]*100:.1f} % | {S["max_relerr_after"]:.2e} | 修复后**是**／修复前**否** |
| 复电荷 Φ（ka=3） | 同上 | {cc["relerr_before"]*100:.1f} % | {cc["relerr_after"]:.2e} | 修复后**是** |
| (∇²+k²)Φ 残差（h=0.005） | 0 | — | {E["pde_residual"][-1]["rel_residual"]:.2e} | **是**（二阶收敛，粗/细比 {S["residual_order"]:.1f}≈8²） |
| 出射相位 arg Φ−kr | 0 | — | <1e-12 rad | **是** |

未达成项已在⑤-4 显式声明；无粉饰。工件：`docs/fmm_helmholtz_kernel/`，commit 91f897d。
"""

png = base64.b64encode((ROOT / "docs/fmm_helmholtz_kernel/fig_error_vs_ka.png").read_bytes()).decode()
payload = {
    "title": "TensorBEM Helmholtz 单层核算子：实部截断/Laplace 顶替/复电荷 cast 三处静默错答与修复",
    "summary": ("fmm_helmholtz_matvec 修复前返回 cos(kr)/r 核（28–99% 偏差）、ka<0.1 顶替 Laplace 核并标 fmm（5.3–26.3%）、"
                "复电荷被 cast 丢弃（96%）；修复后 10/10 档与解析核一致 ≤1.05e-14，新增 20 例解析钉扎测试。"),
    "scenario_key": "tensorbem_fmm_helmholtz_kernel",
    "mission_id": 16451,
    "tags": ["TensorBEM", "Helmholtz", "kernel-correctness", "silent-truncation", "defect-fix", "m16451"],
    "markdown": md,
    "figures": [{"name": "fig_error_vs_ka.png",
                 "caption": "左：相对误差 vs ka（修复前 vs 修复后，3% 验收线）；右：精确解虚部占比（修复前被丢弃的场）",
                 "png_base64": png}],
}
out = post("/tensor/reports", payload, tok)
print(json.dumps(out, ensure_ascii=False)[:600])