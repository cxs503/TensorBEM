"""Post the m15669 progress report to the Tensor platform (six-element format)."""
from __future__ import annotations

import base64
import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def _post(url: str, payload: dict, token: str | None = None) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


_login = json.dumps({"username": USER, "password": PASS}).encode()
token = json.loads(urllib.request.urlopen(urllib.request.Request(
    f"{BASE}/auth/login", data=_login,
    headers={"Content-Type": "application/json"})).read())["access_token"]

png = base64.b64encode(
    open("docs/fmm_acceleration/fig_m15669_operator.png", "rb").read()).decode()

md = """# TensorBEM m15669 — `tensorbem.fmm`: 八叉树几何缺陷修复 + 向量化 matvec

## 1 案例介绍（物理背景/几何/参数）

BEM 单层算子 `S q = Σ_j q_j/(4π|x_i−y_j|)` 的稠密 matvec 是 O(N²)，FMM 用八叉树 +
实球谐多极展开把远场压成「每叶一次交互」。本单重开 owner 关闭的原 PR#665（m14010，
FMM 加速器），不重抄旧产物，而是修掉 m14010/m14931 遗留的**算子缺陷**并给出可测证据。

| 项 | 值 |
|---|---|
| 几何 | `geometry3d.icosphere` subdivisions=2/3/4/5，面板 N=320/1280/5120/20480 |
| 源/电荷 | 面元质心 / 面元面积 |
| 靶点 | 质心（测代价）；网格顶点（测精度，无重合对） |
| 展开阶 p | 2、4（代价）；1..8（阶律） |
| 叶容量 | 16/32/128 | 
| 工具链 | Python 3.11 + numpy（纯 Python 算子，loong64 单节点） |

## 2 怎么做的

**缺陷 A（根因）**：`_build_octree_recursive` 子盒边长 `size/2` 却把子心放在
`center±size/2`（应为 `±size/4`），且子盒枚举序 `ix,iy,iz` 与 `_child_index` 的位编码
（bit0=x,bit1=y,bit2=z）不一致 → 子 1/子 4 互换。实测 N=1280、max_leaf=32：**56 个叶中
48 个叶的点不在自己盒内**，叶源半径中位数 3.19、最大 3.82（域半径 1.0）。因 m14931 的
可容许判据用叶**自身**源半径，巨大半径把全部交互打成近场：多极对占比仅 **19.6%**，
即该「FMM」实为稠密求和的记账包装。

**缺陷 B**：`matvec` 逐靶点逐叶 Python 循环、每次远场交互单点求球谐。改为分块向量化：
(block, n_leaves) 近场掩码（判据逐字不变）、近场 (靶,源) 对扁平拼接 + `bincount`、
远场批量球谐、`max_pairs=65536` 预算按靶递归二分（峰值内存与 N 无关）。
`setup` 未给电荷时 matvec 惰性构造多极系数。

**审计扩展**：`separated_ladder` 的 max_leaf 50→25（树修好后 50 会让梯子全部走近场而
空转）；新增 `scaling_ladder`，把战役判据 (c)「N∈[1e3,1e5] 斜率∈[0.9,1.2]」从外推变成实测。

## 3 计算结果（程序化注入，`docs/fmm_cost_audit/audit.json`）

| 面板 N | p | 叶数 | 多极对占比 | t_FMM (s) | t_dense (s) | FMM/dense | 修复前 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 320 | 2 | 56 | 0.741 | 0.055 | 0.006 | 9.32x | 328x |
| 1280 | 2 | 56 | 0.741 | 0.134 | 0.112 | **1.20x** | 98x |
| 320 | 4 | 56 | 0.741 | 0.151 | 0.007 | 22.18x | 1147x |
| 5120 | 2 | 56 | 0.740 | 1.199 | 1.881 | **0.64x** | 未测 |
| 20480 | 2 | 272 | 0.936 | 9.310 | 23.343 | **0.40x** | 未测 |

实测 log-log 斜率（N=1280→20480）：**FMM 1.48 / 稠密 1.82**，比值单调下降，
cross-over 在 N≈2000–5000。

| 精度配置 | p | 修复前 | 修复后 |
|---|---:|---:|---:|
| 曲面 320 面元（质心→顶点） | 2 | 1.130e-03 | 2.142e-04 |
| 曲面 320 面元（质心→顶点） | 4 | 1.597e-04 | 7.720e-05 |
| 分离构型 r/a=8 | 1 | 2.453e-02 | 3.959e-04 |
| 分离构型 r/a=8 | 2 | 2.893e-03 | 4.191e-05 |
| 分离构型 r/a=8 | 8 | 1.036e-06 | 4.181e-12 |

## 4 结果图

左：代价-规模双对数（FMM vs 向量化稠密，实测点 + 拟合斜率，标注各点 FMM/dense 比值）；
右：精度-阶曲线（分离构型与曲面构型，含 15%/3% 判据线）。

## 5 注意点

- 判据 (c) **未按原措辞达到**：实测斜率 1.48 > 1.2 上界。单层 FMM 在面分布上仍是
  O(N^1.5) 的 (靶, 叶) 对；「O(N) 斜率」不成立，但亚二次且 N≥5120 实测加速。
- 判据 (d)（<1e-10「物理不变」）**原配置空转**：`max_leaf=panel_count` ⇒ 1 叶 ⇒
  直接 vs 直接；已改为曲面多极精度 7.72e-05。
- 判据 (a) 的 1e-12 绝对锚在自项 `r` 夹到 1e-15 时低于一个 ulp（元素量级 1e14），
  已改为相对锚。
- 仍无 M2M/M2L/L2L 层级传递；Helmholtz ka>10 仍 fail-closed；算子仍未接入 solver3d。
- 叶心最近邻路由为近似路由，未量化边界影响。

## 6 误差判定（对基准 = 稠密直接法）

| 关键量 | 值 | 基准 | 相对误差 | 是否 <3% |
|---|---:|---|---:|---|
| 曲面 matvec 最大相对误差 p=2 | 2.142e-04 | 稠密求和 | 0.021% | 是 |
| 曲面 matvec 最大相对误差 p=4 | 7.720e-05 | 稠密求和 | 0.0077% | 是 |
| 分离构型相对误差 p=2 | 4.191e-05 | 稠密求和 | 0.0042% | 是 |
| 分离构型相对误差 p=8 | 4.181e-12 | 稠密求和 | 4e-10% | 是 |
| 向量化 vs 标量参考（语义等价） | <1e-12 | 原算法规格 | — | 是 |
| 单叶退化分支 | 0（相对 ≤1e-12·量级） | 稠密求和 | 0% | 是 |
| 代价 N=20480 FMM/dense | 0.399 | 稠密墙钟 | 加速 2.5x | 是（代价项） |

代价口径：best-of-2 墙钟，同 N 同靶集；噪声量级 ~10%，结论（cross-over 与量级）不受影响。
"""

payload = {
    "title": "TensorBEM m15669 · fmm 八叉树几何缺陷修复 + 向量化 matvec（原 PR#665 重开）",
    "summary": "修复 fmm 八叉树子盒不铺满父盒/子序与位编码错位的缺陷（叶点自含 48/56→56/56，"
               "多极对占比 19.6%→74%），matvec 分块向量化；代价 N=1280 由 98x 降到 1.20x、"
               "N=20480 实测 0.40x（加速），曲面精度 p=4 1.60e-4→7.72e-5。",
    "scenario_key": "tensorbem_fmm_octree_fix",
    "mission_id": 15669,
    "tags": ["TensorBEM", "FMM", "octree", "cost-audit", "defect-fix", "m15669"],
    "markdown": md,
    "figures": [{"name": "fig_m15669_operator.png",
                 "caption": "左：FMM vs 向量化稠密 matvec 墙钟（实测点，标注 FMM/dense 比值）；"
                            "右：分离/曲面构型精度-阶曲线（含 15%/3% 判据线）",
                 "png_base64": png}],
}
resp = _post(f"{BASE}/tensor/reports", payload, token)
print(json.dumps(resp)[:400])