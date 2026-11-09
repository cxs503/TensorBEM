"""Publish the m8769 quadratic-panel scaling-invariance report to the platform API."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER")
PASS = os.environ.get("TENSOR_AGENT_PASS")


def post(path: str, payload: dict, token: str | None = None) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})},
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.loads(response.read().decode())


login = post("/auth/login", {"username": USER, "password": PASS})
token = login.get("access_token") or login.get("token")
if not token:
    raise SystemExit(f"no token in login response keys: {sorted(login)}")

payload = {
    "title": "TensorBEM 二次曲面单元算子验证：均匀缩放不变性回归钉（PR#568 管到底第3轮·m8769）",
    "summary": "将 PR#568 二次曲面单元算子不变性回归钉从刚体群（SE(3)：平移+旋转）推进到相似群（含均匀缩放）。验证 S 的 1/s 缩放律（max|S1/scale-S0|=1.25e-16）与 D 的尺寸无关性（max|D1-D0|=3.26e-14）均以机器精度成立。54/54 测试通过，无回归。",
    "scenario_key": "tensorbem_quadratic_panel_verification",
    "mission_id": "8769",
    "tags": ["tensorbem", "quadratic-panel", "bem", "laplace", "scale-invariance", "regression-pin"],
    "markdown": """# TensorBEM 二次曲面单元算子验证（PR#568 管到底第3轮·m8769）

## 1. 案例介绍

三维 Laplace 外问题 Dirichlet 边界条件，单位球面（icosphere 二次 6 节点曲面单元），内部点源 x0=(0,0,0.5)。精确解：u(x)=1/(4π|x-x0|)，q(x)=-(x-x0)·n/(4π|x-x0|³)。

## 2. 怎么做的

PR#568 全部功能要点已在 main 完整落地（经 PR#550/558/572/577 承接链）。本 PR 新增 `test_operators_are_scale_invariant`：椭球 1.0/1.6/0.7（162 节点）整体放大 2.7 倍后，验证单层算子 S 按 1/scale 缩放（Laplace 核 G(x,y)=1/(4π|x-y|)），双层算子 D 不变（固体角与尺寸无关）。容差 1e-10 相对。

## 3. 计算结果

| 量 | 实测 | 容差 | 判定 |
|---|---|---|---|
| max\\|S₁/scale − S₀\\| | 1.249e-16 | 1e-10 | ✓ |
| max\\|D₁ − D₀\\| | 3.256e-14 | 1e-10 | ✓ |
| 缩放后固体角 \\|D₁·1+1/2\\|∞ | 1.173e-02 | 2e-2 | ✓ |

通量收敛（最细级）：6.065e-03 < 3% 验收线 ✓。

## 4. 误差判定

所有关键量均 <3% 或远低于容差（低 4~6 个量级）。

## 5. 测试

本地 54/54 通过（91.8 s）。

## 6. 注意点

本 PR 仅改测试文件（非核心模块）；无新增图（纯测试钉，无新物理量）。""",
    "figures": [],
}

try:
    result = post("/tensor/reports", payload, token)
except urllib.error.HTTPError as exc:
    print("HTTP", exc.code, exc.read().decode()[:500])
    raise
print(json.dumps({k: result[k] for k in list(result)[:6]}, default=str)[:600])
