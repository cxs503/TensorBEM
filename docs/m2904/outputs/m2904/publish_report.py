"""Build and publish the m2904 progress report (figures downscaled to fit the API)."""
from __future__ import annotations

import base64
import io
import json
import os
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8001/api/v1")
TOKEN = open("/tmp/m2904_token").read().strip()
PNG = "outputs/recovery_robustness/recovery_robustness.png"

image = Image.open(PNG).convert("RGB")
width = 1180
image = image.resize((width, int(image.height * width / image.width)), Image.LANCZOS)
buffer = io.BytesIO()
image.save(buffer, format="PNG", optimize=True)
png_bytes = buffer.getvalue()
print("figure bytes", len(png_bytes))

MARKDOWN = r"""# TensorBEM 表面速度恢复的鲁棒性：面板级数据误差、光滑化与可计算误差指示器（m2904）

## 1. 案例介绍

`docs/three_dimensional_bem.md` §7 局限 9 留了两条无证据的断言：(a) 恢复路径"依赖边界势的光滑性，
真实分片常值输入需先做光滑化"； (b) 恢复算子"二阶超收敛"。光滑化从未实现、从未量化，阶也从未单独测过。
本单把两条都变成可执行判据。

物理背景：半径 $a=1$ 球沿 $+x$ 平移的外域势流，物面势 $u=-x/(2r)$，精确梯度
$\nabla u=-\tfrac12(\hat e_x/r-\mathbf x x_1/r^3)$（与法向正交），逐面元在面元平面投影后与恢复结果比对。

| 参数 | 值 |
|---|---|
| 几何 | 测地球 level 1/2/3/4 → 80/320/1280/5120 面元，$h_{mean}$=0.5823/0.2993/0.1507/0.0755 |
| 数据误差模型 | 白噪声 $\sigma=10^{-2}$；白噪声 $\sigma=h_{mean}$；相关误差 $\epsilon\cos(3x)(1+y/2)$，$\epsilon=10^{-2}$ |
| 光滑化 | 面积加权 面元→顶点→面元，迭代 1/2/3 |
| 随机种子 | `default_rng(20261103)` |

## 2. 怎么做的

1. 隔离恢复算子：只喂精确迹，测算子自身阶（四网格、两参考投影、两场：平移势与度 2 调和 $x^2-z^2$、外加椭球几何）。
2. 注入受控数据误差，量 RMS 误差与无量纲放大 $A=\mathrm{err}\cdot h/\sigma$；白噪声两次（固定 $\sigma$ 与 $\sigma\propto h$）、相关误差一次。
3. 实现文档要的光滑化（`tensorbem.panel_smoothing`），量化收益 gain 与对精确数据的代价 penalty。
4. 构造**只依赖计算结果**的误差指示器（`tensorbem.recovery_uncertainty`）：复用恢复的同一设计矩阵（重构抽出 `patch_fit_system`），
   片内残差估 $\hat\sigma_f=\sqrt{\mathrm{RSS}_f/(m_f-4)}$，经 $(X^TX)^{-1}$ 斜率块投影得预测 $\hat e_f$。
5. 工具链：`python -m tensorbem.benchmarks.recovery_robustness --levels 1 2 3 4`（8.5 s，纯 numpy）；工件 JSON/MD/PNG；`tests/test_recovery_robustness.py`（22 条，全绿）。

## 3. 计算结果

### 表 1 恢复算子对精确迹的阶（解析基准）

| level | 面元 | $h$ | RMS 绝对误差 | 相对 L2 | 观测阶 |
|---:|---:|---:|---:|---:|---:|
| 1 | 80 | 0.5823 | 8.522e-3 | 1.966e-2 | — |
| 2 | 320 | 0.2993 | 4.128e-3 | 9.959e-3 | 1.046 |
| 3 | 1280 | 0.1507 | 2.082e-3 | 5.080e-3 | 0.987 |
| 4 | 5120 | 0.0755 | 1.034e-3 | 2.531e-3 | 1.009 |

**一阶，不是二阶**。椭球网格 3.18e-1→1.31e-1→6.19e-2（比 2.43、2.11）复现同阶；换参考投影结论不变。已改正 `recovery.py` 的“二阶超收敛”表述并由测试守住。

### 表 2 白噪声 $\sigma=10^{-2}$：放大律与光滑化收益

| level | 面元 | $h$ | 原始误差 | 1 次光滑 | gain | $A=\mathrm{err}\cdot h/\sigma$ | 指示器预测 | 实测/预测 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 80 | 0.5823 | 1.505e-2 | 4.986e-2 | 0.30 | 0.877 | 1.262e-2 | 1.19 |
| 2 | 320 | 0.2993 | 1.957e-2 | 1.803e-2 | 1.09 | 0.586 | 1.984e-2 | 0.99 |
| 3 | 1280 | 0.1507 | 4.174e-2 | 2.141e-2 | 1.95 | 0.629 | 4.226e-2 | 0.99 |
| 4 | 5120 | 0.0755 | 8.048e-2 | 3.849e-2 | 2.09 | 0.608 | 8.216e-2 | 0.98 |

$\log$误差–$\log h$ 斜率 −0.850；$N\ge320$ 时放大常数 $A=0.61\pm0.03$。$\sigma\propto h$ 一列误差恒在 0.56–0.71（**不收敛**）→ 光滑化只给常数倍收益，**必要但不充分**。

### 表 3 光滑化对精确数据的代价 / 相关误差的良性

| level | 精确数据 1 次光滑 penalty | 相关误差（同幅度）原始 | 白/相关 比 |
|---:|---:|---:|---:|
| 1 | 5.87 | 1.039e-2 | 1.4 |
| 2 | 3.44 | 4.731e-3 | 4.1 |
| 3 | 4.95 | 2.270e-3 | 18.4 |
| 4 | 7.09 | 1.110e-3 | 72.5 |

放大是**非光滑数据误差**的性质，不是算子缺陷：缓变的 BEM 迹误差可直接用恢复路径。

## 4. 结果图

左：RMS 误差–$h$ 双对数（精确迹 / 精确迹+1 次光滑 / $\sigma=10^{-2}$ 原始与平滑 / 相关误差 / 斜率 −1 参考线）；
右：指示器预测 vs 实测散点 + 1:1 与 ×3 界。产品级原图见 `outputs/recovery_robustness/recovery_robustness.png`（160 dpi），此处为压缩版。

## 5. 注意点（坑 / 局限）

1. **参考量选择是第一个坑**：面元法向与径向差为 $O(h)$（8.6e-3/4.2e-3/2.0e-3/1.0e-3）；本文同时用面元平面投影与径向切平面投影，结论一致。
2. **一阶不是 bug**：换场、换参考、换网格都复现；文档原本的 1.67–1.78 是混合量（恢复 + 通量 + $O(h^2)$ 几何误差）的表观阶，从未单独测算子。
3. **指示器只在未平滑数据上可信**：平滑后系统性低估 3.4–4.1 倍（最粗网格 27 倍），因平滑使残差相关化；用平滑算子传播协方差未实现。
4. **噪声模型是受控注入，不是实测 BEM 迹误差谱**：仓库内无可采样的外域辐射算例，故真实迹误差谱未测（未完成项）。
5. 只覆盖球/椭球、Laplace 外域、度 1–2 调和场；自由液面与船体曲面未覆盖。
6. 光滑化 2–3 次在细网格上收益饱和而代价单调上升，默认建议 1 次。

## 6. 误差判定（逐项回答“是否 <3%”）

| 判据 | 基准 | 实测 | 是否 <3% |
|---|---|---|---|
| 恢复算子阶 | 解析梯度 | 1.046/0.987/1.009 区间阶 | **一阶**（非文档声明的二阶） |
| 恢复误差（相对 L2） | 解析梯度 | 1.97e-2 / 9.96e-3 / 5.08e-3 / 2.53e-3 | **是**（80 面元 1.97%） |
| 指示器（原始白噪声） | 实测 RMS | 19%（最粗）; $N\ge320$ 时 1.4%/1.2%/2.0% | **是（$N\ge320$）**；最粗 19% **否** |
| 指示器（光滑后） | 实测 RMS | 低估 3.4–4.1 倍（240%–310%） | **否**（如实记录） |
| 放大律 | $A=\mathrm{const}$ | $A=0.61\pm0.03$（$N\ge320$），斜率 −0.850 | **是**（偏差 <5%） |
| 光滑化收益 | gain | 1.95/2.09（1280/5120） | **是** |
| 光滑化代价 | 精确数据放大 | 3.44–7.09 倍 | **否**（故必须按需使用） |

**结论**：光滑化必要但不充分；真正的信任门是新增的 `tensorbem.recovery_uncertainty`（未平滑数据上误差 <3%），
船体力积分前应读该指示器，而不是无条件平滑。

## 7. 复现

```bash
PYTHONPATH=src python3 -m tensorbem.benchmarks.recovery_robustness --levels 1 2 3 4
PYTHONPATH=src python3 -m pytest tests/test_recovery_robustness.py -q
```
"""

payload = {
    "title": "TensorBEM 表面速度恢复结清验证欠账：算子实测一阶（非文档二阶）、σ/h 放大律 A=0.61、光滑化必要但不充分",
    "summary": (
        "结清 docs/three_dimensional_bem.md §7 局限 9：实现面元数据面积加权光滑化并量化其对白噪声/"
        "相关误差的收益与代价，重构抽出 patch_fit_system 后构造只依赖计算结果的误差指示器；"
        "测得恢复算子为一阶（1.046/0.987/1.009）、白噪声放大 A=err·h/σ=0.61±0.03、"
        "光滑化只给常数倍收益（1.95–2.09）且对精确数据有 3.4–7.1 倍代价；指示器在未平滑数据上误差 <3%（N>=320）。"
    ),
    "scenario_key": "tensorbem_recovery_robustness",
    "mission_id": 2904,
    "tags": ["TensorBEM", "验证欠账", "表面速度", "恢复算子", "误差指示器", "光滑化"],
    "markdown": MARKDOWN,
    "figures": [
        {
            "name": "recovery_robustness.png",
            "caption": "左：恢复梯度 RMS 误差随面元尺度 h 的双对数（精确迹、精确迹+1 次光滑、σ=1e-2 白噪声原始/平滑、相关误差、斜率 −1 参考线）；右：残差指示器预测 vs 实测（1:1 与 ×3 界）。",
            "png_base64": base64.b64encode(png_bytes).decode(),
        }
    ],
}

body = json.dumps(payload).encode()
print("payload bytes", len(body))
request = urllib.request.Request(
    BASE + "/tensor/reports",
    data=body,
    headers={"Content-Type": "application/json", "Authorization": "Bearer " + TOKEN},
    method="POST",
)
with urllib.request.urlopen(request, timeout=120) as response:
    print(response.status, response.read().decode()[:400])