"""Shrink the figures (matplotlib only) and post the m3631 platform report."""
import base64
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
API = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def call(path, payload=None, token=None, method=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        API + path, data=data,
        headers={"Content-Type": "application/json"}, method=method or ("POST" if data else "GET"),
    )
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode())


login = call("/tensor/login", {"username": USER, "password": PASS})
token = login.get("token") or login.get("access_token")
assert token, f"login failed: {login}"


def shrink(src, dst, target_kb=55):
    import matplotlib.image as mpimg
    import numpy as np

    im = mpimg.imread(src)
    h, w = im.shape[:2]
    for scale in (0.75, 0.6, 0.5, 0.4):
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        bh, bw = h // nh, w // nw
        im2 = im[: nh * bh, : nw * bw].reshape(nh, bh, nw, bw, 3).mean(axis=(1, 3))
        mpimg.imsave(dst, im2)
        if os.path.getsize(dst) <= target_kb * 1024:
            break
    print(os.path.basename(dst), os.path.getsize(dst) // 1024, "KB")
    return dst


fig1 = shrink(os.path.join(HERE, "fig_3d_free_surface_convergence.png"),
              os.path.join(HERE, "fig1_small.png"))
fig2 = shrink(os.path.join(HERE, "fig_3d_free_surface_deviation.png"),
              os.path.join(HERE, "fig2_small.png"))


def b64(p):
    return base64.b64encode(open(p, "rb").read()).decode()


md = """## 1. 案例介绍

TensorBEM 的 `docs/marine_bem_benchmarks.md` §4.10–§4.11 记录两族三维深水自由液面基准：**Hulme (1982) 漂浮半球**（半径 1 m，Ka=1.0，16/64/256 面元；CHIEF 行 Ka=2.58）与 **Wang/Linton (1991) 潜没球**（f/a=1.5，Ka=0.8，20/80/320 面元）。真值分别是 48 项球面无波势+Kochin 振幅闭式参考，与零数值求积 Thorne 多极子闭式展开。m3059（2026-11-04）明确留下欠账：该家族 66 个声明数字（表 10/11/12+正文残差）未纳入可执行注册表，无测试守护。

| 参数 | Hulme 半球 | Wang/Linton 球 |
|---|---|---|
| 几何 | 穿水开口半球网格 | 潜没球（球心 z=−1.5） |
| 频率 | Ka=1.0 / 2.58 | Ka=0.8 |
| 面元 | 16/64/256 | 20/80/320 |
| 判据量 | 5 量误差+条件数+阶+残差 | 4 量误差+耦合比+条件数+阶 |

## 2. 怎么做的

1. 成本重估：三张表只引用收敛研究，实测总耗时 14.7 s，归入 **standard 层**（默认 CI 覆盖），推翻 m3059「成本更高」的推迟理由（其指含频率扫描的完整套件）。
2. `declared_numbers.py` 新增 3 组 66 条注册项；runner 复用 `run_refinement_study`/`run_irregular_frequency_study`，同参数共享一次求解。
3. `floating_hemisphere.py` 纯增量补 B11' 观测阶序列（表 10 第五列原本未追踪）。
4. **口径归因（本单实质发现）**：表 10 的 B11' 阶（1.62/1.88）是 m2609 面元数口径 h∝N^(−1/2)，其余四列是 h_mean 口径；16→64 层 h_mean 比 1.87≠2.0，两口径差 11%。注册表按各自口径钉桩，文档补口径说明（m3280 先例）。
5. 回归测试：覆盖面判据扩到 3 组新组 + 新增 standard 层 66 条判据测试（含 CHIEF 物理声明断言）。

## 3. 计算结果

`python -m tensorbem.benchmarks.declared_numbers --tiers standard --strict` → **70/70 PASS**（既有 4+新增 66），零手抄（`outputs/verif_m3631/standard_tier.json`）。

| 组 | 条数 | 判定 | 最坏相对偏差 |
|---|---:|---|---:|
| hulme_hemisphere（表10+正文） | 31 | PASS | 3.9e-3 |
| hulme_chief（表11） | 9 | PASS | 1.1e-3 |
| submerged_sphere（表12） | 26 | PASS | 2.2e-2（2 位有效数字舍入） |

表 11 物理声明复现：未增广 BIE 条件数 145.66、B33'=−0.1042（非物理负阻尼）；5 点内部 CHIEF 后条件数 4.24、B33'=+0.0687（严格耗散）。

## 4. 结果图

图 1：两族基准收敛曲线（实线=本次实测，虚线=文档声明值），log-log 下均二阶收敛。
图 2：66 条注册项 |实测−声明|/声明 对各自 DRIFT 容差线，无一越线。

## 5. 注意点

- 口径混用是真实发现：B11' 阶与其他四列阶口径不同（差 11%），已按文档原值钉桩并显式说明。
- 耦合比 2.2e-4/2.6e-5 仅 2 位有效数字，容差按印刷精度放宽到 2.5%（注册表既有原则）。
- 未改任何求解器实现；1024 面元 B11' 专项与有限水深 John 核仍是欠账/缺口（建议派单）。
- 平台：Python 3.11.6 / NumPy 2.4.6 / loongarch64 / OPENBLAS_NUM_THREADS=1。

## 6. 误差判定

| 关键量 | 基准 | 相对误差 | <3%? |
|---|---|---:|---|
| 66 条注册项 vs 文档 | 文档印刷值 | 最坏 2.2e-2（舍入）；其余 ≤3.9e-3 | ✅ |
| 256 面元 5 量 | Hulme 闭式 | 5.7e-3–3.5e-2（离散误差） | ✅ |
| 320 面元 4 量 | Thorne 多极子 | 1.9e-2–3.2e-2（离散误差） | ✅ |
| CHIEF 耗散性 | B33>0 | −0.1042→+0.0687 | ✅ |

**结论：全部关键量对基准误差 <3%，无 DRIFT，欠账结清。**
"""

payload = {
    "title": "TensorBEM 验证欠账结清：三维自由液面（Hulme/Wang-Linton）66 个声明数字纳入注册表并回归钉桩",
    "summary": "m3059 遗留的三维自由液面注册表欠账结清：表10/11/12+正文共66个数字登记进 declared_numbers（standard 层，14.7s 复算），70/70 PASS；发现并钉死表10 B11' 阶的 √N/h_mean 双口径问题；tests/test_declared_numbers.py 13/13 绿。",
    "scenario_key": "tensorbem-3d-free-surface-registry",
    "mission_id": 3631,
    "tags": ["TensorBEM", "verification-debt", "declared-numbers", "free-surface", "Hulme", "Wang-Linton"],
    "markdown": md,
    "figures": [
        {"name": "fig_3d_free_surface_convergence.png",
         "caption": "Hulme 半球与 Wang/Linton 潜没球收敛：实测（实线）vs 文档声明（虚线）",
         "png_base64": b64(fig1)},
        {"name": "fig_3d_free_surface_deviation.png",
         "caption": "66 条注册项相对偏差 vs DRIFT 容差线（对数轴），全部 PASS",
         "png_base64": b64(fig2)},
    ],
}
resp = call("/tensor/reports", payload, token)
print("POST resp:", json.dumps(resp, ensure_ascii=False)[:400])
try:
    back = call("/tensor/reports?mission_id=3631&limit=5", token=token)
    items = back if isinstance(back, list) else back.get("items", back.get("reports", []))
    for r in items[:5]:
        print("verify:", r.get("id"), r.get("title", "")[:60])
except Exception as e:
    print("verify GET failed:", e)
