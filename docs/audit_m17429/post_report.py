#!/usr/bin/env python3
"""Publish the m17429 board-audit report to the platform (tensor_reports)."""
import base64
import json
import os
import pathlib
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
OUT = pathlib.Path(__file__).resolve().parent


def api(method, path, payload=None, token=None):
    req = urllib.request.Request(BASE + path, method=method,
                                 data=json.dumps(payload).encode() if payload is not None else None)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, json.loads(r.read() or b"{}")


def fig(name, caption):
    b = base64.b64encode((OUT / name).read_bytes()).decode()
    return {"name": name, "caption": caption, "png_base64": b}


MARKDOWN = r"""# 交差审计：m17373 CASE-BEM-ICE-DISPERSION（TensorBEM）— 判级 L3

审计单 m17429（board-audit）· 对象 m17373（原单 m17274 返修轮1）· 被审 commit `26cbde4`
（分支 `agent/m17373`）· 基准判据 `tensor_benchmark_cases` WHERE `anchor_slug='CASE-BEM-ICE-DISPERSION'`（1 行，容差 <5%）

## 1. 案例介绍（物理背景 / 几何 / 参数）

冰覆盖水域的**质量加载**色散（Weitz–Keller 1950 模型，无弯曲刚度）：1 m 冰盖视为面质量，
覆盖在深度 H 的理想流体上，刚性底，平面周期波 `φ=cos(kx)cosh(k(z+H))/cosh(kH)`。
自由面 Robin 条件给出 `ω² = g k tanh(kH) / (1 + μ k tanh(kH))`，`μ = ρ_i h_i/ρ_w`。

| 量 | 值 |
|---|---|
| 冰厚 h_i / 冰密度 ρ_i / 水密度 ρ_w | 1 m / 917 / 1025 kg·m⁻³ |
| g | 9.80665 m/s² |
| μ | 0.894634 m |
| 截止频率 √(g/μ) / 周期 | 3.31084 rad/s / 1.89776 s |
| 深水系列 | H = 200 m，kH = 4 … 80 |
| 浅水系列 | H = 2 m，kH = 0.04 … 0.4 |
| 面元 | 64 panels/λ；容差 5% |

被审报告 ref 出处：Fox & Squire (1994) Phil. Trans. R. Soc. A 352 式(3)（Weitz & Keller 1950 关系）；
**原表原页 WITHHELD**（离线环境不可达，被审方已自曝）。

## 2. 怎么做的（审计方法 / 被审方方法）

**被审方**：单个周期单元 `0≤x≤2π/k, −H≤z≤0` 四边常值面元，解混合边值 Laplace BIE（自由项 1/2）+ 周期连接
`φ_R=φ_L, q_R=−q_L`；给定自由面 `φ=cos(kx)` 解通量，取 L2 Rayleigh 商 `Λ=Σqφl/Σφ²l`；另一条独立代码路径
装配自由面稠密 DtN 矩阵取同一商。由 Robin `Λ=ω²/(g−μω²)` 反解 ω。

**审计方（独立复算，不 import 被审方 driver）**：
- R1：按注册基准行 metric 的散文项，用交付代码 `tensorbem.ice` 重算 RK4 shooting / Robin 残差 / 冰厚研究 / 浅水长波偏差；
- R2：ref 由审计脚本从报告附录**重写**闭式求值，与被审 `numbers.json` 的 got 逐行对照（9 行）；
- R3：审计方自己的 ppw 阶梯（8…128）调 `strip_bem_depth_factor`，不用被审方的 `convergence_study`；
- R4：在隔离副本（m17373 @26cbde4 的 src/tests 字节拷贝）重跑 repro_cmd，比对重生成 numbers.json 是否逐字段一致；
- R5：pytest 交付测试（31 项）。

工具链：Python 3.11.6 / numpy 2.4.6 / matplotlib 3.10.8，单核 31 s，未上集群。
脚本与日志：`docs/audit_m17429/{audit_m17373.py, audit_metrics.json, logs/*.log}`。

## 3. 计算结果（数字表，程序化复算零手抄）

### 3.1 9 行 dispersion：ref（审计独立求值）vs got（BEM）

| 系列 | k [1/m] | kH | ω_got | ω_ref | ω 误差 | Λ_BEM | Λ_ref=k·tanh(kH) | Λ 误差 | 面元 |
|---|---|---|---|---|---|---|---|---|---|
| deep | 0.02 | 4 | 0.4391210 | 0.4388147 | 0.0698 % | 0.020015 | 0.019987 | 0.1421 % | 154 |
| deep | 0.05 | 10 | 0.6855525 | 0.6850826 | 0.0686 % | 0.050072 | 0.050000 | 0.1434 % | 164 |
| deep | 0.1 | 20 | 0.9493843 | 0.9487549 | 0.0663 % | 0.100145 | 0.100000 | 0.1446 % | 172 |
| deep | 0.2 | 40 | 1.2906232 | 1.2898283 | 0.0616 % | 0.200291 | 0.200000 | 0.1454 % | 182 |
| deep | 0.4 | 80 | 1.7005790 | 1.6996670 | 0.0537 % | 0.400583 | 0.400000 | 0.1458 % | 190 |
| shallow | 0.02 | 0.04 | 0.0888590 | 0.0885185 | 0.3846 % | 0.000806 | 0.000800 | 0.7712 % | 144 |
| shallow | 0.05 | 0.10 | 0.2210627 | 0.2205755 | 0.2209 % | 0.005006 | 0.004983 | 0.4442 % | 144 |
| shallow | 0.1 | 0.20 | 0.4366132 | 0.4361199 | 0.1131 % | 0.019783 | 0.019738 | 0.2304 % | 144 |
| shallow | 0.2 | 0.40 | 0.8358278 | 0.8353256 | 0.0601 % | 0.076087 | 0.075990 | 0.1285 % | 144 |

**max |ω_got/ω_ref−1| = 0.3850 %（容差 5 %）** —— 与被审自报 C1=0.003846 一致。

### 3.2 审计方自己的收敛阶梯（Λ 相对误差）

| 系列 | 8 ppw | 16 | 32 | 64 | 128 | 观测阶 |
|---|---|---|---|---|---|---|
| deep (H=200) | 3.6334 % | 1.2941 | 0.4301 | 0.1446 | 0.0517 | 1.49/1.59/1.57/1.48 |
| intermediate (H=20) | 2.3101 % | 0.8688 | 0.5178 | 0.1370 | 0.0491 | 1.41/0.75/1.92/1.48 |
| shallow (H=2) | 5.9085 % | 2.9660 | 0.9091 | 0.2304 | 0.0595 | 0.99/1.71/1.98/1.95 |

### 3.3 注册基准行（散文 metric）逐项复算

| 项 | 注册值 | 审计复算 | 判定 |
|---|---|---|---|
| deep-water k 求解一致性 | 4.50e-14 | 2.20e-13 | 同量级 PASS |
| shallow k 长波偏差 | 2.87e-02 @kH=0.426 | kH=0.426472 ✓、ω 回收 3.138e-14 ✓，但长波偏差 4.13 % | 部分不可复现（F4），仍<5% |
| RK4 shooting | 1.03e-13 | 1.43e-13 | PASS |
| Robin 残差 | 4.47e-15 | 4.02e-15 | PASS |
| 冰厚研究 | 0.0 % | 1.11e-14 | PASS |

### 3.4 两次运行的逐字节一致性

重生成 `numbers.json` 与被审工件**逐字段相同**（除 `environment` 块），摘要 sha256 `a101fa49b9aa1cb6…`；
`pytest tests/test_ice_mass_loading_dispersion.py` → **31 passed, 15.1 s**；driver exit 0，30.8 s。

## 4. 结果图

- **图1** got vs ref：ω 随 k 的曲线（深/浅水）与相对误差阶梯（5 % 容差线）；
- **图2** 审计方独立收敛阶梯（Λ 误差随 panels/λ）；
- **图3** BEM 真正独立算出的量：自由面深度因子 Λ_BEM vs 解析 Λ_ref=k·tanh(kH) 及误差。

## 5. 注意点（坑 / 局限 / 适用范围）

- **F1（blocking，对集成）**：分支 `agent/m17373` **未合并、无远程引用**（main `fc205ca` 无此模块）→ 生态主检出里没有该能力，
  gate 的 `grep integration_callers` 在主检出必然失败（预期裁定 L2）；交付物仅存本地工作树，有丢失风险。
- **F2**：「集成」只在库 API 层（`tensorbem.ice.mass_loading_omega_bem` + 包导出），**无既有求解路径调用**
  （`ice_rao.py` / `hydroelastic.py` 仍用闭式），caller 只有测试。
- **F3**：benchmark 头条量 ω 与 ref 共用同一条解析 Robin 映射（`ω=f(Λ)`，f 单调）→ ω 误差只是 Λ 误差的重新表达；
  BEM 真正独立算出的是 Λ（0.14 %@64 ppw）。因此「BEM 验证了色散关系」应表述为「BEM 验证了 Λ，Λ→ω 由解析 Robin 换算」。
- **F4**：注册基准行是散文 metric、无逐项 ref/got，其中浅水 2.87 % 不可精确复现（同 kH 下审计得 4.13 %；h_i 扫描 3.29–4.27 %）。
- **F5**：外部锚 WITHHELD（1950 原表离线不可达）→ L5 不考虑。
- **F6**：C2（两路线 1.5e-14）说服力有限：driven 与 DtN 共用同一套积分装配；真正独立的是精确模态边界值复原与加密收敛。
- 适用范围：无限冰盖、等深、法向入射、线性势流；冰缘、斜入射、有限冰筏不在范围内。

## 6. 误差判定（逐项，显式回答是否 <3%）

| 关键量 | 值 | 基准 | <3 %? | <5 %（注册容差）? |
|---|---|---|---|---|
| 9 行 ω 最大相对误差 | 0.385 % | Weitz–Keller 闭式（审计独立求值） | **是** | 是 |
| 64 ppw Λ 误差（deep） | 0.145 % | k·tanh(kH) | **是** | 是 |
| 64 ppw Λ 误差（shallow kH=0.04） | 0.771 % | k·tanh(kH) | **是** | 是 |
| 128 ppw Λ 误差（shallow） | 0.060 % | k·tanh(kH) | **是** | 是 |
| 注册行浅水长波偏差（复算） | 4.13 % | √(gH)·k | 否 | 是（余量 0.9 pp） |

**判级：L3**。数字诚实且强（远好于 3 % 线、加密单调、逐字节可复现），但 **L4 不判**：
(a) 集成在 main 不生效（F1/F2）；(b) 外部锚 WITHHELD（F5）；(c) 头条量 ω 非独立预测（F3）。
补上「合并进 main」+「一个外部数值锚」即可到 L4，工作量很小。
verdict 已落表 `tensor_maturity_verdicts` id=561（repo=TensorBEM, scope='m17373 CASE-BEM-ICE-DISPERSION', level=3, verdict_by='board-audit'），
gate 机器复核预期降 L2（原因=integration_callers 在 main 无引用，即 F1）。"""

FIGURES = [
    fig("fig_ref_vs_got.png", "图1 BEM got vs Weitz–Keller ref（审计独立求值）：ω(k) 深/浅水系列与相对误差（红虚线=5% 容差）"),
    fig("fig_convergence_independent.png", "图2 审计方独立收敛阶梯：Λ 相对误差随 panels/λ（8…128），三系列单调下降，观测阶 ~1–2"),
    fig("fig_depth_factor.png", "图3 BEM 真正独立算出的量：自由面深度因子 Λ_BEM 与解析 Λ_ref=k·tanh(kH) 对比及误差"),
]

if __name__ == "__main__":
    st, d = api("POST", "/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                                        "password": os.environ["TENSOR_AGENT_PASS"]})
    tok = d["access_token"]
    payload = {
        "title": "交差审计 m17373 CASE-BEM-ICE-DISPERSION（TensorBEM）：判级 L3 — BEM 数值诚实（0.385%<5%）、集成未并入 main",
        "summary": ("独立复算 m17373 的三件套：31 tests 通过、repro 逐字节可复现、9 行 ω max 0.385% < 5% 容差；"
                    "BEM 真解 BIE 取深度因子 Λ（0.14%@64ppw），但头条 ω 与 ref 共用解析 Robin 映射；"
                    "分支 agent/m17373 未合并、无既有求解路径调用 → 判 L3（L4 差一合并+一外部锚），verdict id=561。"),
        "scenario_key": "CASE-BEM-ICE-DISPERSION",
        "mission_id": 17429,
        "tags": ["board-audit", "TensorBEM", "CASE-BEM-ICE-DISPERSION", "Weitz-Keller", "maturity-L3", "m17373"],
        "markdown": MARKDOWN,
        "figures": FIGURES,
    }
    st, r = api("POST", "/tensor/reports", payload, tok)
    print(st, json.dumps(r)[:300])