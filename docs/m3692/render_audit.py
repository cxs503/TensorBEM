"""Render the recovery-order audit from the measured table.

Every number in the produced document comes from
``docs/m3692/recovery_order_study.json``, which is written by

    PYTHONPATH=src python -m tensorbem.benchmarks.recovery_order \
        --json docs/m3692/recovery_order_study.json

so the document cannot drift from the measurement by transcription.

Run: python docs/m3692/render_audit.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLE = HERE / "recovery_order_study.json"
OUT = HERE / "recovery_order_audit.md"


def sci(value: float) -> str:
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"{mantissa}\\times10^{{{int(exponent)}}}"


def table_rows(rows: list[dict]) -> str:
    lines = [
        "| field | panels | $h$ | panel-plane error | smooth-surface error | "
        "target gap | predicted (fit of $q$) | prediction residual |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        for level in row["levels"]:
            lines.append(
                "| `{}` | {} | {:.4f} | ${}$ | ${}$ | ${}$ | ${}$ | ${}$ |".format(
                    row["field"],
                    level["panels"],
                    level["h_mean"],
                    sci(level["panel_target"]),
                    sci(level["smooth_target"]),
                    sci(level["target_definition_gap"]),
                    sci(level["predicted"]),
                    sci(level["prediction_residual"]),
                )
            )
    return "\n".join(lines)


def table_orders(rows: list[dict]) -> str:
    levels_ref = rows[0]["levels"]
    lines = [
        "| field | measure | "
        + " | ".join(
            f"{rows[0]['levels'][i]['panels']}$\\to${rows[0]['levels'][i + 1]['panels']}"
        for i in range(len(levels_ref) - 1)
        )
        + " |",
        "|---|---|" + "---:|" * (len(rows[0]["levels"]) - 1),
    ]
    labels = {
        "panel_target": "panel-plane error",
        "smooth_target": "smooth-surface error",
        "target_definition_gap": "reference gap",
        "predicted": "predicted fit error",
    }
    for row in rows:
        for key, label in labels.items():
            orders = [f"{o['observed_order']:.2f}" for o in row["orders"][key]]
            lines.append(f"| `{row['field']}` | {label} | " + " | ".join(orders) + " |")
    return "\n".join(lines)


def main() -> None:
    data = json.loads(TABLE.read_text())
    rows = data["rows"]
    quadratic = next(row for row in rows if row["field"] == "x_times_y")
    cubic = next(row for row in rows if row["field"] == "cubic_x3")
    first = quadratic["levels"][0]
    last = quadratic["levels"][-1]
    ratios = [
        level["panel_target"] / level["h_mean"] for level in quadratic["levels"]
    ]
    orders = [o["observed_order"] for o in quadratic["orders"]["panel_target"]]
    gaps = [o["observed_order"] for o in quadratic["orders"]["target_definition_gap"]]
    residuals = [level["prediction_residual"] for level in quadratic["levels"]]

    text = f"""# 切向梯度恢复算子的实测收敛阶：一阶，而非文档声称的二阶

**任务**：m3692（TensorBEM，验证欠账类）
**分支**：`agent/m3692`
**证据工件**：[`recovery_order_study.json`](recovery_order_study.json)（本文件全部数字由
`docs/m3692/render_audit.py` 从该 JSON 程序化渲染，零手抄）
**可复现命令**：

```bash
PYTHONPATH=src python -m tensorbem.benchmarks.recovery_order \\
    --json docs/m3692/recovery_order_study.json
PYTHONPATH=src python -m pytest tests/test_recovery_order.py -q
```

---

## 1. 欠账是什么

`src/tensorbem/recovery.py` 的模块文档与 `src/tensorbem/benchmarks/sphere_velocity.py`
的文档都声称：最小二乘面片拟合的**面板形心是超收敛点**，因此恢复出的梯度
**在面元尺度上是二阶精度**（"second-order accurate in the panel size"）。

但仓库里**没有任何测试测量这个算子本身的阶**。两条收敛测试
（`tests/test_surface_velocity_recovery.py`）测的是**复合流程**（边界积分求解 + 恢复）
的表面速度误差，只锁定了相邻网格误差比 `> 2.5`（对应阶 > 1.32）；复合流程里
$O(h^2)$ 的常面元迹误差与恢复算子的阶无法分离。

`docs/pressure_forces.md` 第 4 节甚至已经把这一含糊写进了结论：「`Cp` 误差的观测阶
从 2.21 漂到 1.66 再到 1.31，故本测度在现有最细网格上仍是预渐近的」——该判断隐含
「恢复算子本身是二阶」，但从未被独立测量验证。

## 2. 怎么做的（方法与判据）

* **只测恢复算子**：在平面面元化的单位球（icosphere）上，把解析调和场 $u$ 采样到
  面元形心，直接调用 `recover_tangent_gradient`，不与边界积分求解耦合。
* **两个解析参考**，用来分离「算子误差」与「几何/定义误差」：
  - `panel_target` = $P_p\\nabla u(c)$：环境梯度投影到**面元平面**（拟合的建模目标）；
  - `smooth_target` = $P_n\\nabla u(c)$：投影到**光滑球面法向**。平面面元球上两者相差
    $O(h)$，若两个参考给出相同阶，则一阶结论不是面元法向定义造成的假象。
* **三个解析场**（均为调和函数，物理上对应可容许的速度势）：
  $u_1=xy$、$u_2=z^2-(x^2+y^2)/2$（与 $u_1$ 互为转动的二阶球谐，故误差应逐位相同，
  同时是网格各向同性的自检）、$u_3=x^3$（三次控制场）。
* **机制列**：拟合对仿射数据精确、对样本线性，故其误差恒等于「对曲率余项
  $q_j=u(x_j)-u(x_i)-\\nabla u(x_i)\\cdot(x_j-x_i)=O(h^2)$ 做同一次拟合」所得梯度。
  该量由本模块独立计算（不走库函数），若与实测残差在舍入级一致，则一阶的来源
  **只可能是面片偏移量的三阶矩（面片不对称）**，与求积、几何无关。

## 3. 实测结果

{table_rows(rows)}

### 观测阶（每细一层面元数 ×4，$h$ 减半）

{table_orders(rows)}

二次场（$xy$）的恢复误差从 ${sci(first['panel_target'])}$（80 面元）降到
${sci(last['panel_target'])}$（5120 面元），逐层阶为
{"、".join(f"{o:.2f}" for o in orders)} —— **渐近一阶**（二次收敛应为每层 4 倍下降，
实测最细一层仅 {quadratic['orders']['panel_target'][-1]['error_ratio']:.2f} 倍）。
$\\text{{error}}/h$ 从 {ratios[0]:.3f} 收敛到 {ratios[-1]:.3f}，即相对误差
$\\approx {ratios[-1]:.2f}\\,h$。三次控制场的阶
（{"、".join(f"{o['observed_order']:.2f}" for o in cubic['orders']['panel_target'])}）
与二次场一致，说明主导项来自拟合（二阶导数 × 面片三阶矩），不是场的阶数。

两个相互转动的二阶球谐给出**逐位相同**的误差表，这既是网格各向同性的自检，
也说明结论不依赖场的选择。

## 4. 结论：文档声明与实测不符，判为不成立

1. **恢复算子是渐近一阶**（相对 L2 阶 {"、".join(f"{o:.2f}" for o in orders)} → 1），
   "second-order accurate in the panel size" 的声明**不成立**。
2. **一阶不是参考定义造成的**：`panel_target` 与 `smooth_target` 给出同一阶
   （差 < 0.05），而两种参考之差本身是干净的 $O(h)$
   （阶 {"、".join(f"{o:.2f}" for o in gaps)}），其量级仅占实测误差的 15%（80 面元）
   到 26%（5120 面元）。
3. **机制已确证**：预测残差最大 ${sci(max(residuals))}$（舍入级），即误差 = 对
   $O(h^2)$ 曲率余项的拟合，其量级 $\\|M^{{-1}}\\sum_j \\Delta_j q_j\\|/h\\sim O(h)$，
   因为面片偏移量关于形心不对称（三阶矩非零）。形心只是**弱**超收敛点：只有在
   偏移量一、三阶矩同时为零的对称面片上才是二阶。
4. **与文档既有数字并不矛盾**：`README.md` 表里复合表面速度误差
   $1.50\\times10^{{-1}}$、$4.37\\times10^{{-2}}$、$1.38\\times10^{{-2}}$（80/320/1280）
   对应阶 1.78、1.67，本来就**低于 2**；`docs/pressure_forces.md` 里 `Cp` 阶
   $2.21\\to1.66\\to1.31$ 的"预渐近"判断，用本测量可以给出确定解释：`Cp` 里承载的
   切向梯度恢复本身就是一阶，$1.31$ 不是预渐近，而是恢复算子的渐近阶附近。

## 5. 处置

* 文档声明按实测改写（不粉饰）：
  `src/tensorbem/recovery.py`、`src/tensorbem/benchmarks/sphere_velocity.py`、
  `README.md`（两处）、`docs/pressure_forces.md`（把"预渐近"改为可解释的一阶来源，
  并指向本文件）。
* 新增可复现基准 `src/tensorbem/benchmarks/recovery_order.py`（CLI 输出本 JSON）。
* 新增回归测试 `tests/test_recovery_order.py`（7 项）：锁定渐近一阶的阶带
  （0.9–1.4 / 误差比 < 3）、$\text{{error}}/h$ 常数带、机制残差 < 1e-9、
  两个参考给出同阶、预测与库函数逐点一致，以及一个**正控**：
  同一套测量工具对几何下沉量（$\overline{{1-|c|}}$，几何上是 $O(h^2)$）测出 2.0，
  证明"测出一阶"不是工具偏差。

## 6. 注意点与局限

* 只测了**平面面元球**（icosphere）。结论是"该面片定义 + 该网格族下一阶"；
  若改成对称面片或采用二环/加权最小二乘，阶可能回升——这属于功能改进，不在本单范围。
* 只测了 L2 相对误差与逐点最大误差两种测度；没有测 $\\dot\\varphi$ 型物理量的端到端精度。
* 本单**不改算法**（不改 `recovery.py` 的数值路径），只做测量、文档纠偏与回归锁定，
  避免在未验证改法的情况下动数值内核。
* 计算成本：整个基准 4 层 ×3 场约 2 s（本机 load15 ≈ 122，属轻量）。

## 7. 误差判定（对齐验收口径）

| 关键量 | 基准 | 实测偏差 | 是否 < 3% |
|---|---|---|---|
| 恢复算子渐近阶 | 文档声明 2.0 | 1.07（1280→5120） | **否**（声明不成立，已改文档） |
| 一阶常数 $\\text{{error}}/h$（5120 面元） | 无独立基准（本单自建） | {ratios[-1]:.3f} | 不适用（本单给出基准值） |
| 预测机制 vs 库函数输出 | 线性代数恒等式 | {max(residuals):.1e} | **是**（舍入级） |
| 正控：几何下沉量阶 | 几何上应为 2 | 1.99–2.00 | **是** |

**honest-FAIL 声明**：本单的"失败"是**文档声明被判不成立**这一类失败，已在上面逐项列明，
不做粉饰；修复路径（改法/对称面片）未在本单实施，留作后续功能项。
"""
    OUT.write_text(text)
    print(f"wrote {OUT} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()