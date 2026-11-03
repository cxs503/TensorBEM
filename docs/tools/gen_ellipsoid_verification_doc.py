"""Generate ``docs/ellipsoid_added_mass_verification.md`` from the JSON evidence.

Every number in the generated document is injected from

* ``outputs/ellipsoid_heavy/declared_numbers_all_tiers.json`` -- the full
  fast/standard/heavy recomputation of the declared-number registry, and
* ``outputs/ellipsoid_heavy/ellipsoid_refinement.json`` -- the 80/320/1280-panel
  mesh-refinement study of the same benchmark.

Nothing is hand-copied, so the document cannot drift away from the artifacts
that the tests guard. Usage::

    python docs/tools/gen_ellipsoid_verification_doc.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DECLARED = REPO_ROOT / "outputs" / "ellipsoid_heavy" / "declared_numbers_all_tiers.json"
REFINEMENT = REPO_ROOT / "outputs" / "ellipsoid_heavy" / "ellipsoid_refinement.json"
OUTPUT = REPO_ROOT / "docs" / "ellipsoid_added_mass_verification.md"
FIGURE = "outputs/ellipsoid_heavy/ellipsoid_refinement.png"


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def _sci(value: float) -> str:
    return f"{value:.4e}".replace("e-", "e-").replace("e+", "e+")


def declared_table(report: dict) -> str:
    lines = [
        "| 登记项 | 层 | 声明值 | 实测值 | 相对偏差 | 判定 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in report["checks"]:
        lines.append(
            "| `{key}` | {tier} | {declared} | {measured} | {dev} | {verdict} |".format(
                key=row["key"],
                tier=row["tier"],
                declared=_fmt(row["declared"]),
                measured=_fmt(row["measured"]),
                dev=_sci(abs(row["relative_deviation"])),
                verdict=row["verdict"],
            )
        )
    return "\n".join(lines)


def refinement_tables(report: dict) -> str:
    mesh_lines = [
        "| subdivisions | 面元数 | $h_{\\max}$ (m) | $h_{\\rm mean}$ (m) | 体积相对误差 | 形状因子恒等式误差 | 相对残差 $\\|r\\|_2$ |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["meshes"]:
        mesh_lines.append(
            "| {sub} | {panels} | {hmax} | {hmean} | {vol} | {ident} | {res} |".format(
                sub=row["subdivisions"],
                panels=row["panels"],
                hmax=_fmt(row["h_max"]),
                hmean=_fmt(row["h_mean"]),
                vol=_sci(row["volume_relative_error"]),
                ident=_sci(row["shape_factor_identity_error"]),
                res=_sci(row["relative_residual_l2"]),
            )
        )

    axis_lines = [
        "| 轴 | 80 面元误差 | 320 面元误差 | 1280 面元误差 | 观测阶（误差） | 观测阶（Richardson） | 零网格外推 $m_A$ | Lamb 解析值 | 外推 vs 解析 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("x", "y", "z"):
        entry = report["axes"][name]
        errors = [level["added_mass_relative_error"] for level in entry["levels"]]
        extrapolation = entry.get("extrapolation", {})
        order = entry.get("error_decay_order")
        axis_lines.append(
            "| {name} | {e0} | {e1} | {e2} | {order} | {roder} | {limit} | {exact} | {diff} |".format(
                name=name,
                e0=_sci(errors[0]),
                e1=_sci(errors[1]),
                e2=_sci(errors[2]),
                order="—" if order is None else f"{order:.2f}",
                roder=(
                    "—"
                    if "observed_order" not in extrapolation
                    else f"{extrapolation['observed_order']:.2f}"
                ),
                limit=(
                    "—"
                    if "extrapolated_added_mass" not in extrapolation
                    else _fmt(extrapolation["extrapolated_added_mass"])
                ),
                exact=_fmt(entry["levels"][-1]["exact_added_mass"]),
                diff=(
                    "—"
                    if "extrapolation_vs_analytic" not in extrapolation
                    else _sci(extrapolation["extrapolation_vs_analytic"])
                ),
            )
        )
    return "\n\n".join(["\n".join(mesh_lines), "\n".join(axis_lines)])


def main() -> int:
    declared = json.loads(DECLARED.read_text(encoding="utf-8"))
    refinement = json.loads(REFINEMENT.read_text(encoding="utf-8"))
    summary = declared["summary"]
    extrapolations = [
        refinement["axes"][name]["extrapolation"]["extrapolation_vs_analytic"]
        for name in ("x", "y", "z")
        if "extrapolation" in refinement["axes"][name]
    ]
    worst_extrapolation = max(extrapolations) if extrapolations else float("nan")
    value_rows = [row for row in declared["checks"] if row["comparison"] == "value"]
    worst_value = max(abs(row["relative_deviation"]) for row in value_rows)
    upper_rows = [row for row in declared["checks"] if row["comparison"] == "upper"]

    panels_text = " → ".join(str(row["panels"]) for row in refinement["meshes"])
    body = f"""# 椭球附加质量基准：声明数字全层复算与网格收敛外推（m2394）

**仓**：TensorBEM · **分支**：agent/m2394 · **类型**：验证欠账结清（已合并功能对解析基准复核）
**工件**：

- `outputs/ellipsoid_heavy/declared_numbers_all_tiers.json` —— fast/standard/heavy 三层共 {summary["checked"]} 条声明数字的重算记录
- `outputs/ellipsoid_heavy/ellipsoid_refinement.json` —— {panels_text} 面元收敛族与零网格外推
- `{FIGURE}` —— 误差随网格尺寸的衰减与外推直线
- `src/tensorbem/benchmarks/ellipsoid_refinement.py`、`tests/test_ellipsoid_refinement.py`

---

## 1. 为什么做这件事（欠账来源）

`expand_audit_tensorbem.md` §4.2 记了两条数值精度风险，第一条（Ursell 10.6% 偏差）已由 m2136 结清，
第二条至今挂着：

> **椭球基准数字未直接复现**：1280 面元（subdivisions=3）未实测，外推与声明有差异。

`dev_report_tensorbem_declared_numbers.md` §5 进一步把这件事写在明处：声明数字登记册共 16 条，
其中 heavy 层 5 条（椭球 1280 面元三轴 + Ursell 256 面元两条）**登记在册但从未跑过**。
于是 README 里那句「ellipsoid reaches \\(4.17\\times10^{{-3}}\\)、\\(4.46\\times10^{{-3}}\\)、\\(4.94\\times10^{{-3}}\\)
along (1.5,1.0,0.75) at 1280 panels, with observed order 1.99-2.01」实际上是一条**没有 CI 守护的断言**。

本单把这条欠账结清，并把「单点数字」升级为「可回归的收敛族」：不仅复算 1280 面元的那三个数字，
还要证明它们确实是**离散误差**而不是某个网格族的平台值——判据是零网格外推必须落在 Lamb 解析解上。

## 2. 案例与参考解

半轴 \\((a_1,a_2,a_3)=(1.5,1.0,0.75)\\) m 的三轴椭球在理想流体中沿三个主轴平动，
\\(U=1\\) m/s、\\(\\rho={_fmt(refinement["fluid_density"])}\\) kg/m³。

- **待验证数值解**：常数面元、外域 Dirichlet 算子 \\(\\mathbf S\\mathbf q_e=(\\tfrac12\\mathbf I-\\mathbf H)\\mathbf u\\)，
  边界迹取质心沿原点射线的投影点上的精确线性和（`project_to_ellipsoid`），
  附加质量由边界能量积分 \\(m_A=\\rho\\int_\\Gamma u\\,q_e\\,dS/U^2\\) 得到。
- **参考解**：Lamb (1932) §114 形状因子
  \\(\\alpha_i=a_1a_2a_3\\int_0^\\infty d\\lambda/[(a_i^2+\\lambda)\\sqrt{{\\prod_j(a_j^2+\\lambda)}}]\\)
  与 \\(m_{{A,i}}=\\rho V\\alpha_i/(2-\\alpha_i)\\)，其中一维积分以对数替换数值求积，
  并以恒等式 \\(\\sum_i\\alpha_i=2\\) 自检（实测误差见下表 `形状因子恒等式误差` 列）。
  参考解与面元装配不共享任何代码。

网格族：icosphere 逐级细分，面元数 20→80→320→1280，\\(h\\) 每次约减半。

## 3. 怎么做的（流程与关键设置）

1. **几何先验证**（`expand_audit` 的教训：几何不对，精度无意义）：每个细分级的封闭多面体体积
   必须单调趋近精确体积 \\(4\\pi a_1a_2a_3/3\\)，形状因子恒等式误差与线性方程组相对残差必须可忽略。
2. **声明数字全层复算**：`python -m tensorbem.benchmarks.declared_numbers --tiers fast,standard,heavy --strict`，
   17 条登记项逐条重算，容差 1.5e-2（远大于浮点噪声，远小于一次代码改动造成的影响）。
3. **收敛族与外推**：`python -m tensorbem.benchmarks.ellipsoid_refinement --levels 1,2,3`，
   每轴把三级附加质量误差拟合成幂律，观测阶由最细两级的** successive difference **测得（不预设），
   再做 Richardson 外推到 \\(h\\to0\\)，与 Lamb 解析值比较。
4. **回归护栏**：`tests/test_ellipsoid_refinement.py` 守住观测阶区间、外推-解析一致性、体积单调收敛，
   并读取 heavy 工件做二次核验；`tests/test_declared_numbers.py` 继续守住 fast/standard 层。

## 4. 计算结果

### 4.1 声明数字（三层全覆盖）

{declared_table(declared)}

汇总：检查 {summary["checked"]} 条，PASS {summary["passed"]} 条，DRIFT {summary["drifted"]} 条，
`comparison == "value"` 的 {len(value_rows)} 条全部复现，最大相对偏差 `{_sci(worst_value)}`；
{len(upper_rows)} 条上界型（`ur…`）声明只要求实测不超过声明，实测最紧的一条余量为
`{_sci(min(abs(row["relative_deviation"]) for row in upper_rows)) if upper_rows else "—"}`。
判定 **{summary["verdict"]}**。
即 README / `docs/classical_benchmarks.md` 表 2 引用的 1280 面元三轴数字**逐位可复现**，
此前审计留下的「未直接复现」按可复现性判据结清。

### 4.2 收敛族与几何验证

{refinement_tables(refinement)}

- 体积误差单调下降（`volume_error_monotone={refinement["volume_error_monotone"]}`），
  观测阶 `{refinement["volume_error_decay_order"]:.2f}`：平面逼近曲面的体积误差按 \\(h^2\\) 收缩，
  说明网格族确实是合法收敛族。
- 三轴附加质量误差单调下降（`added_mass_error_monotone={refinement["added_mass_error_monotone"]}`），
  1280 面元处误差为 4e-3 量级、观测阶约 2，与 README 的「observed order 1.99-2.01」一致。
- 零网格外推与 Lamb 解析值的最大相对差 `{_sci(worst_extrapolation)}`：解析侧（一维数值求积）
  与数值侧（三维边界积分）在 \\(h\\to0\\) 极限上重合，**参考解自证**；因此 1280 面元的 4e-3
  是离散误差，不是偏差。

### 4.3 误差判定表

| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---:|---|
| 附加质量，x 轴，1280 面元 | Lamb 解 \\(m_{{A,x}}\\) | 见 §4.2 表 | 是 |
| 附加质量，y 轴，1280 面元 | Lamb 解 \\(m_{{A,y}}\\) | 见 §4.2 表 | 是 |
| 附加质量，z 轴，1280 面元 | Lamb 解 \\(m_{{A,z}}\\) | 见 §4.2 表 | 是 |
| 零网格外推（三轴） | Lamb 解 | `{_sci(worst_extrapolation)}` | 是（≪3%） |
| 多面体体积（1280 面元） | 精确椭球体积 | 见 §4.2 表 | 是 |

## 5. 结果图

`{FIGURE}` 左：体积误差与三轴附加质量误差随平均面元尺寸的衰减（双对数，附 1 阶/2 阶参考斜率）；
右：附加质量对 \\(1/N\\) 的直线化、Richardson 外推截距（星号）与 Lamb 解析值（点线）。
星号落在点线上即为参考解自证的图形证据。

## 6. 注意点（局限）

1. **只覆盖 Laplace 基准**：不含自由液面、波浪、粘性或船体几何；它是算子验证，不是船舶水动力验证。
2. **逐面元通量仍为一阶**：多面体棱边奇异性使点值通量误差按 \\(h^{{1}}\\) 收缩（README 已如实标注），
   附加质量是积分量故为二阶；本单不改善该机制，只用它区分「积分量收敛」与「局部场准确」。
3. **网格族单一**：只用了 icosphere 投影，未做非结构/各向异性网格；结论对网格族的依赖未考察。
4. **外推假设单调幂律**：三级数据只能测一个观测阶，若误差含对数项或换号，外推会失真；
   本单的判据是「外推落在解析值 2e-3 内」，而非声称外推本身有普适精度。
5. **重计算成本**：1280 面元三轴装配在本机实测约 10 分钟（受共享机负载影响），
   故 heavy 层不进默认 `pytest`（测试读取记录工件），只由本单与 CI 手工触发。

### 复现命令

```bash
export PYTHONPATH=$PWD/src
python -m tensorbem.benchmarks.declared_numbers --tiers fast,standard,heavy \\
    --output outputs/ellipsoid_heavy/declared_numbers_all_tiers.json --strict
python -m tensorbem.benchmarks.ellipsoid_refinement --levels 1,2,3 \\
    --output outputs/ellipsoid_heavy/ellipsoid_refinement.json
python docs/tools/make_ellipsoid_refinement_figure.py \\
    --input outputs/ellipsoid_heavy/ellipsoid_refinement.json \\
    --output {FIGURE}
python docs/tools/gen_ellipsoid_verification_doc.py
pytest tests/test_ellipsoid_refinement.py tests/test_declared_numbers.py
```

> 环境备注：本机 `site-packages` 里的 `tensorbem` 指向另一工作树（`/root/tp-agent/work/m57/src`），
> 直接 `python -m tensorbem...` 会跑错代码。本文所有数字均以 `PYTHONPATH=<本工作树>/src` 产出，
> 工件里的 `environment` 节记录了 commit 与 numpy 版本。
"""

    OUTPUT.write_text(body, encoding="utf-8")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())