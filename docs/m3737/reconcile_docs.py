#!/usr/bin/env python3
"""Append the m3737 convention-reconciliation block to docs/marine_bem_benchmarks.md.

The merge agent/m3181 -> main leaves one *semantic* collision that git itself
cannot see: both branches re-derived the same four cells of tables 10/12 from
the same error sequences but with different normalisations, and each branch
wrote its own values into the document.

* m3181 (section 4.12, this branch) quotes the observed order of ``B11'`` as
  ``log2(e0/e1)``, i.e. 1.62 / 1.88 -- the "ideal h halving" convention that was
  in table 10 when the branch forked.
* m3150 (main, section 4.10) re-quoted the same column against the *measured*
  mean edge length ratio ``h_mean`` = 1.87 / 1.96, i.e. 1.80 / 1.94, and also
  replaced the two-significant-digit rotational couplings of table 12 by
  2.15e-4 / 2.65e-5.

A plain text merge keeps both statements, so the document would assert two
different "documented" values for the same table cell.  This script inserts an
explicit reconciliation paragraph (values computed here, not typed) right after
section 4.12's conclusion, and rewrites that conclusion so it no longer claims
"the document needs no number change" unconditionally.
"""

from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/marine_bem_benchmarks.md"

OLD_CONCLUSION = (
    "**结论**：三维自由液面层（Kelvin/Havelock Green 函数、6-DOF 辐射–绕射、Kochin/Haskind 守恒、"
    "免盖板 CHIEF）的文档数字在四位有效数字上全部可复现，未发现腐化；文档不需要改数。\n"
)

#: The two registrations of table 10's B11' column and the two registrations of
#: table 12's rotational coupling, as they now coexist in the registry.
LOG2_B11 = (1.62, 1.88)          # group ``hulme3d``      (m3181, log2 convention)
MEASURED_B11 = (1.8021, 1.9377)  # group ``hulme_hemisphere`` (m3150, h_mean ratio)
ROT_2SIG = (2.2e-4, 2.6e-5)      # group ``wang_linton3d`` (m3181, 2 significant digits)
ROT_4SIG = (2.1510e-4, 2.6481e-5)  # group ``submerged_sphere`` (m3150, 4 significant digits)


def block() -> str:
    h2_implied = 2.0 ** (LOG2_B11[0] / MEASURED_B11[0])
    h3_implied = 2.0 ** (LOG2_B11[1] / MEASURED_B11[1])
    p2 = LOG2_B11[0] * math.log(2.0) / math.log(h2_implied)
    p3 = LOG2_B11[1] * math.log(2.0) / math.log(h3_implied)
    rot80 = ROT_2SIG[0] / ROT_4SIG[0] - 1.0
    rot320 = ROT_2SIG[1] / ROT_4SIG[1] - 1.0
    return (
        "\n"
        "**与 §4.10 / §4.11（m3150 复算订正）的口径对齐**：本节表格是 m3181 分支在旧基线上复算的结果；"
        "同一批格子随后由 m3150 在另一条分支上复算，并**改写了 4 个格子**（表 10 的 \\(B_{11}'\\) 观测阶两格、"
        "表 12 的球心转动耦合比两格）。两条分支合并（m3737）后，本节表格的「文档声明」列因此有 4 行与 §4.10/§4.11 现表不再逐字相同；"
        "两套取值都保留为可执行注册项，差异来源如下，**不是求解器漂移**：\n"
        "\n"
        "| 格子 | 本节（m3181，组 `hulme3d` / `wang_linton3d`） | §4.10/§4.11 现表（m3150，组 `hulme_hemisphere` / `submerged_sphere`） | 差异来源 |\n"
        "|---|---:|---:|---|\n"
        f"| 表 10 \\(B_{{11}}'\\) 观测阶 16→64 | {LOG2_B11[0]:.2f} | {MEASURED_B11[0]:.4f} | 观测阶归一化：本节按 \\(\\log_2(e_0/e_1)\\)（理想 \\(h\\) 减半），m3150 按实测平均棱长比 \\(h_0/h_1\\) |\n"
        f"| 表 10 \\(B_{{11}}'\\) 观测阶 64→256 | {LOG2_B11[1]:.2f} | {MEASURED_B11[1]:.4f} | 同上（该级实测 \\(h_1/h_2\\)） |\n"
        f"| 表 12 球心转动耦合比 80 面元 | {ROT_2SIG[0]:.1e} | {ROT_4SIG[0]:.4e} | 本节按两位有效数字声明（容差 2.5e-2 舍入带），m3150 改记四位有效数字 |\n"
        f"| 表 12 球心转动耦合比 320 面元 | {ROT_2SIG[1]:.1e} | {ROT_4SIG[1]:.4e} | 同上 |\n"
        "\n"
        "两者的关系是纯代数恒等式：同一误差序列在两种归一化下的观测阶之比为 \\(\\ln 2/\\ln(h_0/h_1)\\)。"
        f"由本节与 m3150 的数值反解得到实测棱长比 \\(h_0/h_1={h2_implied:.4f}\\)、\\(h_1/h_2={h3_implied:.4f}\\)（网格逐级加密时该比值趋近 2），"
        f"代回即 \\({LOG2_B11[0]:.2f}\\times\\ln 2/\\ln {h2_implied:.4f}={p2:.4f}\\)、\\({LOG2_B11[1]:.2f}\\times\\ln 2/\\ln {h3_implied:.4f}={p3:.4f}\\)（此步为恒等式的逆用，故回路自洽，不构成独立校验）；"
        f"表 12 两格的两位/四位有效数字之差为 {rot80:+.2%} / {rot320:+.2%}，落在本节声明的 2.5e-2 舍入带内。"
        "**非循环的独立校验**在 heavy 层完成：`hulme_hemisphere` 组用基准实际输出的逐级 `h_mean` 比值算观测阶、`hulme3d` 组用 \\(\\log_2\\) 算，"
        "两者各自与本表现值比对，均在 5e-3 容差内 PASS（`TENSORBEM_HEAVY_TESTS=1 pytest tests/test_declared_numbers.py`）；"
        "反解得到的 1.86→1.96 与基准输出的棱长比一致，且随加密趋近 2，是同一事实的两种呈现。"
        "该恒等与四条注册项的并存性由 `tests/test_declared_numbers.py::test_three_dimensional_b11_order_conventions_are_reconciled` 钉死"
        "（任一侧被静默丢弃或口径被改动即转红）。\n"
    )


NEW_CONCLUSION = (
    "**结论**：三维自由液面层（Kelvin/Havelock Green 函数、6-DOF 辐射–绕射、Kochin/Haskind 守恒、"
    "免盖板 CHIEF）的文档数字 63 格中 59 格在四位有效数字上可复现、无需改数；其余 4 格（表 10 的 "
    "\\(B_{11}'\\) 观测阶 16→64 / 64→256、表 12 的球心转动耦合比 80 / 320 面元）在同期 m3150 复算中按另一套"
    "归一化/舍入口径改写，本节表格保留原始口径值，两者的代数关系与并存的注册项见上「口径对齐」表。\n"
)


def main() -> int:
    text = DOC.read_text()
    assert "口径对齐" not in text, "reconciliation block already present"
    assert text.count(OLD_CONCLUSION) == 1
    text = text.replace(OLD_CONCLUSION, block() + "\n" + NEW_CONCLUSION)
    DOC.write_text(text)
    print("inserted reconciliation block into", DOC.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())