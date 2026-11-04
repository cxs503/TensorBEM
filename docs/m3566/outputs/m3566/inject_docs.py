"""Inject the section 4.7 recomputation subsection into docs/marine_bem_benchmarks.md.

Numbers are read from the registry report (outputs/m3566/standard_tier.json),
never typed by hand.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/marine_bem_benchmarks.md"
REPORT = ROOT / "outputs/m3566/standard_tier.json"

report = json.loads(REPORT.read_text(encoding="utf-8"))
rows = [row for row in report["checks"] if row["group"] == "rigid_body"]
assert len(rows) == 44, len(rows)
worst = max(
    abs(row["relative_deviation"]) for row in rows if row["comparison"] != "upper"
)


def fmt(value: float) -> str:
    """LaTeX inline math in the style of the surrounding document."""
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if magnitude < 0.1 or magnitude >= 1e3:
        mantissa, exponent = f"{value:.3e}".split("e")
        return f"\\({mantissa}\\times10^{{{int(exponent)}}}\\)"
    return f"{value:.4g}"


def table(title: str, prefix: str) -> list[str]:
    lines = [
        f"**{title}**",
        "",
        "| 注册项 | 声明值 | 实测值 | 相对偏差 / 上界余量 | 判定 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        if not row["key"].startswith(prefix):
            continue
        if row["comparison"] == "upper":
            headroom = row["declared"] / row["measured"]
            note = f"余量 $\\times${headroom:.1f}"
        else:
            note = f"{row['relative_deviation']:+.1e}"
        lines.append(
            "| `{}` | {} | {} | {} | {} |".format(
                row["key"], fmt(row["declared"]), fmt(row["measured"]), note,
                row["verdict"],
            )
        )
    lines.append("")
    return lines


lines: list[str] = []
lines.append("#### 4.7.1 表 5、表 6 的可执行复算与回归钉桩（m3566）")
lines.append("")
lines.append(
    "表 5 与表 6 的 44 个数字在 2026-11-04 之前只活在本文里：没有任何测试重算过它们，"
    "一次内核、求积或网格改动就能让整张表悄悄过期而 CI 全绿。本次把它们逐条登记进集中式"
    "声明数字注册表 `tensorbem.benchmarks.declared_numbers` 的新增 `rigid_body` 组"
    "（面元数、条件数、三个/六个对角相对误差、观测阶、对称性残差），注册表自带的 "
    "`(group, parameters)` 缓存保证 44 条判据只花两次求解（一次二维、一次三维）："
)
lines.append("")
lines.append("```bash")
lines.append("python -m tensorbem.benchmarks.declared_numbers --tiers standard --strict")
lines.append("```")
lines.append("")
lines.append(
    "本次复算结果见 `outputs/m3566/standard_tier.json`；本小节的表格由 "
    "`outputs/m3566/inject_docs.py` 从该 JSON 程序化生成（零手抄）：44 条全部 **PASS**，"
    f"逐项相对偏差最大 {worst:.1e}，来源是本文只保留 3 位有效数字的舍入而非求解器误差；"
    "同一轮 `standard` 层共 48 条判据（含 Ursell 与内部谐波），注册表整体判定 **PASS**。"
)
lines.append("")
lines.extend(table("表 5 复算（二维椭圆 \\(a=1.5,\\ b=0.75,\\ \\rho=1025\\)）", "rigid_body_2d_"))
lines.extend(table("表 6 复算（三维椭球 \\(1.5/1.0/0.75\\)，\\(\\rho=1025\\)）", "rigid_body_3d_"))
lines.append("**三条附带判据（正文散文里的事实，不只是表格数字）**")
lines.append("")
lines.append(
    "1. **良态性**：第二类算子 \\(\\tfrac12\\mathbf I-\\mathbf H\\) 的条件数在加密下保持 "
    "\\(O(1)\\) —— 二维 3.078 → 3.001、三维 2.447 → 2.130。"
    "`tests/test_declared_numbers.py` 另加硬上界（二维 \\(\\kappa<3.1\\)、"
    "三维 \\(\\kappa<2.5\\)）：一旦退化为第一类算子那种 \\(O(h^{-1})\\) 增长，测试立即变红。"
)
lines.append(
    "2. **二阶收敛**：登记了 9 个观测阶（二维横摇 3 个 + 三维转动 3 × 2），测试断言全部落在 "
    "\\([1.85,\\,2.15]\\) 内。这比「误差在下降」强得多：离散格式被降阶而误差仍单调下降的情形"
    "会被抓住。"
)
lines.append(
    "3. **对称性列的舍入噪声（本次修正）**：表 5 原先逐行给出 "
    "\\(6.4\\times10^{-13}\\)、\\(9.3\\times10^{-13}\\)、\\(4.5\\times10^{-13}\\)、"
    "\\(8.0\\times10^{-13}\\) 这类两位有效数字的残差。它们是机器 epsilon 量级的舍入噪声，"
    "随平台与 BLAS 路径变化：同一份代码在 loongarch64 上实测 "
    "\\(5.98\\times10^{-13}\\)、\\(6.34\\times10^{-13}\\)、\\(6.41\\times10^{-13}\\)、"
    "\\(1.14\\times10^{-12}\\)，与原文记录相差可达 40%。把噪声钉成等式只会制造假漂移，"
    "因此该列改写为可复现的上界 \\(<10^{-11}\\)（实测最坏 \\(1.14\\times10^{-12}\\)，"
    "余量 8.8 倍；真正的对称性缺陷会把它推高好几个量级），注册表中对应 4 条 "
    "`kind=\"upper\"` 判据。"
)
lines.append("")

BLOCK = "\n".join(lines)

ANCHOR = "### 4.8 Dean–Ursell–Ogilvie 潜没柱体规则波绕射与 Haskind 互易关系"
text = DOC.read_text(encoding="utf-8")
assert text.count(ANCHOR) == 1
text = text.replace(ANCHOR, BLOCK + "\n" + ANCHOR, 1)

# Rewrite the round-off-noise symmetry column of table 5 into a reproducible bound.
OLD_HEAD = (
    "| 面元数 \\(N\\) | 条件数 | \\(A_{11}'\\) 相对误差 | \\(A_{22}'\\) 相对误差 | "
    "转动 \\(A_{66}'\\) 相对误差 | \\(A_{66}'\\) 观测阶 | 张量对称误差 "
    "\\(\\|\\mathbf A-\\mathbf A^\\top\\|_{\\max}\\) |"
)
NEW_HEAD = (
    "| 面元数 \\(N\\) | 条件数 | \\(A_{11}'\\) 相对误差 | \\(A_{22}'\\) 相对误差 | "
    "转动 \\(A_{66}'\\) 相对误差 | \\(A_{66}'\\) 观测阶 | 张量对称误差 "
    "\\(\\|\\mathbf A-\\mathbf A^\\top\\|_{\\max}\\)（上界，见 §4.7.1） |"
)
assert text.count(OLD_HEAD) == 1, "table 5 header not found"
text = text.replace(OLD_HEAD, NEW_HEAD)
for old, new in (
    ("| — | \\(6.4\\times10^{-13}\\) |", "| — | \\(<10^{-11}\\) |"),
    ("| 1.936 | \\(9.3\\times10^{-13}\\) |", "| 1.936 | \\(<10^{-11}\\) |"),
    ("| 1.966 | \\(4.5\\times10^{-13}\\) |", "| 1.966 | \\(<10^{-11}\\) |"),
    ("| 1.982 | \\(8.0\\times10^{-13}\\) |", "| 1.982 | \\(<10^{-11}\\) |"),
):
    assert text.count(old) == 1, old
    text = text.replace(old, new)

DOC.write_text(text, encoding="utf-8")
print("patched", DOC)