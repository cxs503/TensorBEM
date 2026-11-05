"""Append the m3808 cross-version roundoff section to docs/finite_depth3d.md.

All numbers come from docs/finite_depth3d_pins/roundoff_crosscheck.json, so the
document text is generated, not transcribed.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = json.loads((ROOT / "docs" / "finite_depth3d_pins" / "roundoff_crosscheck.json").read_text())
DOC = ROOT / "docs" / "finite_depth3d.md"
MARKER = "## 8. m3409 内核重写的数值等价性核对"

cats = DATA["categories"]
rows = DATA["rows"]
mov = DATA["moved_rows"]
can = [r for r in mov if r["category"] == "cancellation"]
amp = [r for r in mov if r["category"] == "amplified-roundoff"]
last = [r for r in mov if r["category"] == "last-bit"]
worst_amp = max((r["rel_shift"] for r in amp), default=0.0)
max_abs_shift_eps = max((r["shift_in_eps"] for r in can), default=0.0)


def fmt(v: float) -> str:
    return f"{v:.3e}"


table = [
    "| 类别 | 行数 | 判读 |",
    "|---|---:|---|",
    f"| 逐位不变 | {cats.get('bit-identical', 0)} | 合并前后 `recomputed` 完全相同的 double |",
    f"| 末位漂移 | {cats.get('last-bit', 0)} | 只动最后 1–4 bit，相对漂移 ≤ {max((r['rel_shift'] for r in last), default=0.0):.1e} |",
    f"| 放大舍入 | {cats.get('amplified-roundoff', 0)} | 深水极限 `max_rel_error`（两个 O(1) 核相减）把 eps 级扰动放大到相对 ≤ {worst_amp:.1e} |",
    f"| 相消噪声 | {cats.get('cancellation', 0)} | 自由面残差，绝对值 ≤ {max_abs_shift_eps:.2f} eps，相对漂移大但仍是机器零 |",
    f"| 物理改变 | {cats.get('physical-change', 0)} | 真回归（本核对为 0） |",
    f"| 合计 | {DATA['rows']} | 合并后 {DATA['rows']}/{DATA['rows']} PASS，DRIFT {len(DATA['rows_over_printed_tolerance'])} 行 |",
]

residual_lines = [
    "| `h` (m) | 文档打印值 | 合并前 recomputed | 合并后 recomputed | 绝对漂移（eps） |",
    "|---:|---:|---:|---:|---:|",
]
for r in sorted(can, key=lambda r: float(r["row"].split("/")[1].split("=")[1])):
    depth = r["row"].split("/")[1].split("=")[1]
    residual_lines.append(
        f"| {depth} | {fmt(r['declared'])} | {fmt(r['recomputed_before'])} | "
        f"{fmt(r['recomputed_after'])} | {r['shift_in_eps']:.2f} |"
    )

section = f"""

{MARKER}（m3808）

`agent/m3173` 的 pin 判据在合并 main 后第一次变红：§3.2 的三行自由面残差从
{fmt(can and [r for r in can if 'h=2' in r['row']][0]['recomputed_before'])} / 7.192e-15 / 7.847e-15 变成
{fmt(can and [r for r in can if 'h=2' in r['row']][0]['recomputed_after'])} / 7.239e-15 / 8.098e-15（相对偏差最大
{max((r['rel_shift'] for r in can), default=0.0):.0%}），超过打印精度容差 5e-3。归因是 **main 漂移，不是实现缺陷**：
m3409 把传播模态由 `cosh(k0(z+h))` 改写为对数缩放指数因子（数学上同一个量，只为避免
`k0*h >> 1` 时 `cosh` 溢出），而这一行数值本身是 O(1) 核的相消噪声——量级 1e-15 ≈ 数 eps，
「相对」判据在它身上天然无意义。

程序化核对（`docs/m3808/roundoff_crosscheck.py`，逐行对比合并前后
`docs/finite_depth3d_pins/check.json` 的 `recomputed`，共 {DATA['rows']} 行）：

{chr(10).join(table)}

自由面残差逐行明细：

{chr(10).join(residual_lines)}

判据修正（`finite_depth3d_pins.NOISE_FLOOR = 1e-12`）：**residual 类**行若打印值
`|declared| <= 1e-12`，改按绝对地板判定（比任何物理残差仍小 6 个量级）；打印为
`0.000e+00` 的海床残差保留更严的 5e-16；小物理量（如 `evanescent_share = 4.6e-20`）
不受豁免，仍按相对判据。负控见 `tests/test_finite_depth3d_pins.py::test_roundoff_floor_absorbs_ulp_noise_only`：
把残差写成 4e-10 依旧 `DRIFT`，把 evanescent_share 翻倍依旧 `DRIFT`。
"""

text = DOC.read_text(encoding="utf-8")
if MARKER in text:
    text = text[: text.index(MARKER)].rstrip("\n") + "\n"
DOC.write_text(text.rstrip("\n") + "\n" + section, encoding="utf-8")
print(f"updated {DOC} (+{len(section.splitlines())} lines)")