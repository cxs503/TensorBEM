#!/usr/bin/env python3
"""Publish the mission-3796 merge report to the Tensor platform.

Figures are downscaled before embedding: the platform silently drops report
payloads above ~250 KB (known trap), so three full-size PNGs (91 + 37 + 101 KB)
would blow the limit once base64-encoded.
"""
from __future__ import annotations

import base64
import io
import json
import os
import subprocess
from pathlib import Path

from PIL import Image

FIG_DIR = Path("docs/m3796/figures")
CAPTIONS = {
    "fig1_registry_union.png": "并集合并后的声明数字注册表盘点：main 贡献 84 条（蓝），agent/m3236 贡献 11 条 Dean 记录（红），合计 95 条无重复 key。",
    "fig2_test_parity.png": "测试对等性：同一并行设置下 main 基线树与合并树的 pytest 结果计数。",
    "fig3_dean_pinned.png": "m3236 钉桩的 Dean 表 7 共 11 行复算偏差（蓝=value 行 |rel dev|；灰=上限守护行 measured/bound，须 <1），虚线为 3% 验收参考。",
}
MAX_W = 560


def encode(path: Path) -> str:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if w > MAX_W:
        im = im.resize((MAX_W, int(h * MAX_W / w)), Image.LANCZOS)
    buf = io.BytesIO()
    # platform requires a real PNG payload; quantise so three figures stay well
    # under the ~250 KB request-body cliff after base64 inflation
    im.quantize(colors=64, method=Image.MEDIANCUT if hasattr(Image, "MEDIANCUT") else 0).save(
        buf, format="PNG", optimize=True
    )
    data = buf.getvalue()
    print(f"  {path.name}: {path.stat().st_size/1024:.0f} KB -> {len(data)/1024:.0f} KB png")
    return base64.b64encode(data).decode()


def main() -> int:
    md = Path("docs/m3796/merge_report.md").read_text()
    figures = []
    for name, caption in CAPTIONS.items():
        p = FIG_DIR / name
        if p.exists():
            figures.append({"name": name, "caption": caption, "png_base64": encode(p)})
    payload = {
        "title": "PR #134 治愈：agent/m3236 × main 三方并集合并（声明数字注册表）",
        "summary": (
            "PR #134 对 main 变脏的根因是两侧在同一锚点各自追加 declared_numbers 记录与 "
            "_measure 分派分支。以 origin/main 为 ours 做三方合并，唯一冲突文件的两个冲突块按语义并集解析"
            "（84 条 main 记录 + 11 条 Dean 记录，9 个 group 分支全通），"
            "并用 AST 门、父提交重放逐字节可复现门、126 路径多重集行审计（0 丢失）三重校验；"
            "测试与 main 基线同设置对比，未低于基线。"
        ),
        "scenario_key": "tensorbem_pr_conflict_union_merge",
        "mission_id": 3796,
        "tags": ["TensorBEM", "merge-conflict", "declared-numbers", "PR134", "verification"],
        "markdown": md,
        "figures": figures,
    }
    body = json.dumps(payload)
    Path("/tmp/report_3796.json").write_text(body)
    print(f"payload {len(body)/1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())