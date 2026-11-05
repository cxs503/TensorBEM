#!/usr/bin/env python3
"""Post the PR #109 merge-healing report to the platform report API.

The markdown body is the generated report (docs/m3984/build_report.py); the
figures are the generated PNGs, re-encoded so the whole payload stays well
inside the API's 250 KB limit (the platform has silently dropped oversized
payloads before), and the numeric summary is attached as machine-readable
``metrics`` rather than re-typed prose.

    TENSOR_API_BASE=... TENSOR_AGENT_USER=... TENSOR_AGENT_PASS=... \
        python docs/m3984/post_report.py --mission-id 3984
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "m3984" / "pr109_merge_healing.md"
FOUT = ROOT / "docs" / "m3984" / "figures"
CAPTIONS = {
    "mfs_rows_vs_reference.png": (
        "MFS 三行（64/128/256 面元、24 源）对 Ursell 闭式解的相对误差："
        "三条均在 O(N^-1) 参考线附近且远低于 3% 验收线（最大 0.35%）。"
        "main 原本只有前两行，第三行是本分支被合并保留的内容。"
    ),
    "panel_referee_resolution.png": (
        "均匀分片 Gauss 裁判在共面 log 板块上的分辨率极限：观测阶 1.000，"
        "400 片残差 7.5e-6，10^-8 由该裁判不可达——这是两套断言能同处一个测试的物理依据。"
    ),
    "test_parity.png": (
        "测试对等性：全域 passed/failed/skipped 的 main 基线 vs 合并树对照，"
        "以及受影响面 4 个模块的逐模块对照（合并树不少于基线）。"
    ),
}


def png_base64(path: Path, budget: int = 60_000) -> tuple[str, int]:
    """Downscale/re-encode until the base64 payload is under ``budget`` bytes."""
    raw = path.read_bytes()
    for scale, quality in ((1.0, 90), (0.85, 85), (0.7, 80), (0.55, 75), (0.45, 70)):
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        if scale < 1.0:
            img = img.resize(
                (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                Image.LANCZOS,
            )
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        data = buf.getvalue()
        if len(data) * 4 / 3 <= budget:
            return base64.b64encode(data).decode("ascii"), len(data)
    return base64.b64encode(data).decode("ascii"), len(data)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission-id", type=int, default=3984)
    ap.add_argument("--scenario-key", default="tensorbem-pr109-merge-healing")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--out", default="/tmp/report.json")
    args = ap.parse_args(argv)

    checks = json.loads((ROOT / "docs/m2966/outputs/merge_checks/merge_checks.json").read_text())
    audit = json.loads((ROOT / "docs/m3984/outputs/merge_union_check.json").read_text())
    figs = json.loads((ROOT / "docs/m3984/outputs/figure_summary.json").read_text())

    figures = []
    for name, caption in CAPTIONS.items():
        path = FOUT / name
        if not path.exists():
            continue
        b64, size = png_base64(path)
        figures.append({"name": name, "caption": caption, "png_base64": b64})
        print(f"{name}: {size} bytes raw, payload ok")

    metrics = {
        "merge_audit_criteria_passed": sum(1 for c in audit["checks"] if c["ok"]),
        "merge_audit_criteria_total": len(audit["checks"]),
        "merge_audit_ok": audit["ok"],
        "mfs_rows": [
            {
                "panels": r["panels"],
                "sources": r["sources"],
                "added_mass_rel_error": r["added_mass_rel_error"],
                "damping_rel_error": r["damping_rel_error"],
                "doc_gap_max": max(r["added_mass_doc_gap"], r["damping_doc_gap"]),
            }
            for r in checks["mfs_rows"]["rows"]
        ],
        "worst_relative_error_vs_ursell": figs["mfs_rows"]["worst_relative_error"],
        "panel_referee": {
            "observed_order": checks["panel_referee"]["headline_observed_order"],
            "error_at_400_pieces": figs["referee"]["error_at_400_pieces"],
            "single_panel_error": checks["panel_referee"]["single_panel_error"],
        },
        "tests": {
            "main_baseline": {k: v for k, v in figs["tests"]["main"].items() if k != "log"},
            "merged": {k: v for k, v in figs["tests"]["merged"].items() if k != "log"},
            "affected_surface": figs["tests"]["subsets"],
        },
    }

    payload = {
        "title": "PR #109 治愈：TensorBEM agent/m2966 × main 三方合并、静默丢内容审计与复测",
        "summary": (
            "PR #109（agent/m2966）对 main 变脏后，按语义并集逐文件解冲：MFS 表保留 main 的生成式标签"
            f"与本分支 256 面元第三行，裁判测试同时保留双方断言；两处跨侧接缝（doc_claims 行选择、"
            f"文档守卫标签）显式声明并有测试覆盖。9 条机械审计判据全过，受影响面测试不低于 main 基线。"
        ),
        "scenario_key": args.scenario_key,
        "mission_id": args.mission_id,
        "tags": ["TensorBEM", "PR#109", "merge-healing", "union-merge", "ursell", "mfs",
                 "silent-loss-audit"],
        "markdown": DOC.read_text(),
        "figures": figures,
        "metrics": metrics,
    }
    blob = json.dumps(payload, ensure_ascii=False)
    Path(args.out).write_text(blob)
    print(f"payload {len(blob.encode())} bytes -> {args.out}")

    if not args.upload:
        return 0

    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    s = requests.Session()
    cred = {"username": os.environ["TENSOR_AGENT_USER"],
            "password": os.environ["TENSOR_AGENT_PASS"]}
    r = s.post(base + "/auth/login", json=cred, timeout=60)
    if r.status_code != 200:
        r = s.post(base + "/tensor/auth/login", json=cred, timeout=60)
    r.raise_for_status()
    body = r.json()
    tok = (body.get("data") or body).get("access_token") or body.get("token")
    resp = s.post(f"{base}/tensor/reports", json=payload,
                  headers={"Authorization": f"Bearer {tok}"}, timeout=300)
    print("POST /tensor/reports ->", resp.status_code, resp.text[:400])
    return 0 if resp.status_code in (200, 201) else 1


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))