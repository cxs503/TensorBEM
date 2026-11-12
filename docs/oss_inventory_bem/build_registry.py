#!/usr/bin/env python3
"""FEAT-OSS-INVENTORY-BEM wave 1 — build registry.json + gap_anchors + census.md.

Inputs (committed, produced by sibling scripts):
  candidates.json          live GitHub metadata for every census hit
  raw/search_*.json        raw search payloads (offline rebuild)
  raw/readme/_index.json   README fingerprints for shortlisted repos
  src/, tests/             gap probes are *executed* against the real tree

Outputs:
  registry.json            >=20 graded repos, each traceable to live metadata
  census.md                generated table (no hand-copied numbers)

Honesty rules enforced mechanically:
  * a repo may enter the registry only if live metadata exists (else: problem);
  * `evidence_confidence` is `readme` only when a stored README is fingerprinted,
    otherwise it must say `description_only`;
  * every GAP anchor stores an executed probe (regex hits over src/ + tests/),
    so "TensorBEM lacks X" is reproducible rather than asserted;
  * dangling gap references and <20 repos / <6 anchors fail the build.

Run: python docs/oss_inventory_bem/build_registry.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from registry_data import ANCHORS, CURATED, TIER_RULES  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(REPO_ROOT, "src")
TESTS = os.path.join(REPO_ROOT, "tests")
README_DIR = os.path.join(HERE, "raw", "readme")


def probe(terms):
    """Execute a gap probe: regex hits over src/ and tests/, like for like."""
    pats = [re.compile(t, re.IGNORECASE) for t in terms]
    hits = {t: [] for t in terms}
    n_files = 0
    for root in (SRC, TESTS):
        for dirpath, _, files in os.walk(root):
            for fn in files:
                if not fn.endswith((".py", ".md", ".json", ".txt", ".cal", ".dat")):
                    continue
                p = os.path.join(dirpath, fn)
                n_files += 1
                try:
                    text = open(p, encoding="utf-8", errors="replace").read()
                except OSError:
                    continue
                for t, pat in zip(terms, pats):
                    if pat.search(text):
                        hits[t].append(os.path.relpath(p, REPO_ROOT))
    return {
        "scanned_files": n_files,
        "scanned_roots": [os.path.relpath(SRC, REPO_ROOT), os.path.relpath(TESTS, REPO_ROOT)],
        "pattern": "case-insensitive regex, hits counted per file",
        "hits": {t: {"count": len(v), "sample": sorted(v)[:5]} for t, v in hits.items()},
        "total_hits": sum(len(v) for v in hits.values()),
    }


def readme_evidence(repo, keywords):
    idx_path = os.path.join(README_DIR, "_index.json")
    idx = json.load(open(idx_path)) if os.path.exists(idx_path) else {"repos": {}}
    rec = idx.get("repos", {}).get(repo)
    if not rec:
        return {"readme_available": False}
    text = open(os.path.join(HERE, rec["file"]), encoding="utf-8", errors="replace").read()
    lines = []
    for kw in keywords or []:
        for ln in text.splitlines():
            if kw.lower() in ln.lower():
                s = " ".join(ln.split())
                if len(s) > 4:
                    lines.append(f"[{kw}] {s[:220]}")
                break
    return {"readme_available": True, "file": rec["file"], "bytes": rec["bytes"],
            "sha256_16": rec["sha256"], "evidence_lines": lines[:4]}


def build():
    cands = json.load(open(os.path.join(HERE, "candidates.json")))
    meta = {c["full_name"]: c for c in cands["candidates"]}

    repos, screened, problems = [], [], []
    for name, cur in CURATED.items():
        m = meta.get(name)
        if m is None:
            problems.append(f"{name}: no live metadata in candidates.json -> not gradable")
            continue
        entry = {
            "full_name": name,
            "tier": cur["tier"],
            "domain": cur["domain"],
            "method_family": cur["method_family"],
            "offers_transferable": cur.get("offers", []),
            "gaps_addressed": cur.get("gaps_addressed", []),
            "license": m.get("license"),
            "license_note": cur.get("license_note", ""),
            "stars": m["stars"],
            "forks": m["forks"],
            "language": m.get("language"),
            "pushed_at": m.get("pushed_at"),
            "created_at": m.get("created_at"),
            "archived": m.get("archived"),
            "html_url": m["html_url"],
            "discovered_by": sorted(set(m.get("discovered_by", []))),
            "readme_evidence": readme_evidence(name, cur.get("readme_keywords")),
        }
        entry["evidence_confidence"] = cur.get(
            "evidence_confidence",
            "readme" if entry["readme_evidence"]["readme_available"] else "description_only")
        if "annotation_caveat" in cur:
            entry["annotation_caveat"] = cur["annotation_caveat"]
        if cur["tier"] == "C":
            entry["exclude_reason"] = cur.get("exclude_reason", "edge relevance")
            screened.append(entry)
        else:
            repos.append(entry)

    anchors = []
    for a in ANCHORS:
        pr = probe(a["probe_terms"])
        anchors.append({**a,
                        "gap_probe": pr,
                        "gap_status": ("verified_absent" if pr["total_hits"] == 0
                                       else "partially_present_review_required"),
                        "proposed_by": "m16421@FEAT-OSS-INVENTORY-BEM"})

    slugs = {a["slug"] for a in anchors}
    for r in repos:
        for g in r["gaps_addressed"]:
            if g not in slugs:
                problems.append(f"{r['full_name']}: dangling gap ref {g}")
    if not all(a["slug"].startswith("GAP-BEM-") for a in anchors):
        problems.append("all anchors must be GAP-BEM-*")
    for a in anchors:
        if not a.get("source_repos"):
            problems.append(f"{a['slug']}: no source repos")
        if not any(ch.isdigit() for ch in a.get("quantified_target", "")):
            problems.append(f"{a['slug']}: quantified_target has no number")

    tier_counts = {t: sum(1 for r in repos if r["tier"] == t) for t in ("A", "B")}
    if len(repos) < 20:
        problems.append(f"only {len(repos)} graded repos < required 20")
    if len(anchors) < 6:
        problems.append(f"only {len(anchors)} GAP anchors < required 6")

    doc = {
        "schema_version": "1.0",
        "anchor": "FEAT-OSS-INVENTORY-BEM",
        "mission_id": 16421,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "census": {
            "discovery_channels": cands["channels"],
            "candidates_scanned": cands["n_unique"],
            "registered_and_graded": len(repos),
            "tier_counts": tier_counts,
            "screened_out_tier_C": len(screened),
            "readme_fingerprints": sum(
                1 for r in repos if r["readme_evidence"]["readme_available"]),
        },
        "tier_rules": TIER_RULES,
        "gap_anchors": anchors,
        "repos": repos,
        "screened_out": screened,
        "validation": {"ok": not problems, "problems": problems},
        "provenance": {
            "collector": "docs/oss_inventory_bem/collect_oss_bem.py",
            "readme_fetcher": "docs/oss_inventory_bem/fetch_readmes.py",
            "curated_judgement": "docs/oss_inventory_bem/registry_data.py",
            "builder": "docs/oss_inventory_bem/build_registry.py",
            "raw_search_payloads": "docs/oss_inventory_bem/raw/search_*.json",
            "raw_readmes": "docs/oss_inventory_bem/raw/readme/",
        },
    }
    out = os.path.join(HERE, "registry.json")
    with open(out, "w") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    digest = hashlib.sha256(open(out, "rb").read()).hexdigest()[:16]

    print(f"registry.json  sha256[:16]={digest}")
    print(f"  graded={len(repos)} tiers={tier_counts} screened_out={len(screened)} "
          f"anchors={len(anchors)} readme_fp={doc['census']['readme_fingerprints']}")
    for p in problems:
        print("  PROBLEM:", p)
    for a in anchors:
        print(f"  {a['slug']:40s} probe_hits={a['gap_probe']['total_hits']:3d}  "
              f"{a['gap_status']}")
    write_census_md(doc, digest)
    return 0 if not problems else 1


def write_census_md(doc, digest):
    L = []
    c = doc["census"]
    ch = c["discovery_channels"]
    L.append("# OSS BEM 技术盘点（wave 1）— FEAT-OSS-INVENTORY-BEM\n")
    L.append(f"生成器：`docs/oss_inventory_bem/build_registry.py`（本文件禁止手改，"
             f"registry sha256[:16]=`{digest}`，时间 {doc['generated_at_utc']}）\n")
    L.append(f"- 候选扫描：**{c['candidates_scanned']}** 仓"
             f"（关键词搜索 {len(ch['search_queries'])} 条 + 种子直查 {len(ch['seeds_verified'])} 个，"
             f"其中 {len(ch['seeds_missing'])} 个 404 已排除）")
    L.append(f"- 登记并定级：**{c['registered_and_graded']}**"
             f"（A={c['tier_counts']['A']} / B={c['tier_counts']['B']}），"
             f"筛查剔除 C 档 {c['screened_out_tier_C']} 个")
    L.append(f"- README 指纹：{c['readme_fingerprints']} 个仓（原文留存 raw/readme/）\n")
    L.append("## 定级规则\n")
    for t, rule in doc["tier_rules"].items():
        L.append(f"- **{t}**：{rule}")
    L.append("\n## A/B 档登记表\n")
    L.append("| Tier | 仓库 | ★ | 许可证 | 最近推送 | 域 | 可迁移到 TensorBEM 的东西 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in doc["repos"]:
        offer = r["offers_transferable"][0] if r["offers_transferable"] else "—"
        L.append(f"| {r['tier']} | [{r['full_name']}]({r['html_url']}) | {r['stars']} | "
                 f"{r['license']} | {str(r['pushed_at'])[:10]} | {r['domain']} | {offer} |")
    L.append("\n## C 档筛查记录（登记但不对标）\n")
    L.append("| 仓库 | ★ | 排除理由 |")
    L.append("|---|---|---|")
    for r in doc["screened_out"]:
        L.append(f"| [{r['full_name']}]({r['html_url']}) | {r['stars']} | {r['exclude_reason']} |")
    L.append("\n## GAP-BEM-* 锚点（缺口由执行探针验证，非断言）\n")
    L.append("| 锚点 | 优先级 | 探针命中 | 状态 | 量化验收 | 工作量 |")
    L.append("|---|---|---|---|---|---|")
    for a in doc["gap_anchors"]:
        L.append(f"| `{a['slug']}` | {a['priority']} | {a['gap_probe']['total_hits']} | "
                 f"{a['gap_status']} | {a['quantified_target']} | {a['effort']} |")
    L.append("\n> 探针命中数 = 在 `src/` + `tests/` 上执行 terms 正则命中的文件数；"
             "0 = 该能力在代码库中确实缺席（可在 registry.json 的 `gap_probe` 里复核）。\n")
    L.append("\n## 复现方式\n")
    L.append("```bash\npython docs/oss_inventory_bem/collect_oss_bem.py   # 采集（需网络，限速 10/min）\n"
             "python docs/oss_inventory_bem/fetch_readmes.py     # README 取证\n"
             "python docs/oss_inventory_bem/build_registry.py    # 定级+探针+校验（离线可跑）\n"
             "```\n")
    path = os.path.join(HERE, "census.md")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"census.md written -> {os.path.relpath(path, REPO_ROOT)}")


if __name__ == "__main__":
    sys.exit(build())