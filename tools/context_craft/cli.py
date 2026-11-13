"""CLI for the context-craft toolkit.

    python -m tools.context_craft.cli extract --root . --pattern 'docs/*.md' --out items.json
    python -m tools.context_craft.cli pack --root . --pattern 'docs/*.md' \
        --query "tolerance" --budget 800 --profile verification --out brief
    python -m tools.context_craft.cli verify --brief brief.json --root .
    python -m tools.context_craft.cli memory --memory mem.json --mission-index 7 --out-dir mem/
    python -m tools.context_craft.cli ab --root . --pattern 'docs/dev_report_*.md' --out ab/
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .extract import extract_corpus
from .memory import compress, load_memory, save_memory, verify_forget_log
from .model import FactItem, sha256_text
from .pack import pack


def _load_items(root: str, patterns: list[str]) -> list[FactItem]:
    return extract_corpus(root, patterns)


def cmd_extract(a) -> int:
    items = _load_items(a.root, a.pattern)
    payload = [i.to_dict() for i in items]
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
    kinds: dict[str, int] = {}
    for i in items:
        kinds[i.kind] = kinds.get(i.kind, 0) + 1
    print(json.dumps({"items": len(items), "kinds": kinds, "out": a.out}, ensure_ascii=False))
    return 0


def cmd_pack(a) -> int:
    items = _load_items(a.root, a.pattern)
    pkg = pack(items, a.query, a.budget, a.profile)
    base = a.out[:-3] if a.out.endswith(".md") else a.out
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(pkg.to_dict(), fh, ensure_ascii=False, indent=2, sort_keys=True)
    with open(base + ".md", "w", encoding="utf-8") as fh:
        fh.write(pkg.to_markdown())
    print(json.dumps({"tokens_used": pkg.tokens_used, "items": len(pkg.items),
                      "package_sha256": pkg.package_sha256,
                      "json": base + ".json", "markdown": base + ".md"}, ensure_ascii=False))
    return 0


def cmd_verify(a) -> int:
    """Re-hash every carried line against the live tree (zero-hand-copy gate)."""
    with open(a.brief, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    bad = []
    for prov in data["provenance"]:
        path = os.path.join(a.root, prov["source"])
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
            actual = lines[prov["lineno"] - 1].strip()
        except (OSError, IndexError):
            bad.append((prov, "missing line"))
            continue
        # classification cleans list/quote/table markers -> compare on content
        if sha256_text(actual) != prov["sha256"] and prov["sha256"] not in sha256_text(actual):
            from .extract import _clean

            if sha256_text(_clean(actual)) != prov["sha256"]:
                bad.append((prov, "hash mismatch"))
    ok = len(data["provenance"]) - len(bad)
    print(json.dumps({"checked": len(data["provenance"]), "ok": ok, "bad": len(bad),
                      "problems": [b[1] + " @ " + b[0]["source"] for b in bad][:10]},
                     ensure_ascii=False))
    return 0 if not bad else 1


def cmd_memory(a) -> int:
    entries = load_memory(a.memory)
    res = compress(entries, a.mission_index, a.out_dir, force=a.force)
    os.makedirs(a.out_dir, exist_ok=True)
    kept = [e for lst in res.kept.values() for e in lst]
    save_memory(kept, os.path.join(a.out_dir, "memory.json"))
    with open(os.path.join(a.out_dir, "compress_report.json"), "w", encoding="utf-8") as fh:
        json.dump(res.to_dict(), fh, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps({"triggered": res.triggered, "forgotten": len(res.forgotten),
                      "kept_tokens": res.kept_tokens, "overflow": res.overflow_tokens,
                      "forget_log": res.log_path}, ensure_ascii=False))
    if res.forgotten:
        print(json.dumps(verify_forget_log(res.log_path, os.path.join(a.out_dir, "archive")),
                         ensure_ascii=False))
    return 0


def cmd_ab(a) -> int:
    from .ab import run, to_markdown

    os.makedirs(a.out_dir, exist_ok=True)
    res = run(a.root, a.pattern, a.budgets, a.limit, out_path=os.path.join(a.out_dir, "ab_results.json"))
    with open(os.path.join(a.out_dir, "ab_results.md"), "w", encoding="utf-8") as fh:
        fh.write(to_markdown(res))
    print(to_markdown(res))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="context_craft", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract")
    e.add_argument("--root", default=".")
    e.add_argument("--pattern", nargs="+", default=["docs/*.md"])
    e.add_argument("--out", default="items.json")
    e.set_defaults(fn=cmd_extract)

    k = sub.add_parser("pack")
    k.add_argument("--root", default=".")
    k.add_argument("--pattern", nargs="+", default=["docs/*.md"])
    k.add_argument("--query", required=True)
    k.add_argument("--budget", type=int, default=800)
    k.add_argument("--profile", default="generic")
    k.add_argument("--out", default="brief")
    k.set_defaults(fn=cmd_pack)

    v = sub.add_parser("verify")
    v.add_argument("--brief", required=True)
    v.add_argument("--root", default=".")
    v.set_defaults(fn=cmd_verify)

    m = sub.add_parser("memory")
    m.add_argument("--memory", required=True)
    m.add_argument("--mission-index", type=int, required=True)
    m.add_argument("--out-dir", default="memory_out")
    m.add_argument("--force", action="store_true")
    m.set_defaults(fn=cmd_memory)

    b = sub.add_parser("ab")
    b.add_argument("--root", default=".")
    b.add_argument("--pattern", nargs="+", default=["docs/dev_report_*.md"])
    b.add_argument("--budgets", nargs="+", type=int, default=[400, 800, 1600])
    b.add_argument("--limit", type=int, default=40)
    b.add_argument("--out-dir", default="ab")
    b.set_defaults(fn=cmd_ab)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())