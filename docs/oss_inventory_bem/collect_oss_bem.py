#!/usr/bin/env python3
"""FEAT-OSS-INVENTORY-BEM wave 1 — live GitHub census for boundary-element /
potential-flow hydrodynamics code.

Why: the feature ledger anchor FEAT-OSS-INVENTORY-BEM asks for
    >=20 repos, registry JSON, tier grading, GAP-BEM-* anchor proposals.
Nothing may be hand-copied: every repo record carries the live API payload that
produced it (stars / license / pushed_at / archived), and raw responses are kept
under raw/ so the candidate set can be rebuilt offline.

Discovery uses two independent channels so the census is not a popularity list:
  A. keyword search (name/description/topics), several discovery axes;
  B. seed lookups for canonical solvers that keyword search misses because their
     description uses jargon (NEMOH, capytaine, HAMS, BEMIO ...). A seed only
     enters the registry if the API confirms it -- identical evidence standard.

Rate-limit discipline (unauthenticated): search 10/min, core 60/h.
Run: python docs/oss_inventory_bem/collect_oss_bem.py
"""
from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
OUT = os.path.join(HERE, "candidates.json")

SEARCH_API = "https://api.github.com/search/repositories"
REPO_API = "https://api.github.com/repos"

# Channel A: discovery axes (solver stacks, method families, application domains)
QUERIES = [
    "boundary element method hydrodynamics",
    "boundary element",
    "potential flow hydrodynamics",
    "panel method",
    "seakeeping",
    "ship hydrodynamics",
    "wave energy converter",
    "boundary element method",
    "capytaine",
    "nemoh",
]

# Channel B: canonical solvers / components a domain expert would look for.
# Verified against the API; unverifiable names are reported, never invented.
SEEDS = [
    "capytaine/capytaine",
    "WEC-Sim/WEC-Sim",
    "WEC-Sim/BEMIO",
    "bempp/bempp-cl",
    "exafmm/exafmm-t",
    "OpenMDAO/OpenMDAO",
    "LHEEA/NEMOH",
    "Nemoh/Nemoh",
    "mcodevmill/NEMOH",
    "YumingLiu/HAMS",
    "pmgbergen/BEM",
    "pmgbergen/bempp",
]


def _get(url: str, tries: int = 3) -> dict:
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "tensorbem-oss-census/1.0",
    })
    for _ in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                print(f"  rate limited, sleeping 65s ({url[:60]})")
                time.sleep(65)
                continue
            if e.code == 404:
                return {"__missing__": True, "status": 404}
            raise
        except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
            print(f"  transient network error ({type(e).__name__}), retrying in 8s")
            time.sleep(8)
    raise RuntimeError(f"unrecoverable failure for {url}")


def record(it: dict, source: str) -> dict:
    return {
        "full_name": it["full_name"],
        "html_url": it["html_url"],
        "description": it.get("description"),
        "stars": it["stargazers_count"],
        "forks": it["forks_count"],
        "language": it.get("language"),
        "license": (it.get("license") or {}).get("spdx_id"),
        "pushed_at": it.get("pushed_at"),
        "created_at": it.get("created_at"),
        "archived": it.get("archived", False),
        "topics": it.get("topics", []),
        "discovered_by": [source],
    }


def main() -> None:
    os.makedirs(RAW, exist_ok=True)
    seen: dict[str, dict] = {}
    order: list[str] = []

    def add(rec: dict) -> None:
        if rec["full_name"] in seen:
            seen[rec["full_name"]]["discovered_by"].extend(rec["discovered_by"])
        else:
            seen[rec["full_name"]] = rec
            order.append(rec["full_name"])

    for i, q in enumerate(QUERIES):
        url = (f"{SEARCH_API}?q={urllib.parse.quote(q)}"
               f"&sort=stars&order=desc&per_page=50")
        cache = os.path.join(RAW, f"search_{i:02d}.json")
        if os.path.exists(cache):
            payload = json.load(open(cache))["payload"]
            print(f"[search {i}] {q!r}: cached")
            for it in payload.get("items", []):
                add(record(it, f"search:{q}"))
            continue
        payload = _get(url)
        with open(cache, "w") as f:
            json.dump({"query": q, "payload": payload}, f, indent=1)
        items = payload.get("items", [])
        print(f"[search {i}] {q!r}: {len(items)} hits (total {payload.get('total_count')})")
        for it in items:
            add(record(it, f"search:{q}"))
        if i + 1 < len(QUERIES):
            time.sleep(7)

    missing = []
    for s in SEEDS:
        payload = _get(f"{REPO_API}/{s}")
        if payload.get("__missing__"):
            print(f"[seed] {s}: 404 (not verified -> excluded)")
            missing.append(s)
            continue
        print(f"[seed] {s}: {payload['stargazers_count']} stars, pushed {payload.get('pushed_at')}")
        add(record(payload, "seed"))
        with open(os.path.join(RAW, f"seed_{s.replace('/', '__')}.json"), "w") as f:
            json.dump(payload, f, indent=1)

    cands = [seen[k] for k in order]
    cands.sort(key=lambda r: -r["stars"])
    with open(OUT, "w") as f:
        json.dump({
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "channels": {"search_queries": QUERIES, "seeds_verified": SEEDS,
                         "seeds_missing": missing},
            "n_unique": len(cands),
            "candidates": cands,
        }, f, indent=1)
    print(f"unique repos: {len(cands)} -> {OUT}")


if __name__ == "__main__":
    main()