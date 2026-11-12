#!/usr/bin/env python3
"""Fetch READMEs for shortlisted OSS BEM repos so that every registry
capability claim can be traced to source text instead of memory.

Writes raw/readme/<owner>__<repo>.md and raw/readme/_index.json
(sha256 + byte count per file, so the report can quote a fingerprint).
Run: python docs/oss_inventory_bem/fetch_readmes.py
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "raw", "readme")

REPOS = [
    "capytaine/capytaine", "BEMRosetta/BEMRosetta", "LHEEA/Nemoh", "WEC-Sim/bemio",
    "JuliaOceanWaves/MarineHydro.jl", "PengYunqiang/CRESTU",
    "Xu-Duan/Potential-Flow-With-Complex-Green-Functions",
    "LukeMcCulloch/adjoint-3D-linear-wave-resistance", "bempp/bempp-cl",
    "ZeppSav/BEMUse", "Savin-Viswanathan/OMHyD-PA", "JuliaOceanWaves/Hydrodynamics.jl",
    "shash-sharma/OpenBEM", "AzadSaeed/Parallel-Nemoh", "mathLab/pi-BEM",
    "exafmm/exafmm-t",
]
# raw.githubusercontent.com is unreachable from this host (TCP reset); the REST
# readme endpoint returns the same bytes and is reachable, so use it instead.
API = "https://api.github.com/repos/{repo}/readme"


def get(url: str, tries: int = 4) -> str | None:
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "tensorbem-oss-census/1.0",
                "Accept": "application/vnd.github.raw",
            })
            with urllib.request.urlopen(req, timeout=30) as f:
                return f.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(4 * (k + 1))
        except Exception:  # connection reset / timeout -- transient on this host
            time.sleep(4 * (k + 1))
    return None


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    index, missing = {}, []
    for r in REPOS:
        text = get(API.format(repo=r))
        if not text:
            missing.append(r)
            continue
        path = os.path.join(OUT, r.replace("/", "__") + ".md")
        with open(path, "w") as f:
            f.write(text)
        index[r] = {"file": os.path.relpath(path, HERE), "bytes": len(text),
                    "sha256": hashlib.sha256(text.encode()).hexdigest()[:16]}
        print(f"  {r:58s} {len(text):7d} B")
    with open(os.path.join(OUT, "_index.json"), "w") as f:
        json.dump({"repos": index, "missing": missing}, f, indent=1)
    print(f"fetched {len(index)}/{len(REPOS)}; missing={missing}")


if __name__ == "__main__":
    main()