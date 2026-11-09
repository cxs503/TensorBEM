"""Shard runner: evaluate a subset of (ka) for one (level, config) pair."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from tensorbem.benchmarks import interior_panel_suppression as study

parser = argparse.ArgumentParser()
parser.add_argument("--level", type=int, required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--shards", type=int, default=1)
parser.add_argument("--grid", choices=("main", "full", "fine", "custom"), default="full")
parser.add_argument("--kas", nargs="*", type=float, default=None)
args = parser.parse_args()

if args.grid == "main":
    grid = list(study.KA_MAIN)
elif args.grid == "fine":
    grid = list(study.KA_FINE)
elif args.grid == "custom":
    grid = [float(k) for k in (args.kas or [])]
else:
    grid = list(study.ka_grid(args.level))

mine = [ka for i, ka in enumerate(grid) if i % args.shards == args.shard]
out = Path("runs/interior_panel_suppression") / f"shard_L{args.level}_{args.config}_{args.shard}of{args.shards}.json"
t0 = time.time()
rows = [study.hemisphere_point(args.level, ka, args.config) for ka in mine]
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"level": args.level, "config": args.config,
                           "shard": args.shard, "shards": args.shards,
                           "cases": rows, "wall_seconds": time.time() - t0}, indent=1) + "\n")
print(f"[ok] {out} n={len(rows)} {time.time()-t0:.1f}s", flush=True)
