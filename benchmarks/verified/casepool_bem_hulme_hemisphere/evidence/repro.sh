#!/bin/sh
# one-shot reproduction of this evidence bundle
set -e
PYTHONPATH=src python3 -m tensorbem.benchmarks.hulme_hemisphere_case --out benchmarks/verified/casepool_bem_hulme_hemisphere/evidence --shard 0 --shards 1
PYTHONPATH=src python3 -m tensorbem.benchmarks.hulme_hemisphere_case --out benchmarks/verified/casepool_bem_hulme_hemisphere/evidence --merge
