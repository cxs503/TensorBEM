#!/usr/bin/env bash
# One-click reproduction of CASE-BEM-SPHERE-ADDED-MASS (m17262, rework of m17176).
# Regenerates every artifact under runs/m17262/ from the production entry.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHONPATH=src python3 runs/m17262/make_artifacts.py
