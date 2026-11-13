#!/bin/sh
set -e
PYTHONPATH=src python3 -m tensorbem.benchmarks.case_ellipse_2d_added_mass --output benchmarks/verified/case_bem_ellipse_2d_added_mass --doc docs/CASE-BEM-ELLIPSE-2D-ADDED-MASS.md
