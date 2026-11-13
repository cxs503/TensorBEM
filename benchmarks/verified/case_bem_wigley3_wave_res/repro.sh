#!/bin/sh
set -e
PYTHONPATH=src python3 -m tensorbem.benchmarks.case_wigley3_wave_resistance --output benchmarks/verified/case_bem_wigley3_wave_res
