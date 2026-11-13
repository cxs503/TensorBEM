#!/usr/bin/env python3
"""Simulate maturity_gate.judge() on the m17300 verdict evidence (pre-INSERT check)."""
import json
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, "/root/tp-agent")
# Reuse the gate's own functions for fidelity
import importlib.util
spec = importlib.util.spec_from_file_location("maturity_gate", "/root/tp-agent/maturity_gate.py")
mg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mg)

EVIDENCE = json.load(open("/root/tp-agent/work/m17300/tools/verdict_evidence_m17300.json"))

gl, why = mg.judge(4, EVIDENCE, "TensorBEM", 17300)
print("gate_level:", gl)
print("why:", why)
assert gl == 4, f"expected L4, got L{gl}: {why}"
print("SIMULATION PASS: gate would confirm at L4")
