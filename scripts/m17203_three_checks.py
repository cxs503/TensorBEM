"""m17203 开工三查（v2 正确端点）：knowledge / negative-knowledge / capabilities audit。

三查 = ①knowledge_base 同域 capability ②negative_knowledge_ledger keywords='TensorBEM'
③oss_solutions 同域（本平台以 knowledge assets / capability audit 承载）。
结果原样落盘 artifacts/m17203_dean_diffraction/three_checks.json（零手抄）。
"""
from __future__ import annotations

import json
import os
import pathlib
import urllib.parse
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
if not BASE.endswith("/api/v1"):
    BASE = BASE.rstrip("/") + "/api/v1"
OUT = pathlib.Path(__file__).resolve().parent.parent / "artifacts/m17203_dean_diffraction"
OUT.mkdir(parents=True, exist_ok=True)


def post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def get(path, token):
    req = urllib.request.Request(BASE + path, headers={"Authorization": f"Bearer {token}"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


token = None
try:
    tok = post("/auth/login", {"username": os.environ.get("TENSOR_AGENT_USER", "agent"),
                               "password": os.environ.get("TENSOR_AGENT_PASS", "")})
    token = tok.get("access_token") or tok.get("token")
except Exception as exc:
    print("login failed:", exc)

q = urllib.parse.quote
targets = {
    "knowledge_search_TensorBEM": f"/tensor/knowledge/search?q={q('TensorBEM diffraction')}&limit=20",
    "knowledge_search_dean": f"/tensor/knowledge/search?q={q('Dean cylinder diffraction')}&limit=20",
    "negative_knowledge_search": f"/tensor/negative-knowledge/search?q={q('TensorBEM')}&limit=50",
    "capabilities_audit_summary": "/tensor/capabilities/audit/summary",
    "knowledge_overview": "/tensor/knowledge/overview",
}

result = {}
for name, path in targets.items():
    try:
        result[name] = get(path, token)
    except Exception as exc:
        result[name] = {"error": str(exc)}

(OUT / "three_checks.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
for name, payload in result.items():
    if isinstance(payload, dict) and "error" in payload:
        print("==", name, "ERROR:", payload["error"][:160])
        continue
    print("==", name, "keys:", list(payload)[:8] if isinstance(payload, dict) else type(payload))
    blob = json.dumps(payload, ensure_ascii=False)
    hits = [k for k in ("dean", "Dean", "haskind", "Haskind", "ursell", "Ursell", "diffraction", "绕射")]
    print("   contains:", {h: blob.count(h) for h in hits if blob.count(h)})
    print("   head:", blob[:400])