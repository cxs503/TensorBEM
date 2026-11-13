"""开工三查 probe: knowledge_base / negative_knowledge_ledger / oss_solutions for this case."""
from __future__ import annotations

import json
import os
import pathlib
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
OUT = pathlib.Path(__file__).resolve().parent.parent / "artifacts" / "cummins_wigley_kernel" / "kb_probe.json"


def post(path: str, payload: dict, token: str | None = None) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})},
    )
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def get(path: str, token: str) -> dict:
    req = urllib.request.Request(BASE + path, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def main() -> None:
    tok = post("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                               "password": os.environ["TENSOR_AGENT_PASS"]})
    token = tok.get("access_token") or tok.get("token")
    spec = get("/openapi.json", token)
    paths = sorted(spec.get("paths", {}))
    hits = [p for p in paths if any(k in p for k in ("knowledge", "negative", "oss"))]
    result = {"paths": hits}
    for q in ("cummin", "kramers", "TensorBEM", "wigley"):
        for p in hits:
            if "{id}" in p or "{" in p or "search" not in p:
                continue
            try:
                result.setdefault("queries", {})[f"{p}?q={q}"] = get(f"{p}?q={q}&limit=5", token)
            except Exception as exc:  # noqa: BLE001
                result.setdefault("queries", {})[f"{p}?q={q}"] = f"ERR {exc}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps(result, indent=1, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()