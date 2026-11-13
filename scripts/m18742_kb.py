"""Knowledge-base probes for m18742 (negative ledger + capability + knowledge search)."""
import json
import os
import sys
import urllib.parse
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
U = os.environ["TENSOR_AGENT_USER"]
P = os.environ["TENSOR_AGENT_PASS"]


def req(path, data=None, tok=None):
    url = BASE + path
    body = json.dumps(data).encode() if data is not None else None
    r = urllib.request.Request(url, data=body, method="POST" if data else "GET")
    r.add_header("Content-Type", "application/json")
    if tok:
        r.add_header("Authorization", "Bearer " + tok)
    try:
        with urllib.request.urlopen(r, timeout=90) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def main():
    tok = req("/auth/login", {"username": U, "password": P})["access_token"]
    qs = [
        "/tensor/negative-knowledge/search?keywords=" + urllib.parse.quote("TensorBEM"),
        "/tensor/negative-knowledge/search?keywords=" + urllib.parse.quote("XVAL-FVM"),
        "/tensor/knowledge/search?q=" + urllib.parse.quote("XVAL-FVM-THREE-GRID"),
        "/tensor/capabilities/audit?domain=TensorBEM",
    ]
    out = {}
    for p in qs:
        out[p] = req(p, tok=tok)
        print("\n===", p)
        print(json.dumps(out[p], ensure_ascii=False)[:3000])
    with open("/tmp/m18742_kb.json", "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.exit(main())