"""Small API helper for mission 8641 (read-only queries)."""
import json
import os
import sys
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
TOKEN = open("/tmp/m8641_token").read().strip()


def get(path):
    req = urllib.request.Request(BASE + path, headers={"Authorization": "Bearer " + TOKEN})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        status, data = get(p)
        print("==", p, status)
        print(json.dumps(data, ensure_ascii=False, indent=1)[:6000])
