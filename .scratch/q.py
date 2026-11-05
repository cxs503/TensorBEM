import json
import os
import urllib.request
import sys
base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
tok = json.load(open("/tmp/login.json"))["access_token"]
def get(path):
    req = urllib.request.Request(base + path, headers={"Authorization": "Bearer " + tok})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)
if __name__ == "__main__":
    data = get(sys.argv[1])
    print(json.dumps(data, ensure_ascii=False, indent=1)[: int(sys.argv[2]) if len(sys.argv) > 2 else 4000])
