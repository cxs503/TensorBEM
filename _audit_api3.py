import os, json, urllib.request

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
tok = open("/tmp/tok.txt").read().strip()

def get(path):
    r = urllib.request.Request(base + path)
    r.add_header("Authorization", "Bearer " + tok)
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")
    except Exception as e:
        return -1, {"err": str(e)}

s, d = get("/openapi.json")
print("openapi", s)
if s == 200:
    paths = d.get("paths", {})
    for p in sorted(paths):
        if any(k in p.lower() for k in ["benchmark", "case", "report", "maturity", "verdict", "mission", "anchor"]):
            print(p, list(paths[p].keys()))
else:
    # try docs
    for p in ["/docs", "/api/v1/openapi.json"]:
        s2, d2 = get(p)
        print(p, s2, str(d2)[:200])
