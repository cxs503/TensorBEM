import os, json, urllib.request

base = os.environ["TENSOR_API_BASE"]
user = os.environ["TENSOR_AGENT_USER"]
pw = os.environ["TENSOR_AGENT_PASS"]

def req(method, path, body=None, tok=None):
    url = base + path
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Content-Type", "application/json")
    if tok:
        r.add_header("Authorization", "Bearer " + tok)
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")
    except Exception as e:
        return -1, {"err": str(e)}

s, d = req("POST", "/auth/login", {"username": user, "password": pw})
print("login", s, list(d.keys())[:6])
tok = d.get("access_token") or d.get("token") or ""
open("/tmp/tok.txt", "w").write(tok)

for p in ["/tensor/benchmark-cases?anchor_slug=CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
          "/tensor/benchmark_cases?anchor_slug=CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
          "/tensor/benchmark-cases",
          "/tensor/cases?anchor_slug=CASE-BEM-SUBMERGED-SPHERE-HAVELOCK"]:
    s, d = req("GET", p, tok=tok)
    print("GET", p, "->", s, str(d)[:400])
    if s == 200:
        break
