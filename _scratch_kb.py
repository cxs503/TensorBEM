import json, os, urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
U = os.environ["TENSOR_AGENT_USER"]; P = os.environ["TENSOR_AGENT_PASS"]

def req(path, data=None, tok=None, method=None):
    url = BASE + path
    body = json.dumps(data).encode() if data is not None else None
    r = urllib.request.Request(url, data=body, method=method or ("POST" if data else "GET"))
    r.add_header("Content-Type", "application/json")
    if tok: r.add_header("Authorization", "Bearer " + tok)
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}

tok = req("/auth/login", {"username": U, "password": P})
t = tok.get("access_token") or tok.get("token") or tok
print("LOGIN keys:", list(tok.keys()) if isinstance(tok, dict) else tok)
if isinstance(t, str):
    open("/tmp/tok.txt", "w").write(t)

for p in ["/tensor/knowledge_base?keywords=TensorBEM",
          "/tensor/knowledge_base/capability?domain=TensorBEM",
          "/tensor/negative_knowledge_ledger?keywords=TensorBEM",
          "/tensor/oss_solutions?domain=TensorBEM"]:
    out = req(p, tok=t if isinstance(t, str) else None)
    print("\n=== ", p)
    s = json.dumps(out, ensure_ascii=False)
    print(s[:2000])