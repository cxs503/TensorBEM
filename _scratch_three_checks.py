import json, os, urllib.request

base = os.environ["TENSOR_API_BASE"]
user = os.environ["TENSOR_AGENT_USER"]
pw = os.environ["TENSOR_AGENT_PASS"]

def post(path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

def get(path, token):
    req = urllib.request.Request(base + path, headers={"Authorization": "Bearer " + token})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

tok = post("/auth/login", {"username": user, "password": pw})
token = tok.get("access_token") or tok.get("token")

cands = [
    "/knowledge-base",
    "/knowledge_base/capabilities?keywords=rigid",
    "/kb",
    "/knowledge",
    "/knowledge_base/capabilities?keywords=rigid_body",
    "/knowledge_base/capabilities?keywords=added_mass",
    "/knowledge_base/capabilities?keywords=TensorBEM",
    "/negative-knowledge-ledger",
    "/negative_knowledge",
    "/oss",
    "/oss-solutions",
    "/solutions",
    "/knowledge_base/capabilities?keywords=rigid",
]
for c in cands:
    try:
        d = get(c, token)
        print("==", c, "->", json.dumps(d, ensure_ascii=False)[:500])
    except Exception as e:
        print("==", c, "ERR", e)
