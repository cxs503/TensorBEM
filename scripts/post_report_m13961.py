import json, os, urllib.request

base = os.environ["TENSOR_API_BASE"]
user = os.environ["TENSOR_AGENT_USER"]
pw = os.environ["TENSOR_AGENT_PASS"]

login = urllib.request.Request(
    base + "/auth/login",
    data=json.dumps({"username": user, "password": pw}).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(login, timeout=120) as response:
    token = json.loads(response.read())
token = token.get("access_token") or token.get("token")

payload = json.loads(open("docs/acoustic3d_ices209/report_payload.json").read())
request = urllib.request.Request(
    base + "/tensor/reports",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
)
with urllib.request.urlopen(request, timeout=300) as response:
    body = json.loads(response.read())
print(json.dumps({key: body[key] for key in body if key in ("id", "mission_id", "title", "created_at")}, ensure_ascii=False))