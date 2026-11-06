import json
import os
import urllib.request
import urllib.error

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
user = os.environ.get("TENSOR_AGENT_USER")
passwd = os.environ.get("TENSOR_AGENT_PASS")

auth_url = f"{base}/tensor/auth"
data = json.dumps({"username": user, "password": passwd}).encode()
req = urllib.request.Request(auth_url, data=data, headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req) as resp:
        print(resp.read().decode())
except urllib.error.HTTPError as e:
    print(f"Error {e.code}: {e.read().decode()}")
