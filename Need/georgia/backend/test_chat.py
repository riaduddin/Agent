import requests
import json

url = "http://127.0.0.1:8080/backend/api/v2/docs/chat"
token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJmcmVzaCI6ZmFsc2UsImlhdCI6MTc2ODIxMjU3MSwianRpIjoiNDJmMGYxMzUtYThjZS00ZjQ3LTg0NjktMmE0NTcwNWFiOTdiIiwidHlwZSI6ImFjY2VzcyIsInN1YiI6InN1cGVyYWRtaW5AZXhhbXBsZS5jb20iLCJuYmYiOjE3NjgyMTI1NzEsImNzcmYiOiJlNjViZmRiZS00Y2ZjLTQxNjYtODRiZS0wNmExNTQ0Yjk1OGUiLCJleHAiOjE3NjgyOTg5NzEsInJvbGUiOiJzdXBlcmFkbWluIn0.YUzutqGG3myhjXVReHTRkLnk4KrHuo1lt8_CQ4kLVn0"

headers = {
    "Accept": "text/event-stream",
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

payload = {
    "query": "get me information about 309145",
    "session_id": "VpTd2vs9wtcDYM842Egb"
}

print(f"Sending request to {url}...")
try:
    response = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
    print(f"Status Code: {response.status_code}")
    if response.status_code != 200:
        print(f"Response: {response.text}")
    else:
        for line in response.iter_lines():
            if line:
                print(line.decode('utf-8'))
except Exception as e:
    print(f"Error: {e}")
