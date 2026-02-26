import requests

BASE_URL_SCRAPE = "https://smartcrawl.live"  #
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjp7Il9pZCI6IjY4ODhiYjc5ZTA0ZmYzNDQ3OTg5M2Q1OSIsIm5hbWUiOiJSYXNoYWR1emFtbWFuIFJpYW4iLCJlbWFpbCI6InJpYW4ubWVybkBnbWFpbC5jb20iLCJhdXRoX3R5cGUiOiJlbWFpbCIsInJvbGUiOiJ1c2VyIiwic3RhdHVzIjoiYWN0aXZlIiwiZW1haWxWZXJpZmllZCI6ZmFsc2UsImNyZWF0ZWRBdCI6IjIwMjUtMDctMjlUMTI6MTU6NTMuMDYzWiIsInVwZGF0ZWRBdCI6IjIwMjUtMDctMjlUMTI6MTU6NTMuMDYzWiIsIl9fdiI6MH0sImlhdCI6MTc1Mzc5MTQ4NCwiZXhwIjoxNzU2MzgzNDg0fQ.z3GAolchKsp3-2rlproG3E_ii5JHtNBRnwt0y1xZdJQ"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
    "Accept": "application/json"
}

body = {
    "URLS": [
        "https://www.tesla.com/careers",
        "https://segmentify.com/blog/tesla-marketing-strategy/"
    ]
}

resp = requests.post(f"{BASE_URL_SCRAPE}/backend/scraper/scrape", headers=headers, json=body)
print(resp.status_code, resp.text)