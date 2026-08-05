import httpx
import time
import json

t0 = time.time()

# Login
token_r = httpx.post(
    'http://localhost:8000/api/v1/auth/login',
    content='username=user%40hbia.ai&password=Password123!',
    headers={'Content-Type': 'application/x-www-form-urlencoded'}
)
token = token_r.json()['access_token']
t1 = time.time()
print(f"Auth: {t1-t0:.1f}s")

# Query
r = httpx.post(
    'http://localhost:8000/api/v1/chat/query',
    json={'query': 'Who is Chandra Babu Naidu?', 'use_web_search': True},
    headers={
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    },
    timeout=30.0
)
t2 = time.time()
print(f"Query: {t2-t1:.1f}s")
d = r.json()
print(f"Status: {r.status_code}")
print(f"Trust: {d.get('trust_score')}")
print(f"Sources: {len(d.get('sources', []))}")
print(f"Response:\n{d.get('response', '')[:600]}")
