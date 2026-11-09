import subprocess, sys, time, urllib.request, json, os

os.chdir(r"c:\Users\MSii\Desktop\NIBM\3rd Year Matrerial\Project\PhishingDetection")
sys.path.insert(0, r"c:\Users\MSii\Desktop\NIBM\3rd Year Matrerial\Project\PhishingDetection")

python_exe = r"c:\Users\MSii\Desktop\NIBM\3rd Year Matrerial\Project\PhishingDetection\.venv\Scripts\python.exe"

print("Starting uvicorn server...")
proc = subprocess.Popen(
    [python_exe, "-m", "uvicorn", "app.main:app",
     "--host", "127.0.0.1", "--port", "8765",
     "--timeout-keep-alive", "120"],
    cwd=r"c:\Users\MSii\Desktop\NIBM\3rd Year Matrerial\Project\PhishingDetection",
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

# Wait for server to start
time.sleep(10)

def do_get(url, timeout=30):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except Exception as e:
        return None, str(e)

def do_post(url, body, timeout=60):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())
    except Exception as e:
        return None, str(e)

BASE = "http://127.0.0.1:8765"

print("\n--- Test 1: GET /health ---")
status, body = do_get(f"{BASE}/health")
print(f"Status: {status}, Body: {body}")

print("\n--- Test 2: POST /predict ---")
status, body = do_post(f"{BASE}/predict", {"url": "http://example.com"})
print(f"Status: {status}, Body: {body}")

print("\n--- Test 3: POST /explain (ig) ---")
status, body = do_post(f"{BASE}/explain", {"url": "http://example.com", "method": "ig"}, timeout=60)
print(f"Status: {status}")
if isinstance(body, dict):
    print(f"  method: {body.get('method')}, tokens: {len(body.get('tokens', []))}, components: {body.get('components')}")
else:
    print(f"  Body: {body}")

print("\n--- Test 4: POST /explain (lime) ---")
status, body = do_post(f"{BASE}/explain", {"url": "http://example.com", "method": "lime"}, timeout=60)
print(f"Status: {status}")
if isinstance(body, dict):
    print(f"  method: {body.get('method')}, tokens: {len(body.get('tokens', []))}, components: {body.get('components')}")
else:
    print(f"  Body: {body}")

print("\n--- Test 5: POST /explain (shap) ---")
status, body = do_post(f"{BASE}/explain", {"url": "http://example.com", "method": "shap"}, timeout=60)
print(f"Status: {status}")
if isinstance(body, dict):
    print(f"  method: {body.get('method')}, tokens: {len(body.get('tokens', []))}, components: {body.get('components')}")
else:
    print(f"  Body: {body}")

print("\n--- Test 6: GET /research_results ---")
status, body = do_get(f"{BASE}/research_results", timeout=30)
print(f"Status: {status}")
if isinstance(body, dict):
    for k, v in body.items():
        print(f"  {k}: n_urls={v.get('n_urls')}, spearman={v.get('occlusion_spearman_mean')}, runtime={v.get('explain_seconds_mean')}")
else:
    print(f"  Body: {body}")

print("\n--- Server output (last 30 lines) ---")
proc.terminate()
try:
    out, _ = proc.communicate(timeout=5)
    lines = out.strip().split('\n')
    for line in lines[-30:]:
        print(line)
except:
    pass

print("\nDone.")
