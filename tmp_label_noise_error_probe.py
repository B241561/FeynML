import os
import uuid
import http.cookiejar
import urllib.request
import urllib.parse
from sklearn.datasets import load_iris

# Reuse the existing upload and login flow to reach the label-noise run endpoint.
iris = load_iris()
cols = iris.feature_names + ["species"]
rows = [list(iris.data[i]) + [int(iris.target[i])] for i in range(len(iris.data))]
fn = "test_iris.csv"
with open(fn, "w", newline="", encoding="utf-8") as f:
    f.write("\n".join([",".join(cols)] + [",".join(map(str, row)) for row in rows]))

cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
base = "http://127.0.0.1:5000"

signup_data = urllib.parse.urlencode({
    "name": "Error Probe",
    "email": f"error_probe_{uuid.uuid4().hex[:8]}@example.com",
    "password": "Password123"
}).encode("utf-8")
req = urllib.request.Request(base + "/signup", data=signup_data, method="POST")
resp = opener.open(req, timeout=20)
print("signup", resp.getcode(), resp.geturl())

boundary = "----WebKitFormBoundary" + uuid.uuid4().hex
body = []
body.append(
    f"--{boundary}\r\nContent-Disposition: form-data; name=\"dataset\"; filename=\"{fn}\"\r\nContent-Type: text/csv\r\n\r\n".encode("utf-8")
)
with open(fn, "rb") as f:
    body.append(f.read())
body.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
req = urllib.request.Request(base + "/upload", data=b"".join(body), method="POST")
req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
resp = opener.open(req, timeout=40)
print("upload", resp.getcode(), resp.geturl())

req = urllib.request.Request(base + "/analysis/label-noise")
resp = opener.open(req, timeout=20)
html = resp.read().decode("utf-8")
print("/analysis/label-noise GET", resp.getcode())
dataset_id = html.split('<option value="')[1].split('"')[0]
print("dataset_id", dataset_id)

form_data = urllib.parse.urlencode({"dataset_id": dataset_id, "target_col": "species"}).encode("utf-8")
req = urllib.request.Request(base + "/analysis/label-noise/run", data=form_data, method="POST")
try:
    resp = opener.open(req, timeout=120)
    final = resp.geturl()
    print("label-noise run", resp.getcode(), final)
    req = urllib.request.Request(final)
    resp = opener.open(req, timeout=20)
    print("report page", resp.getcode(), final)
except urllib.error.HTTPError as e:
    print('http error', e.code, e.reason)
    body = e.read().decode('utf-8', errors='replace')
    print('error body:')
    print(body)
