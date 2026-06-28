
import requests
import json
from bs4 import BeautifulSoup

response = requests.get("http://127.0.0.1:5003", timeout=10)
html = response.text
soup = BeautifulSoup(html, "html.parser")

print("=== 1. audienceReports from rendered page ===")
lines = html.splitlines()
for line in lines:
    if 'const audienceReports' in line:
        print("Found line!")
        json_str = line.split('=',1)[1].strip().rstrip(';')
        parsed = json.loads(json_str)
        print("Keys in audienceReports:", list(parsed.keys()))
        print("ML Engineer summary:", repr(parsed['ML Engineer']['executive_summary'][:50]))
        print("Executive summary:", repr(parsed['Executive']['executive_summary'][:50]))
        break

print("\n=== 2. Checking for DOM elements ===")
exec_summary = soup.find(id="audience-executive-summary")
findings = soup.find(id="audience-findings")
recommendations = soup.find(id="audience-recommendations")
print("audience-executive-summary exists:", exec_summary is not None)
print("audience-executive-summary initial content:", repr(exec_summary.text.strip()[:80]) if exec_summary else "NOT FOUND")
print("audience-findings exists:", findings is not None)
print("audience-recommendations exists:", recommendations is not None)
