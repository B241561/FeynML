
from super_simple_render import app
import json

with app.test_client() as client:
    response = client.get("/")
    html = response.data.decode("utf-8")
    
    print("=== 1. Find audienceReports ===")
    lines = html.splitlines()
    for line in lines:
        if "const audienceReports" in line:
            print("Found line!")
            json_str = line.split("=", 1)[1].strip().rstrip(";")
            parsed = json.loads(json_str)
            print("audienceReports keys:", list(parsed.keys()))
            print("ML Engineer's executive_summary starts with:", repr(parsed["ML Engineer"]["executive_summary"][:60]))
            print("Executive's executive_summary starts with:", repr(parsed["Executive"]["executive_summary"][:60]))
            break
            
    print("\n=== 2. Check for element IDs ===")
    print("audience-executive-summary exists:", "audience-executive-summary" in html)
    print("audience-findings exists:", "audience-findings" in html)
    print("audience-recommendations exists:", "audience-recommendations" in html)
    
    # Find AI Investigator initial content
    print("\n=== 3. Initial content ===")
    idx1 = html.find("audience-executive-summary")
    if idx1 != -1:
        print("audience-executive-summary found around position", idx1)
        snippet = html[idx1:idx1 + 500]
        print(snippet)
