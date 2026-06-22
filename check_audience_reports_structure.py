
import json

with open("webapp/reports/report_1782045837.json", "r") as f:
    data = json.load(f)

print("=== AI Investigator ===")
print("executive_summary:", repr(data['ai_investigator']['executive_summary']))
print("\n=== ML Engineer Audience Report ===")
print("executive_summary:", repr(data['audience_reports']['ML Engineer']['executive_summary']))
print("\nAre they the same?", data['ai_investigator']['executive_summary'] == data['audience_reports']['ML Engineer']['executive_summary'])
print("\n=== Executive Audience Report ===")
print("executive_summary:", repr(data['audience_reports']['Executive']['executive_summary']))
