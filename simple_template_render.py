
import json
import os
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Set up Jinja2 with fake url_for dummy function (just for static urls!
def dummy_url_for(endpoint, **kwargs):
    if endpoint == "static":
        return "/static/" + kwargs.get("filename", "")
    return "#"

env = Environment(
    loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "webapp", "templates")),
    autoescape=select_autoescape(["html", "xml"])
)
env.globals['url_for'] = dummy_url_for

with open("webapp/reports/report_1782045837.json", "r") as f:
    data = json.load(f)

template = env.get_template("dashboard.html")
rendered = template.render(
    data=data,
    report_id="test-report",
    risk_level="HIGH",
    charts={},
    # Add other variables that base.html uses
    current_user=None,
    request=None
)

# Find audienceReports line
print("\n".join([f"{i:4}: {line}" for i, line in enumerate(rendered.splitlines()) if "audienceReports" in line])

print("\n\n=== Exec Summary Element:")
print("\n".join([f"{i:4}: {line}" for i, line in enumerate(rendered.splitlines()) if "audience-executive-summary" in line]))

print("\n\n=== AI Investigator section ===")
for line in rendered.splitlines():
    if "audience-executive-summary" in line:
        print(line)
    if "audience-findings" in line:
        print(line)
    if "audience-recommendations" in line:
        print(line)
