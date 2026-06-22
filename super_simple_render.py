
import json
import os
import flask
from jinja2 import Environment, FileSystemLoader

# Create minimal Flask app just for testing
app = flask.Flask(__name__, template_folder="webapp/templates", static_folder="webapp/static")

with open("webapp/reports/report_1782045837.json", "r") as f:
    report_data = json.load(f)

@app.route("/")
def index():
    return flask.render_template("dashboard.html",
                                 data=report_data,
                                 report_id="test",
                                 risk_level="HIGH",
                                 charts={})

if __name__ == "__main__":
    app.run(port=5003, debug=True, use_reloader=False)
