import os
import json
from uuid import uuid4
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.join(BASE_DIR, 'reports', 'generated')
os.makedirs(OUT_DIR, exist_ok=True)


def _sanitize_json(value):
    if isinstance(value, dict):
        return {k: _sanitize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(v) for v in value]
    if isinstance(value, (str, bytes, bytearray)):
        return value
    if hasattr(value, 'to_dict'):
        try:
            return _sanitize_json(value.to_dict())
        except Exception:
            pass
    if hasattr(value, 'tolist'):
        try:
            return _sanitize_json(value.tolist())
        except Exception:
            pass
    return value


def build_report(results, dataset_meta=None):
    """
    Build an HTML and JSON report from results dict and save to disk.
    Returns report_id.
    """
    report_id = uuid4().hex
    timestamp = datetime.utcnow().isoformat()

    payload = {
        'report_id': report_id,
        'timestamp': timestamp,
        'dataset': dataset_meta or {},
        'results': _sanitize_json(results)
    }

    json_path = os.path.join(OUT_DIR, f"{report_id}.json")
    html_path = os.path.join(OUT_DIR, f"{report_id}.html")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # Minimal HTML generation using a simple template layout
    title = f"FeynML Phase4 Report {report_id}"
    html = ["<!doctype html>", "<html><head>", f"<title>{title}</title>", "<meta charset=\"utf-8\">", "</head><body>"]
    html.append(f"<h1>{title}</h1>")
    html.append(f"<p>Generated: {timestamp} UTC</p>")
    ds = payload['dataset'] or {}
    html.append(f"<p>Dataset: {ds.get('name', 'unknown')} - shape: {ds.get('shape', '')}</p>")

    # Render a few sections
    for k, v in (results or {}).items():
        html.append(f"<h2>{k}</h2>")
        try:
            html.append(f"<pre>{json.dumps(v, indent=2)}</pre>")
        except Exception:
            html.append(f"<pre>{str(v)}</pre>")

    html.append(f"<p><a href=\"{report_id}.json\">Download JSON</a></p>")
    html.append("</body></html>")

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))

    return report_id


def get_report(report_id):
    json_path = os.path.join(OUT_DIR, f"{report_id}.json")
    html_path = os.path.join(OUT_DIR, f"{report_id}.html")
    if os.path.exists(json_path) and os.path.exists(html_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        with open(html_path, 'r', encoding='utf-8') as f:
            html = f.read()
        return {'json': data, 'html': html}
    return None
