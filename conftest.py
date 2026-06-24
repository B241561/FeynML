from __future__ import annotations

import os
import sys


ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

for env_name in (".venv311", ".venv", "venv"):
    site_packages = os.path.join(ROOT_DIR, env_name, "Lib", "site-packages")
    if os.path.isdir(site_packages) and site_packages not in sys.path:
        sys.path.insert(0, site_packages)

try:
    import pandas as pd
except Exception:
    pd = None

if pd is not None and not hasattr(pd, "TimedOffset"):
    pd.TimedOffset = pd.Timedelta
