"""
Repository-local interpreter bootstrap.

Ensures that running `python` or `pytest` from the project root can see the
bundled virtual-environment packages when the shell is not already activated.
"""

from __future__ import annotations

import os
import sys


def _candidate_site_packages(root: str) -> list[str]:
    candidates = []
    for env_name in (".venv311", ".venv", "venv"):
        site_packages = os.path.join(root, env_name, "Lib", "site-packages")
        if os.path.isdir(site_packages):
            candidates.append(site_packages)
    return candidates


_ROOT = os.path.abspath(os.path.dirname(__file__))

for _site_packages in reversed(_candidate_site_packages(_ROOT)):
    if _site_packages not in sys.path:
        sys.path.insert(0, _site_packages)
