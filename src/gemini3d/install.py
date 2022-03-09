"""
installs Gemini3D prerequisite libraries
"""

from __future__ import annotations
import tempfile
import subprocess
import json
from pathlib import Path
import importlib.resources

from .web import git_download
from .cmake import exe


src_dir = Path(tempfile.gettempdir()) / "gemini3d-libs"

if not (src_dir / "CMakeLists.txt").is_file():
    jmeta = json.loads(importlib.resources.read_text("gemini3d", "libraries.json"))
    git_download(src_dir, repo=jmeta["external"]["git"], tag=jmeta["external"]["tag"])

script = src_dir / "scripts/requirements.cmake"
if not script.is_file():
    raise FileNotFoundError(script)

cmd = [exe(), "-P", str(script)]

print(" ".join(cmd))
ret = subprocess.run(cmd)

raise SystemExit(ret.returncode)
