"""What every results/metric/ file records about how it was made (metric plan, required fields):
git commit (and whether the worktree was dirty), command line, timestamp, host."""
import hashlib
import json
import socket
import subprocess
import sys
from datetime import datetime


def _git(*cmd):
    return subprocess.run(["git", *cmd], capture_output=True, text=True).stdout.strip()


def provenance():
    return {"commit": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
            "command": " ".join(sys.argv), "written": datetime.now().isoformat(timespec="seconds"),
            "host": socket.gethostname()}


def config_hash(config):
    """Short hash of a JSON-serialisable config (feature or metric definition)."""
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
