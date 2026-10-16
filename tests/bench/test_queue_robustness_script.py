"""tools/queue/robustness_run_uuid.sh with stub server and checker (no GPU): the checker's exit status is logged and
returned, and the server is stopped and reaped on every path."""

import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "queue" / "robustness_run_uuid.sh"
SERVE = """#!/usr/bin/env bash
exec {py} -c '
import http.server, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()' "$2"
"""


def _run(tmp_path, rc):
    w = tmp_path / "wt"
    (w / "scripts").mkdir(parents=True)
    (w / "scripts" / "env.sh").write_text("")
    (w / "serve_stub.sh").write_text(SERVE.format(py=sys.executable))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = tmp_path / "out"
    (w / "checker_stub.sh").write_text(f"exit {rc}\n")
    env = {**os.environ, "FAKE_GPU_UUID": "GPU-fake", "SERVE": "serve_stub.sh", "HEALTH_POLL_S": "0.2", "CHECKER": f"bash {w / 'checker_stub.sh'}"}
    p = subprocess.run(["bash", str(SCRIPT), "0", "99", str(w), "cfg", str(port), str(out)], env=env, capture_output=True, text=True, timeout=120)
    pid = int((out / "server.pid").read_text())
    return p, pid


def test_checker_status_is_returned_and_the_server_reaped(tmp_path):
    for rc in (7, 0):
        p, pid = _run(tmp_path / str(rc), rc)
        assert p.returncode == rc, (p.stdout, p.stderr)
        assert f"robustness rc={rc}" in p.stdout
        assert not Path(f"/proc/{pid}").exists() or "Z" in Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0]
