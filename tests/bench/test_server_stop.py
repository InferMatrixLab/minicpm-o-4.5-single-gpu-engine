"""Server.stop signals the leader first, so a server retires its own workers in order (sglang-omni MPS teardown)."""

import subprocess
import sys
import time

import bench.servers as servers
from bench.servers import Server, ServerSpec

LEADER = r'''
import signal, subprocess, sys, time
log, mode = sys.argv[1], sys.argv[2]
child = subprocess.Popen([sys.executable, "-c", """
import signal, sys, time
def h(*_):
    open(sys.argv[1], "a").write("worker-term\\n"); sys.exit(0)
signal.signal(signal.SIGTERM, h)
while True: time.sleep(0.1)
""", log])
def h(*_):
    open(log, "a").write("leader-term\n")
    if mode == "orderly":
        time.sleep(0.5)
        open(log, "a").write("leader-retires-worker\n")
        child.terminate(); child.wait()
    sys.exit(0)
signal.signal(signal.SIGTERM, h)
while True: time.sleep(0.1)
'''


def _start(tmp_path, mode, monkeypatch):
    monkeypatch.setattr(servers, "gpu_memory_used_mib", lambda gpu: 0.0)
    script = tmp_path / "leader.py"
    script.write_text(LEADER)
    log = tmp_path / f"{mode}.log"
    srv = Server(ServerSpec(name="fake", cmd=[sys.executable, str(script), str(log), mode], port=1, gpu=0), log_path=tmp_path / "srv.log")
    env = dict(__import__("os").environ)
    srv.proc = subprocess.Popen(srv.spec.cmd, start_new_session=True, env=env)
    srv.pgid = srv.proc.pid
    time.sleep(1.0)
    return srv, log


def test_leader_retires_workers_before_any_group_signal(tmp_path, monkeypatch):
    srv, log = _start(tmp_path, "orderly", monkeypatch)
    t0 = time.monotonic()
    srv.stop()
    assert log.read_text().splitlines() == ["leader-term", "leader-retires-worker", "worker-term"]
    assert time.monotonic() - t0 < 10


def test_orphaned_workers_are_still_stopped(tmp_path, monkeypatch):
    srv, log = _start(tmp_path, "abandon", monkeypatch)
    pgid = srv.pgid
    srv.stop()
    assert log.read_text().splitlines() == ["leader-term", "worker-term"]  # the group signal reached the orphan
    assert not Server._group_alive(pgid)
