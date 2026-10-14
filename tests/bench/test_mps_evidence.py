import os

from bench.servers import classify_mps, descendants, process_info


def P(pid, typ, name="python"):
    return {"pid": pid, "type": typ, "process_name": name, "used_memory": "1 MiB"}


def test_unrelated_mps_server_does_not_make_candidate_mps():
    gpu = [P(11, "C", "nvidia-cuda-mps-server"), P(22, "C"), P(33, "M+C")]
    ev = classify_mps({22, 40}, gpu)  # 33 is another tenant's MPS client
    assert ev["observed"] == "off"
    assert [w["pid"] for w in ev["workers"]] == [22]
    assert {p["pid"] for p in ev["foreign"]} == {11, 33}


def test_candidate_mps_clients_observed_on():
    gpu = [P(11, "C", "nvidia-cuda-mps-server"), P(22, "M+C"), P(23, "M+C")]
    assert classify_mps({22, 23}, gpu)["observed"] == "on"


def test_mixed_and_missing_observations():
    assert classify_mps({22, 23}, [P(22, "M+C"), P(23, "C")])["observed"] == "mixed"
    assert classify_mps({22}, None)["observed"] == "unknown"  # query failed
    assert classify_mps({22}, [P(99, "C")])["observed"] == "unknown"  # no worker of this server on the GPU


def test_process_tree_and_resolved_environment():
    me = os.getpid()
    assert me in descendants(os.getppid())
    info = process_info(me)
    assert info["cmdline"] and isinstance(info["env"], dict)
