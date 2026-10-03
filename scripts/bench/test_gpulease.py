import os
from pathlib import Path
import signal
import subprocess
import time

import pytest

LEASE = Path(__file__).resolve().parents[2] / "perf-data/tools/gpulease"


@pytest.fixture
def env(tmp_path):
    return dict(os.environ, GPU_LEASE_DIR=str(tmp_path / "lease"), GPU_LEASE_NGPU="1",
                GPU_LEASE_VENDOR="none", GPU_LEASE_TIMEOUT="60")


def lease(env, *args, **kw):
    return subprocess.Popen([str(LEASE), *args], env=env, start_new_session=True, **kw)


def wait_for(predicate, timeout=20):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError("timed out")


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    state = Path(f"/proc/{pid}/stat").read_text().split()[2]
    return state != "Z"


def test_waiters_are_served_in_arrival_order(env, tmp_path):
    order = tmp_path / "order"
    holder = lease(env, "hold", "sleep", "3")
    wait_for(lambda: "hold ACQUIRED" in (Path(env["GPU_LEASE_DIR"]) / "lease.log").read_text()
             if (Path(env["GPU_LEASE_DIR"]) / "lease.log").exists() else False)
    waiters = []
    for name in "abcde":
        waiters.append(lease(env, name, "sh", "-c", f"echo {name} >> {order}"))
        time.sleep(0.3)
    assert holder.wait(30) == 0
    assert [w.wait(30) for w in waiters] == [0] * 5
    assert order.read_text().split() == list("abcde")


def test_killing_the_lease_stops_the_whole_job_tree(env, tmp_path):
    pidfile = tmp_path / "pid"
    run = lease(env, "tree", "bash", "-c", f"sleep 300 & echo $! > {pidfile}; wait")
    wait_for(lambda: pidfile.exists() and pidfile.read_text().strip())
    child = int(pidfile.read_text())
    run.send_signal(signal.SIGTERM)
    run.wait(30)
    wait_for(lambda: not alive(child))


def test_strays_left_by_the_command_are_stopped_with_the_lease(env, tmp_path):
    pidfile = tmp_path / "pid"
    run = lease(env, "stray", "bash", "-c", f"sleep 300 & echo $! > {pidfile}")
    assert run.wait(30) == 0
    wait_for(lambda: not alive(int(pidfile.read_text())))


def test_hold_limit_preempts_only_when_someone_waits(env, tmp_path):
    alone = lease(env, "--max-hold", "1", "alone", "sleep", "4")
    assert alone.wait(30) == 0
    hog = lease(env, "--max-hold", "2", "hog", "sleep", "120")
    time.sleep(1)
    waiter = lease(env, "next", "true")
    assert hog.wait(30) == 124
    assert waiter.wait(30) == 0
    assert "hog PREEMPTED" in (Path(env["GPU_LEASE_DIR"]) / "lease.log").read_text()


def test_status_word_is_not_a_label_and_a_label_needs_a_command(env):
    out = subprocess.run([str(LEASE), "status"], env=env, capture_output=True, text=True, timeout=30)
    assert out.returncode == 0 and "queue" in out.stdout
    assert not (Path(env["GPU_LEASE_DIR"]) / "lease.log").exists()
    bare = subprocess.run([str(LEASE), "-n", "1", "job"], env=env, capture_output=True, text=True, timeout=30)
    assert bare.returncode == 2 and "no command" in bare.stderr


def test_a_newcomer_cannot_take_cards_an_older_waiter_needs(env, tmp_path):
    env = dict(env, GPU_LEASE_NGPU="2")
    order = tmp_path / "order"
    log = Path(env["GPU_LEASE_DIR"]) / "lease.log"
    holder = lease(env, "-n", "1", "hold", "sleep", "3")
    wait_for(lambda: log.exists() and "hold ACQUIRED" in log.read_text())
    big = lease(env, "-n", "2", "big", "sh", "-c", f"echo big >> {order}")
    time.sleep(0.5)
    small = lease(env, "-n", "1", "small", "sh", "-c", f"echo small >> {order}")
    assert [p.wait(30) for p in (holder, big, small)] == [0, 0, 0]
    assert order.read_text().split() == ["big", "small"]
