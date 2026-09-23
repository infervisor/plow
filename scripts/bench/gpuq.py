#!/usr/bin/env python3
"""Persistent FIFO for campaign GPU jobs; gpulease owns device locks and audits."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

REPO = Path(__file__).resolve().parents[2]
LEASE = REPO / "perf-data/tools/gpulease"


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def work(root):
    render_nodes = list(Path("/dev/dri").glob("renderD*"))
    if Path("/dev/kfd").exists() and (not render_nodes or not all(os.access(p, os.R_OK | os.W_OK) for p in render_nodes)):
        raise SystemExit("queue runner needs render-device access; enter the render group before submit")
    with (root / "runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        idle = time.monotonic()
        while time.monotonic() - idle < 1800:
            pending = []
            for path in sorted(root.glob("*.json")):
                job = json.loads(path.read_text())
                if job["state"] == "queued":
                    pending.append((path, job))
            if not pending:
                time.sleep(1)
                continue
            idle = time.monotonic()
            # Do not run a benchmark alongside a server that bypassed the lease locks.
            audit = subprocess.run([str(LEASE), "--audit"], capture_output=True, text=True)
            if audit.returncode or audit.stdout.strip() != "GPU: no foreign compute procs":
                time.sleep(5)
                continue
            path, job = pending[0]
            job.update(state="running", runner_pid=os.getpid(), started=time.time())
            save(path, job)
            with (root / (path.stem + ".log")).open("ab") as log:
                child = subprocess.Popen([str(LEASE), "-n", str(job["ngpu"]), job["label"], *job["command"]],
                                         cwd=job["cwd"], stdout=log, stderr=subprocess.STDOUT)
                job["pid"] = child.pid
                save(path, job)
                rc = child.wait()
            job.update(state="done" if rc == 0 else "failed", rc=rc, finished=time.time())
            save(path, job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/tmp/plow-gpuq"))
    sub = parser.add_subparsers(dest="action", required=True)
    submit = sub.add_parser("submit")
    submit.add_argument("label")
    submit.add_argument("ngpu", type=int)
    submit.add_argument("command", nargs=argparse.REMAINDER)
    sub.add_parser("run")
    sub.add_parser("status")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.action == "run":
        work(root)
    elif args.action == "status":
        for path in sorted(root.glob("*.json")):
            print(path.stem, json.loads(path.read_text()))
    else:
        if not os.environ.get("ROCM_PATH"):
            parser.error("submit inside nix develop")
        if args.ngpu < 1 or not args.command:
            parser.error("submit needs a positive GPU count and a command")
        job_id = f"{time.time_ns()}-{uuid.uuid4().hex[:8]}"
        save(root / (job_id + ".json"), dict(state="queued", label=args.label, ngpu=args.ngpu,
             command=args.command, cwd=os.getcwd(), submitted=time.time()))
        with (root / "runner.log").open("ab") as log:
            subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--root", str(root), "run"],
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
        print(job_id)


if __name__ == "__main__":
    main()
