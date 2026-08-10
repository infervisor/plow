#!/usr/bin/env python3
"""Print the pid LISTENING on a TCP port, by walking /proc.

`ss`, `lsof`, `netstat` and `fuser` are all absent on the gfx942 dev box, and the shape of
failure that matters here is a SILENT one: a check written against `ss` degrades into no check
at all, which is exactly how a concurrent run's server came to answer a whole A/B battery for
somebody else. So this uses /proc, which is always there, and prints a DISTINGUISHABLE token
when it cannot answer rather than an empty string.

Prints: the pid, or `NOSOCK` (nothing is listening), or `UNKNOWN` (listening, but the socket's
inode was not found in any /proc/<pid>/fd we can read — e.g. another user's process).
"""
import os
import sys

TCP_LISTEN = "0A"


def listen_inodes(port: int) -> set[str]:
    found: set[str] = set()
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = open(path).readlines()[1:]
        except OSError:
            continue
        for line in lines:
            f = line.split()
            if len(f) < 10:
                continue
            try:
                local_port = int(f[1].split(":")[1], 16)
            except (IndexError, ValueError):
                continue
            if local_port == port and f[3] == TCP_LISTEN:
                found.add(f[9])
    return found


def owner(inodes: set[str]) -> str:
    targets = {"socket:[%s]" % i for i in inodes}
    for pid in sorted(p for p in os.listdir("/proc") if p.isdigit()):
        fddir = "/proc/%s/fd" % pid
        try:
            fds = os.listdir(fddir)
        except OSError:
            continue  # not ours, or gone between listdir and open
        for fd in fds:
            try:
                if os.readlink(os.path.join(fddir, fd)) in targets:
                    return pid
            except OSError:
                continue
    return "UNKNOWN"


def main() -> None:
    port = int(sys.argv[1])
    inodes = listen_inodes(port)
    print("NOSOCK" if not inodes else owner(inodes))


if __name__ == "__main__":
    main()
