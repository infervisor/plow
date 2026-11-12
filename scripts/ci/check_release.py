#!/usr/bin/env python3
"""Check `nix build .#plowrt-release*` outputs before they are published.

    check_release.py <semver> <release dir>...

Each directory holds the tarball, its `.sha256` and `release.json`. Checked:
the version is `<semver>+<commit>` from a clean tree, the digest matches, the
tarball holds `plowrt` and `plowc` for the triple it names, and a binary this
machine can run reports that version.
"""

import hashlib
import json
import pathlib
import platform
import subprocess
import sys
import tarfile
import tempfile

# ELF e_machine per triple arch.
MACHINE = {"x86_64": 62, "aarch64": 183}
BINARIES = ("plowrt", "plowc")


def interpreter(elf):
    """The ELF64 program interpreter (PT_INTERP), or "" for a static binary."""
    phoff = int.from_bytes(elf[32:40], "little")
    phentsize = int.from_bytes(elf[54:56], "little")
    for i in range(int.from_bytes(elf[56:58], "little")):
        ph = elf[phoff + i * phentsize :][:phentsize]
        if int.from_bytes(ph[:4], "little") == 3:
            offset, size = int.from_bytes(ph[8:16], "little"), int.from_bytes(ph[32:40], "little")
            return elf[offset : offset + size].rstrip(b"\0").decode()
    return ""


def check(semver, directory):
    release = json.loads((directory / "release.json").read_text())
    version, triple, name = release["version"], release["triple"], release["file"]
    if not version.startswith(semver + "+") or "dirty" in version or version.endswith("+unknown"):
        raise SystemExit(f"{directory}: version {version!r} is not a clean build of {semver}")
    tarball = directory / name
    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
    recorded = (directory / f"{name}.sha256").read_text().split()[0]
    if digest != recorded:
        raise SystemExit(f"{tarball}: hashes to {digest}, its .sha256 says {recorded}")
    arch = triple.split("-")[0]
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(tarball) as tar:
            tar.extractall(tmp, filter="data")
        root = pathlib.Path(tmp) / name.removesuffix(".tar.gz")
        for binary in BINARIES:
            path = root / binary
            header = path.read_bytes()[:20]
            if header[:4] != b"\x7fELF" or int.from_bytes(header[18:20], "little") != MACHINE[arch]:
                raise SystemExit(f"{name}: {binary} is not a {arch} ELF binary")
            if interpreter(path.read_bytes()).startswith("/nix/store"):
                raise SystemExit(f"{name}: {binary} loads through /nix/store and runs on no other host")
            if arch == platform.machine():
                said = subprocess.check_output([str(path), "--version"], text=True).strip()
                want = version if binary == "plowrt" else semver
                if not said.endswith(want):
                    raise SystemExit(f"{name}: {binary} --version says {said!r}, not {want}")
    print(f"ok {name} ({len(BINARIES)} binaries, sha256 {digest[:12]})")


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    for directory in sys.argv[2:]:
        check(sys.argv[1], pathlib.Path(directory))


if __name__ == "__main__":
    main()
