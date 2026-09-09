#!/usr/bin/env python3
"""Shared helpers for the asset producers.

`pack_objset.py`, `pack_bundle.py` and `publish_dist.py` all speak the schemas
in `crates/plow-asset/src/dist.rs`. This module is the one place that knows how
a digest, an id, or a store path is formed, so the producers and the runtime
cannot drift on it.

Nothing here runs on a serving host. Producers use nix; consumers use HTTPS.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

INDEX_SCHEMA = "plow.dist.model.v1"
BUNDLE_SCHEMA = "plow.dist.bundle.v1"
OBJSET_SCHEMA = "plow.dist.objset.v1"


class Fail(Exception):
    """A refusal the operator can act on."""


def die(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    raise Fail(msg)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical_json(obj) -> bytes:
    """Stable bytes for a manifest.

    Sorted keys and no incidental whitespace, so republishing an unchanged
    manifest produces an unchanged digest and the store dedups it.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def short_id(*parts: str) -> str:
    """A 12-hex identity over the parts that define an artifact."""
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:12]


# --- git provenance ----------------------------------------------------------


def git_commit(repo: Path, rev: str = "HEAD") -> str:
    """Full 40-character SHA. Abbreviations are not provenance."""
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", rev],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        die(f"cannot resolve {rev} in {repo}: {out.stderr.strip()}")
    sha = out.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        die(f"{rev} resolved to {sha!r}, which is not a full commit SHA")
    return sha


def git_is_clean(repo: Path) -> bool:
    out = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.returncode == 0 and not out.stdout.strip()


# --- ELF inspection ----------------------------------------------------------

# Marker symbols the loader reads before opening an object. Recording them in
# the objset lets the arm check run ahead of any device work and name the
# missing symbol instead of failing later and vaguer.
_ARM_RE = re.compile(r"\b(plow_[A-Za-z0-9_]+)\b")


def readelf_bin() -> str:
    for env in ("PLOW_READELF", "PLOW_K3_READELF"):
        v = os.environ.get(env)
        if v and shutil.which(v):
            return v
    for cand in ("llvm-readelf", "readelf"):
        if shutil.which(cand):
            return cand
    die(
        "no readelf found. Run inside `nix develop`, which sets PLOW_READELF to "
        "the toolchain's llvm-readelf."
    )


def elf_symbols(path: Path) -> list[str]:
    out = subprocess.run(
        [readelf_bin(), "-sW", str(path)], capture_output=True, text=True, check=False
    )
    if out.returncode != 0:
        die(f"{path.name}: readelf failed: {out.stderr.strip()}")
    syms: set[str] = set()
    for line in out.stdout.splitlines():
        for m in _ARM_RE.finditer(line):
            syms.add(m.group(1))
    return sorted(syms)


def packet_hash_from_symbols(syms: list[str]) -> str | None:
    """A specialised object stamps the packet it pairs with.

    The value is encoded in the symbol NAME (`plow_packet_hash_lo_<hex>`),
    because plowrt reads `.symtab` before the object is on a device and a value
    would cost a round trip. A GENERAL object — every arm compiled — carries no
    stamp and pairs with any packet.
    """
    lo = hi = None
    for s in syms:
        if s.startswith("plow_packet_hash_lo_"):
            lo = s.removeprefix("plow_packet_hash_lo_")
        elif s.startswith("plow_packet_hash_hi_"):
            hi = s.removeprefix("plow_packet_hash_hi_")
    if lo is None and hi is None:
        return None
    if lo is None or hi is None:
        die("object carries half a packet-hash stamp; that is a broken build")
    return f"0x{int(hi, 16) << 32 | int(lo, 16):016x}"


# --- compression -------------------------------------------------------------


def compress(data: bytes, level: int = 19) -> bytes | None:
    """zstd-compress for transport, or `None` when no compressor is available.

    The digest always names the UNCOMPRESSED bytes, so a store may hold `.zst`,
    plain, or both, and every client stays correct. Compression is therefore an
    optimization the publisher may skip rather than a format requirement.
    """
    if not shutil.which("zstd"):
        return None
    out = subprocess.run(
        ["zstd", f"-{level}", "-T0", "-c", "-q"],
        input=data,
        capture_output=True,
        check=False,
    )
    if out.returncode != 0:
        return None
    return out.stdout


def human(n: int) -> str:
    v = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if v < 1024 or unit == "TiB":
            return f"{int(n)} B" if unit == "B" else f"{v:.1f} {unit}"
        v /= 1024
    return f"{n} B"


# --- store layout ------------------------------------------------------------


def blob_path(root: Path, digest: str) -> Path:
    return root / "v1" / "blobs" / "sha256" / digest


def model_dir(root: Path, namespace: str, name: str) -> Path:
    return root / "v1" / namespace / name


def objset_path(root: Path, objset_id: str) -> Path:
    return root / "v1" / "objsets" / f"{objset_id}.json"


def write_atomic(path: Path, data: bytes) -> None:
    """Stage and rename.

    Writing in place would let a concurrent reader see a partial manifest under
    a name that promises the whole thing — the same hazard
    `scripts/install_hsaco.sh` exists for on the object side.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".part{os.getpid()}")
    tmp.write_bytes(data)
    tmp.replace(path)


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        die(f"{path} does not exist")
    except json.JSONDecodeError as e:
        die(f"{path}: not valid JSON: {e}")


def load_toml(path: Path):
    import tomllib

    try:
        return tomllib.loads(path.read_text())
    except FileNotFoundError:
        die(f"{path} does not exist")
    except tomllib.TOMLDecodeError as e:
        die(f"{path}: not valid TOML: {e}")


def no_nix_store_paths(obj, where: str) -> None:
    """A published manifest must not reference the build host.

    Exported artifacts are consumed on machines with no nix and no `/nix/store`,
    so a leaked store path is a manifest that resolves to nothing there.
    """
    blob = json.dumps(obj)
    for bad in ("/nix/store/", "/home/", "/root/"):
        if bad in blob:
            die(
                f"{where}: contains a build-host path ({bad}). Published manifests must be "
                f"portable; record a reference, not a local path."
            )


def main_guard(fn):
    """Run `fn`, turning a refusal into a clean message and exit 1."""

    def wrapper() -> int:
        try:
            fn()
        except Fail as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        return 0

    return wrapper
