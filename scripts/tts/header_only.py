#!/usr/bin/env python3
"""Header-only checkpoints: compile packet STRUCTURE without downloading weights.

  python scripts/tts/header_only.py fetch REPO REVISION HUB_DIR [--include GLOB ...]
  HF_HUB_CACHE=HUB_DIR python scripts/tts/header_only.py run EXPORTER.py ARGS...

`fetch` writes HUB_DIR/models--<org>--<name>/snapshots/<REVISION>/ in the Hub cache layout the
exporters read. Small files are downloaded whole; `*.safetensors` keep only their header (two HTTP
range reads) and `*.pth` only the tensor tree of its pickle (the zip index and data.pkl). Every
header-only file is marked, and `header_only.json` records the pinned revision and file digests.

`run` executes an unmodified exporter on shape-only (meta) tensors: `load_file`, `torch.load` and
`save_file` read and write header-only files. Its outputs are header-only, so plowc compiles them to
structure-only packets, which plowrt refuses to execute.
"""
import argparse
import fnmatch
import io
import json
import os
import pickle
import runpy
import struct
import sys
import urllib.request
import zipfile

MARKER = "plow.header_only"
DEFAULT_INCLUDE = ["*.json", "*.safetensors", "*.pth", "*.model", "*.txt"]
SAFETENSORS_DTYPES = {
    "F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16",
    "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8", "U8": "uint8", "BOOL": "bool",
    "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2",
}


def _request(url, headers=None, method=None):
    headers = dict(headers or {})
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers, method=method), timeout=120)


class RangeFile(io.RawIOBase):
    """A seekable read-only view of a remote file through HTTP range requests."""

    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buffer):
        n = min(len(buffer), self.size - self.pos)
        if n <= 0:
            return 0
        with _request(self.url, {"Range": f"bytes={self.pos}-{self.pos + n - 1}"}) as response:
            if response.status != 206:
                raise OSError(f"{self.url}: server ignored the range request")
            data = response.read()
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def safetensors_header(data, extra=None):
    header = dict(data)
    meta = dict(header.pop("__metadata__", None) or {})
    meta[MARKER] = "1"
    meta.update(extra or {})
    header["__metadata__"] = meta
    raw = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("<Q", len(raw)) + raw


def read_safetensors_header(stream):
    (length,) = struct.unpack("<Q", stream.read(8))
    return json.loads(stream.read(length))


def is_header_only_safetensors(path):
    with open(path, "rb") as f:
        header = read_safetensors_header(f)
        data = f.tell()
    marked = (header.get("__metadata__") or {}).get(MARKER) == "1"
    if marked and os.path.getsize(path) != data:
        raise ValueError(f"{path}: header-only marker on a file that carries tensor data")
    return marked, header


class _Mapping(dict):
    """`OrderedDict` stand-in that accepts pickled attributes (`state_dict._metadata`)."""


class _TreeUnpickler(pickle.Unpickler):
    """The tensor tree of a torch zip checkpoint's data.pkl, never touching its storages."""

    def find_class(self, module, name):
        import torch
        if (module, name) in {("torch._utils", "_rebuild_tensor_v2"), ("torch._utils", "_rebuild_tensor")}:
            return lambda dtype, _offset, size, *_rest: {"__tensor__": [str(dtype).removeprefix("torch."), list(size)]}
        if (module, name) == ("torch._utils", "_rebuild_parameter"):
            return lambda tensor, *_rest: tensor
        if (module, name) == ("collections", "OrderedDict"):
            return _Mapping
        if module == "torch" and (name.endswith("Storage") or isinstance(getattr(torch, name, None), torch.dtype)):
            return getattr(torch, name)
        raise pickle.UnpicklingError(f"refusing {module}.{name} in a weights-only checkpoint")

    def persistent_load(self, pid):
        import torch
        if pid[0] != "storage":
            raise pickle.UnpicklingError(f"unsupported persistent id {pid[0]!r}")
        storage = pid[1]
        if isinstance(storage, torch.dtype):
            return storage
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return storage.dtype


def torch_tree(stream):
    archive = zipfile.ZipFile(stream)
    pickles = [n for n in archive.namelist() if n == "data.pkl" or n.endswith("/data.pkl")]
    if len(pickles) != 1:
        raise ValueError("not a torch zip checkpoint with one data.pkl")
    return _TreeUnpickler(io.BytesIO(archive.read(pickles[0]))).load()


def fetch(repo, revision, hub, include):
    info = json.load(_request(f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"))
    if info["sha"] != revision:
        raise SystemExit(f"{repo}: {revision} is not an immutable commit (resolves to {info['sha']})")
    root = os.path.join(hub, "models--" + repo.replace("/", "--"), "snapshots", revision)
    manifest = {"repo": repo, "revision": revision, "files": {}}
    for item in info["siblings"]:
        name = item["rfilename"]
        if not any(fnmatch.fnmatch(os.path.basename(name), g) or fnmatch.fnmatch(name, g) for g in include):
            continue
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
        path = os.path.join(root, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        entry = {"bytes": item.get("size"), "sha256": (item.get("lfs") or {}).get("sha256")}
        if name.endswith((".safetensors", ".pth")):
            stream = io.BufferedReader(RangeFile(url, item["size"]), 1 << 16)
            if name.endswith(".safetensors"):
                content = safetensors_header(read_safetensors_header(stream))
            else:
                content = json.dumps({MARKER: 1, "tree": torch_tree(stream)}).encode()
            entry["kind"] = "header_only"
        else:
            with _request(url) as response:
                content = response.read()
            entry["kind"] = "full"
        with open(path, "wb") as f:
            f.write(content)
        manifest["files"][name] = entry
        print(f"{entry['kind']:>11}  {name}")
    with open(os.path.join(root, "header_only.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(root)


def run(script, args):
    import safetensors.torch
    import torch

    load_file, save_file, load = safetensors.torch.load_file, safetensors.torch.save_file, torch.load

    def meta(dtype, shape):
        return torch.empty(shape, dtype=getattr(torch, dtype), device="meta")

    def header_only_load_file(path, device="cpu"):
        marked, header = is_header_only_safetensors(path)
        if not marked:
            raise ValueError(f"{path}: header-only run read a checkpoint with real weights")
        header.pop("__metadata__")
        return {k: meta(SAFETENSORS_DTYPES[v["dtype"]], v["shape"]) for k, v in header.items()}

    def header_only_save_file(tensors, path, metadata=None):
        names = {v: k for k, v in SAFETENSORS_DTYPES.items()}
        offset, header = 0, {}
        for k, v in tensors.items():
            nbytes = v.numel() * v.element_size()
            header[k] = {"dtype": names[str(v.dtype).removeprefix("torch.")], "shape": list(v.shape),
                         "data_offsets": [offset, offset + nbytes]}
            offset += nbytes
        if metadata:
            header["__metadata__"] = metadata
        with open(path, "wb") as f:
            f.write(safetensors_header(header))

    def header_only_torch_load(path, *a, **kw):
        with open(path, "rb") as f:
            head = f.read(1)
        if head != b"{":
            raise ValueError(f"{path}: header-only run read a checkpoint with real weights")
        with open(path) as f:
            document = json.load(f)
        if document.get(MARKER) != 1:
            raise ValueError(f"{path}: not a header-only torch checkpoint")

        def build(node):
            if isinstance(node, dict) and "__tensor__" in node:
                return meta(*node["__tensor__"])
            if isinstance(node, dict):
                return {k: build(v) for k, v in node.items()}
            if isinstance(node, list):
                return [build(v) for v in node]
            return node
        return build(document["tree"])

    safetensors.torch.load_file = header_only_load_file
    safetensors.torch.save_file = header_only_save_file
    torch.load = header_only_torch_load
    torch.set_default_device("meta")
    sys.argv = [script, *args]
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        safetensors.torch.load_file, safetensors.torch.save_file, torch.load = load_file, save_file, load


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        if len(sys.argv) < 3:
            sys.exit(__doc__)
        return run(sys.argv[2], sys.argv[3:])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("repo")
    f.add_argument("revision")
    f.add_argument("hub")
    f.add_argument("--include", action="append", help=f"file glob (default {' '.join(DEFAULT_INCLUDE)})")
    sub.add_parser("run", help="run EXPORTER.py ARGS... on header-only checkpoints")
    a = parser.parse_args()
    fetch(a.repo, a.revision, a.hub, a.include or DEFAULT_INCLUDE)


if __name__ == "__main__":
    main()
