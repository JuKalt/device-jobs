"""Snapshot explicitly selected files from a job project."""

import base64
import fnmatch
import hashlib
from pathlib import Path, PurePosixPath
import subprocess


def snapshot(root, include):
    root = Path(root).resolve()
    listed = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--cached"],
                            stdout=subprocess.PIPE, check=True).stdout
    selected = set()
    for pattern in include:
        matches = {name for name in listed.decode().split("\0") if name
                   and fnmatch.fnmatchcase(name, pattern)}
        if not matches:
            raise ValueError(f"Source include matches no tracked files: {pattern}")
        selected.update(matches)
    files = {}
    for name in sorted(selected):
        relative = PurePosixPath(name)
        path = root / name
        if (relative.is_absolute() or any(part.startswith(".") for part in relative.parts)
                or path.resolve() != path or not path.is_file()):
            raise ValueError(f"Unsafe or missing source file: {name}")
        data = path.read_bytes()
        files[name] = dict(sha256=hashlib.sha256(data).hexdigest(), mode=path.stat().st_mode & 0o777,
                           data=base64.b64encode(data).decode("ascii"))
    manifest = {name: {k: v for k, v in value.items() if k != "data"} for name, value in files.items()}
    digest = hashlib.sha256(repr(sorted((name, value["sha256"], value["mode"])
                                      for name, value in manifest.items())).encode()).hexdigest()
    return dict(files=files, manifest=manifest, digest=digest)
