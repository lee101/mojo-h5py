"""ctypes bridge to the Mojo raw-chunk kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_H5PY_LIB", os.path.join(ROOT, "dist", "libmojo-h5py.so"))

I = ctypes.c_int64
_SIGNATURES = {
    "mh5_shuffle": ([I, I, I, I], None),
    "mh5_unshuffle": ([I, I, I, I], None),
    "mh5_fletcher32": ([I, I], I),
    "mh5_copy_bytes": ([I, I, I], None),
}

_library: ctypes.CDLL | None = None


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "kernels.mojo")
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(source):
        return LIB
    if os.environ.get("MOJO_H5PY_LIB"):
        raise RuntimeError(f"MOJO_H5PY_LIB does not exist or is stale: {LIB}")
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise RuntimeError((proc.stderr or proc.stdout).strip())
    return LIB


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def address(array: np.ndarray) -> int:
    return int(array.ctypes.data)
