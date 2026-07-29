"""End-to-end chunked HDF5 benchmarks against h5py on identical data."""

from __future__ import annotations

import math
import os
import platform
import sys
import tempfile
import time

import h5py
import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import mojo_h5py as mh  # noqa: E402


def timeit(function, repeat: int = 3) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def write_file(api, path: str, data: np.ndarray, options: dict) -> None:
    with api.File(path, "w") as file:
        file.create_dataset("values", data=data, chunks=(256, 256), **options)


def read_file(api, path: str) -> np.ndarray:
    with api.File(path, "r") as file:
        return file["values"][:]


def main() -> None:
    rows = []
    rng = np.random.default_rng(2026)
    data = np.ascontiguousarray(rng.normal(size=(2048, 1024)))
    cases = [
        ("shuffle + Fletcher32", {"shuffle": True, "fletcher32": True}),
        (
            "gzip-1 + shuffle + Fletcher32",
            {
                "compression": "gzip",
                "compression_opts": 1,
                "shuffle": True,
                "fletcher32": True,
            },
        ),
    ]

    with tempfile.TemporaryDirectory(prefix="mojo-h5py-bench-") as directory:
        for label, options in cases:
            mojo_path = os.path.join(directory, f"mojo-{len(rows)}.h5")
            h5py_path = os.path.join(directory, f"h5py-{len(rows)}.h5")

            write_file(mh, mojo_path, data, options)
            write_file(h5py, h5py_path, data, options)

            mojo_write = timeit(lambda: write_file(mh, mojo_path, data, options))
            h5py_write = timeit(lambda: write_file(h5py, h5py_path, data, options))
            rows.append((f"write: {label}", mojo_write, h5py_write))

            assert np.array_equal(read_file(mh, h5py_path), data)
            mojo_read = timeit(lambda: read_file(mh, h5py_path))
            h5py_read = timeit(lambda: read_file(h5py, h5py_path))
            rows.append((f"read: {label}", mojo_read, h5py_read))

    print(f"Machine: {cpu_name()}, {platform.system()} {platform.release()}")
    print(f"Data: {data.nbytes / (1024 * 1024):.0f} MiB float64, chunks 256 x 256")
    print()
    print("| case | mojo-h5py | h5py | Mojo speed |")
    print("|---|---:|---:|---:|")
    for label, mojo_time, h5py_time in rows:
        print(
            f"| {label} | {mojo_time * 1e3:.1f} ms | {h5py_time * 1e3:.1f} ms "
            f"| {h5py_time / mojo_time:.2f}x |"
        )


if __name__ == "__main__":
    main()
