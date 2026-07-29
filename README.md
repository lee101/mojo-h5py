# mojo-h5py

`mojo-h5py` is a focused port of h5py's HDF5 raw-chunk filter path to Mojo. It
writes standard HDF5 chunks, and the test suite verifies that upstream h5py can
read files written through this package and vice versa.

Use it as `import mojo_h5py as h5py`. For the covered subset, `File`, `Group`,
and `Dataset` retain familiar h5py names and common calling patterns. h5py still
handles the HDF5 container, metadata, and unsupported operations; Mojo handles
byte shuffle, Fletcher32, and whole-array chunk processing.

## Coverage

Implemented and parity-tested against h5py 3.16.0 / HDF5 2.1.0:

- chunked, fixed-width numeric datasets with one or more dimensions;
- whole-dataset reads and writes, including padded edge chunks and fill values;
- `Dataset.read_direct`, `write_direct`, and `iter_chunks`;
- HDF5 byte shuffle and Fletcher32 with byte-identical raw output;
- gzip levels 0–9 in the standard shuffle, deflate, Fletcher32 pipeline;
- big- and little-endian dtypes, groups, attributes, resizing, and partial
  selections;
- transparent fallback to h5py for contiguous and structured datasets.

Partial selections, metadata operations, and unsupported dtypes deliberately
delegate to h5py. LZF, SZIP, N-bit, scale-offset, custom filter plugins, virtual
datasets, external storage, SWMR, MPI, dimension scales, references beyond the
re-exported basic types, and h5py's low-level API are not ported. gzip uses the
system zlib implementation; the shuffle and Fletcher32 loops are Mojo kernels.

## Install

The pinned Mojo nightly and all Python dependencies are managed by Pixi:

```bash
pixi install
pixi run build
pixi run test
```

The build produces `dist/libmojo-h5py.so`. Imports rebuild it when sources are
newer, or `MOJO_H5PY_LIB` can point to a prebuilt shared library.

## Usage

```python
import numpy as np
import mojo_h5py as h5py

values = np.arange(120, dtype=np.float64).reshape(10, 12)

with h5py.File("example.h5", "w") as file:
    dataset = file.create_dataset(
        "values",
        data=values,
        chunks=(4, 5),
        compression="gzip",
        compression_opts=4,
        shuffle=True,
        fletcher32=True,
    )
    assert np.array_equal(dataset[:], values)

with h5py.File("example.h5", "r") as file:
    restored = file["values"][:]
```

Raw filter helpers are also public:

```python
encoded = h5py.encode_chunk(values, shuffle=True, fletcher32=True)
decoded = h5py.decode_chunk(
    encoded, values.shape, values.dtype, shuffle=True, fletcher32=True
)
assert np.array_equal(decoded, values)
```

## Benchmarks

Measured with `pixi run bench` on 2026-07-29 using an Intel Xeon E5-2697 v4 at
2.30 GHz, Linux 6.8.0-136-generic. Each case processes a 16 MiB float64 array
with 256 x 256 chunks; numbers are the best of three end-to-end file operations.
"Mojo speed" is h5py time divided by mojo-h5py time, so values below 1.00x mean
h5py is faster.

| case | mojo-h5py | h5py | Mojo speed |
|---|---:|---:|---:|
| write: shuffle + Fletcher32 | 38.3 ms | 58.0 ms | 1.52x |
| read: shuffle + Fletcher32 | 17.8 ms | 30.9 ms | 1.73x |
| write: gzip-1 + shuffle + Fletcher32 | 222.0 ms | 775.0 ms | 3.49x |
| read: gzip-1 + shuffle + Fletcher32 | 16.5 ms | 51.9 ms | 3.14x |

The float64 shuffle/unshuffle and Fletcher32 kernels use SIMD with unaligned-safe
loads and scalar tails. Datasets of at least 4 MiB process independent chunks
with four workers; smaller operations remain serial. Fused shuffle/checksum
buffers and direct decode into NumPy-owned arrays avoid full-chunk copies, and
parallel work is bounded to eight in-flight chunks.

No GPU path is provided. Shuffle and Fletcher32 are memory-bound, while gzip is
a branch-heavy zlib operation rather than a high-intensity numeric kernel.

## How it works

Python obtains raw chunks through `H5Dread_chunk` and `H5Dwrite_chunk`, exposed
by h5py as `read_direct_chunk` and `write_direct_chunk`. Encoding applies
shuffle, zlib deflate, then Fletcher32; decoding reverses that order and honors
HDF5's per-chunk filter mask. Edge chunks are expanded to their full declared
chunk shape with the dataset fill value, matching HDF5 storage semantics.

The shared library exposes a C ABI through four non-parametric exports. NumPy
owns every input and output allocation. Buffers cross `ctypes` as integer
addresses and Mojo reconstructs mutable `UInt8` pointers with
`AnyOrigin[mut=True]`; no allocation or ownership crosses the FFI boundary.
Chunks are C-contiguous bytes, while the dtype preserves its declared byte
order for exact on-disk compatibility.
