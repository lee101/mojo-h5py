"""HDF5 chunked I/O and filters accelerated with Mojo."""

from h5py import Empty, Reference, RegionReference, special_dtype, string_dtype, vlen_dtype
from h5py.version import version as __h5py_version__

from .filters import (
    append_fletcher32,
    decode_chunk,
    encode_chunk,
    fletcher32,
    shuffle,
    unshuffle,
    verify_fletcher32,
)
from .highlevel import Dataset, File, Group

__version__ = "0.1.0"

__all__ = [
    "Dataset",
    "Empty",
    "File",
    "Group",
    "Reference",
    "RegionReference",
    "append_fletcher32",
    "decode_chunk",
    "encode_chunk",
    "fletcher32",
    "shuffle",
    "special_dtype",
    "string_dtype",
    "unshuffle",
    "verify_fletcher32",
    "vlen_dtype",
]
