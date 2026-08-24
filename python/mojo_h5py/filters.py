"""HDF5-compatible byte shuffle, Fletcher32, and gzip chunk pipelines."""

from __future__ import annotations

import struct
import zlib
from collections.abc import Iterable
from operator import index

import numpy as np

from ._lib import address, lib

FILTER_DEFLATE = 1
FILTER_SHUFFLE = 2
FILTER_FLETCHER32 = 3


def _bytes_array(data: bytes | bytearray | memoryview) -> np.ndarray:
    return np.frombuffer(data, dtype=np.uint8)


def _shuffle_array(
    data: bytes | bytearray | memoryview, itemsize: int, extra: int = 0
) -> np.ndarray:
    source = _bytes_array(data)
    if itemsize < 1 or source.size % itemsize:
        raise ValueError("data length must be divisible by itemsize")
    result = np.empty(source.size + extra, dtype=np.uint8)
    if source.size:
        lib().mh5_shuffle(address(source), address(result), source.size, itemsize)
    return result


def shuffle(data: bytes | bytearray | memoryview, itemsize: int) -> bytes:
    """Apply the HDF5 shuffle filter to complete fixed-width elements."""
    return _shuffle_array(data, itemsize).tobytes()


def _unshuffle_array(data: bytes | bytearray | memoryview, itemsize: int) -> np.ndarray:
    source = _bytes_array(data)
    if itemsize < 1 or source.size % itemsize:
        raise ValueError("data length must be divisible by itemsize")
    result = np.empty(source.size, dtype=np.uint8)
    if source.size:
        lib().mh5_unshuffle(address(source), address(result), source.size, itemsize)
    return result


def _unshuffle_into(
    data: bytes | bytearray | memoryview, itemsize: int, destination: np.ndarray
) -> None:
    source = _bytes_array(data)
    if (
        not isinstance(destination, np.ndarray)
        or not destination.flags.c_contiguous
        or not destination.flags.writeable
    ):
        raise ValueError("destination must be a writable C-contiguous NumPy array")
    target = destination.view(np.uint8).reshape(-1)
    if itemsize < 1 or source.size % itemsize:
        raise ValueError("data length must be divisible by itemsize")
    if source.size != target.size:
        raise OSError(f"decoded chunk has {source.size} bytes, expected {target.size}")
    if source.size:
        lib().mh5_unshuffle(address(source), address(target), source.size, itemsize)


def unshuffle(data: bytes | bytearray | memoryview, itemsize: int) -> bytes:
    """Reverse the HDF5 shuffle filter."""
    return _unshuffle_array(data, itemsize).tobytes()


def fletcher32(data: bytes | bytearray | memoryview) -> int:
    """Return HDF5's Fletcher32 value as ``sum1 << 16 | sum2``."""
    source = _bytes_array(data)
    if not source.size:
        return 0
    return int(lib().mh5_fletcher32(address(source), source.size))


def append_fletcher32(data: bytes | bytearray | memoryview) -> bytes:
    return bytes(_append_fletcher32_array(data))


def _append_fletcher32_array(
    data: bytes | bytearray | memoryview, *, capacity: np.ndarray | None = None
) -> np.ndarray:
    source = _bytes_array(data)
    if capacity is None:
        result = np.empty(source.size + 4, dtype=np.uint8)
        result[:-4] = source
    else:
        result = capacity
    result[-4:] = np.frombuffer(
        struct.pack(">I", fletcher32(result[:-4])), dtype=np.uint8
    )
    return result


def _shuffle8_fletcher32_array(
    data: bytes | bytearray | memoryview,
) -> np.ndarray:
    source = _bytes_array(data)
    if source.size % 8:
        raise ValueError("data length must be divisible by itemsize")
    result = np.empty(source.size + 4, dtype=np.uint8)
    if source.size:
        lib().mh5_shuffle8_fletcher32(address(source), address(result), source.size)
    else:
        result[:] = 0
    return result


def _verified_payload_view(data: bytes | bytearray | memoryview) -> memoryview:
    raw = memoryview(data).cast("B")
    if raw.nbytes < 4:
        raise OSError("Fletcher32 chunk is shorter than its checksum")
    payload = raw[:-4]
    if raw[-4:].tobytes() != struct.pack(">I", fletcher32(payload)):
        raise OSError("Fletcher32 checksum mismatch")
    return payload


def verify_fletcher32(data: bytes | bytearray | memoryview) -> bytes:
    return _verified_payload_view(data).tobytes()


def filter_pipeline(
    *,
    itemsize: int,
    compression: str | None = None,
    compression_opts: int | None = None,
    shuffle: bool = False,
    fletcher32: bool = False,
) -> list[tuple[int, tuple[int, ...]]]:
    """Build the filter order used by h5py for the supported options."""
    try:
        itemsize = index(itemsize)
    except TypeError as error:
        raise TypeError("itemsize must be an integer") from error
    if itemsize < 1:
        raise ValueError("itemsize must be positive")
    pipeline: list[tuple[int, tuple[int, ...]]] = []
    if shuffle:
        pipeline.append((FILTER_SHUFFLE, (itemsize,)))
    if compression is not None:
        if compression != "gzip":
            raise ValueError("only compression='gzip' is supported")
        try:
            level = 4 if compression_opts is None else index(compression_opts)
        except TypeError as error:
            raise TypeError("gzip compression_opts must be an integer") from error
        if not 0 <= level <= 9:
            raise ValueError("gzip compression_opts must be from 0 to 9")
        pipeline.append((FILTER_DEFLATE, (level,)))
    if fletcher32:
        pipeline.append((FILTER_FLETCHER32, ()))
    return pipeline


def encode_pipeline(
    data: bytes | bytearray | memoryview,
    *,
    itemsize: int,
    pipeline: Iterable[tuple[int, tuple[int, ...]]],
    filter_mask: int = 0,
) -> bytes:
    return bytes(
        encode_pipeline_buffer(
            data,
            itemsize=itemsize,
            pipeline=pipeline,
            filter_mask=filter_mask,
        )
    )


def encode_pipeline_buffer(
    data: bytes | bytearray | memoryview,
    *,
    itemsize: int,
    pipeline: Iterable[tuple[int, tuple[int, ...]]],
    filter_mask: int = 0,
) -> bytes | memoryview | np.ndarray:
    raw = memoryview(data).cast("B")
    if filter_mask < 0:
        raise ValueError("filter_mask must be non-negative")
    active = [
        (filter_id, options)
        for index, (filter_id, options) in enumerate(pipeline)
        if not filter_mask & (1 << index)
    ]
    if (
        itemsize == 8
        and len(active) == 2
        and active[0][0] == FILTER_SHUFFLE
        and (not active[0][1] or active[0][1][0] == 8)
        and active[1][0] == FILTER_FLETCHER32
    ):
        return _shuffle8_fletcher32_array(raw)
    position = 0
    while position < len(active):
        filter_id, options = active[position]
        if filter_id == FILTER_SHUFFLE:
            if (
                position + 1 == len(active) - 1
                and active[position + 1][0] == FILTER_FLETCHER32
            ):
                raw = _shuffle_array(
                    raw, options[0] if options else itemsize, extra=4
                )
                return _append_fletcher32_array(raw[:-4], capacity=raw)
            raw = _shuffle_array(raw, options[0] if options else itemsize)
        elif filter_id == FILTER_DEFLATE:
            raw = zlib.compress(raw, options[0] if options else 4)
        elif filter_id == FILTER_FLETCHER32:
            raw = _append_fletcher32_array(raw)
        else:
            raise ValueError(f"unsupported HDF5 filter id {filter_id}")
        position += 1
    return raw


def decode_pipeline(
    data: bytes | bytearray | memoryview,
    *,
    itemsize: int,
    pipeline: Iterable[tuple[int, tuple[int, ...]]],
    filter_mask: int = 0,
) -> bytes:
    raw = memoryview(data).cast("B")
    if filter_mask < 0:
        raise ValueError("filter_mask must be non-negative")
    indexed = list(enumerate(pipeline))
    for index, (filter_id, options) in reversed(indexed):
        if filter_mask & (1 << index):
            continue
        if filter_id == FILTER_FLETCHER32:
            raw = _verified_payload_view(raw)
        elif filter_id == FILTER_DEFLATE:
            try:
                raw = zlib.decompress(raw)
            except zlib.error as error:
                raise OSError(f"gzip chunk decompression failed: {error}") from error
        elif filter_id == FILTER_SHUFFLE:
            raw = _unshuffle_array(raw, options[0] if options else itemsize)
        else:
            raise ValueError(f"unsupported HDF5 filter id {filter_id}")
    return bytes(raw)


def decode_pipeline_into(
    data: bytes | bytearray | memoryview,
    destination: np.ndarray,
    *,
    itemsize: int,
    pipeline: Iterable[tuple[int, tuple[int, ...]]],
    filter_mask: int = 0,
) -> None:
    raw = memoryview(data).cast("B")
    if filter_mask < 0:
        raise ValueError("filter_mask must be non-negative")
    active = [
        (filter_id, options)
        for index, (filter_id, options) in reversed(list(enumerate(pipeline)))
        if not filter_mask & (1 << index)
    ]
    for position, (filter_id, options) in enumerate(active):
        if filter_id == FILTER_FLETCHER32:
            raw = _verified_payload_view(raw)
        elif filter_id == FILTER_DEFLATE:
            try:
                raw = zlib.decompress(raw)
            except zlib.error as error:
                raise OSError(f"gzip chunk decompression failed: {error}") from error
        elif filter_id == FILTER_SHUFFLE:
            if position == len(active) - 1:
                _unshuffle_into(
                    raw, options[0] if options else itemsize, destination
                )
                return
            raw = _unshuffle_array(raw, options[0] if options else itemsize)
        else:
            raise ValueError(f"unsupported HDF5 filter id {filter_id}")
    source = _bytes_array(raw)
    target = destination.view(np.uint8).reshape(-1)
    if source.size != target.size:
        raise OSError(f"decoded chunk has {source.size} bytes, expected {target.size}")
    target[:] = source


def encode_chunk(
    data,
    *,
    compression: str | None = None,
    compression_opts: int | None = None,
    shuffle: bool = False,
    fletcher32: bool = False,
) -> bytes:
    """Encode a C-contiguous NumPy-compatible chunk with h5py filter options."""
    array = np.ascontiguousarray(data)
    if array.dtype.hasobject or array.dtype.fields:
        raise TypeError("object and structured dtypes are not raw-filter compatible")
    pipeline = filter_pipeline(
        itemsize=array.dtype.itemsize,
        compression=compression,
        compression_opts=compression_opts,
        shuffle=shuffle,
        fletcher32=fletcher32,
    )
    return encode_pipeline(
        memoryview(array).cast("B"), itemsize=array.dtype.itemsize, pipeline=pipeline
    )


def decode_chunk(
    data: bytes | bytearray | memoryview,
    shape,
    dtype,
    *,
    compression: str | None = None,
    compression_opts: int | None = None,
    shuffle: bool = False,
    fletcher32: bool = False,
) -> np.ndarray:
    """Decode one filtered chunk into a writable C-contiguous array."""
    dtype = np.dtype(dtype)
    if dtype.hasobject or dtype.fields:
        raise TypeError("object and structured dtypes are not raw-filter compatible")
    try:
        shape = (index(shape),)
    except TypeError:
        try:
            shape = tuple(index(extent) for extent in shape)
        except TypeError as error:
            raise TypeError("shape must contain integers") from error
    if any(extent < 0 for extent in shape):
        raise ValueError("shape dimensions must be non-negative")
    pipeline = filter_pipeline(
        itemsize=dtype.itemsize,
        compression=compression,
        compression_opts=compression_opts,
        shuffle=shuffle,
        fletcher32=fletcher32,
    )
    result = np.empty(shape, dtype=dtype)
    decode_pipeline_into(
        data, result, itemsize=dtype.itemsize, pipeline=pipeline
    )
    return result
