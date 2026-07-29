"""A focused h5py-compatible high-level API with Mojo raw-chunk paths."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from operator import index

import h5py as _h5py
import numpy as np

from .filters import (
    FILTER_DEFLATE,
    FILTER_FLETCHER32,
    FILTER_SHUFFLE,
    decode_pipeline_into,
    encode_pipeline_buffer,
)

_SUPPORTED_FILTERS = {FILTER_DEFLATE, FILTER_SHUFFLE, FILTER_FLETCHER32}
_PARALLEL_MIN_BYTES = 4 * 1024 * 1024
_PARALLEL_WORKERS = 4


def _full_selection(key, ndim: int) -> bool:
    if key is Ellipsis:
        return True
    if isinstance(key, slice):
        return key == slice(None)
    if isinstance(key, tuple):
        if not key:
            return True
        ellipses = sum(part is Ellipsis for part in key)
        explicit_axes = len(key) - ellipses
        return (
            ellipses <= 1
            and explicit_axes <= ndim
            and all(part is Ellipsis or part == slice(None) for part in key)
        )
    return False


def _pipeline(dataset) -> list[tuple[int, tuple[int, ...]]]:
    plist = dataset.id.get_create_plist()
    return [
        (int(info[0]), tuple(int(value) for value in info[2]))
        for info in (plist.get_filter(index) for index in range(plist.get_nfilters()))
    ]


def _raw_supported(dataset) -> bool:
    if dataset.chunks is None or dataset.dtype.hasobject or dataset.dtype.fields:
        return False
    return all(filter_id in _SUPPORTED_FILTERS for filter_id, _ in _pipeline(dataset))


def _chunk_offsets(shape, chunks):
    axes = [range(0, extent, chunk) for extent, chunk in zip(shape, chunks)]
    return product(*axes)


def _chunk_slices(offset, shape, chunks):
    return tuple(
        slice(start, min(start + chunk, extent))
        for start, extent, chunk in zip(offset, shape, chunks)
    )


def _parallel_chunks(nbytes: int, count: int) -> bool:
    return nbytes >= _PARALLEL_MIN_BYTES and count >= _PARALLEL_WORKERS


def _parallel_map(function, iterable):
    iterator = iter(iterable)
    with ThreadPoolExecutor(max_workers=_PARALLEL_WORKERS) as executor:
        pending = deque()
        for _ in range(2 * _PARALLEL_WORKERS):
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                pass


class Dataset:
    """Proxy for :class:`h5py.Dataset` with accelerated whole-array chunk I/O."""

    def __init__(self, dataset):
        self._dataset = dataset

    def _read_all(self) -> np.ndarray:
        dataset = self._dataset
        if not _raw_supported(dataset):
            return dataset[...]
        result = np.empty(dataset.shape, dtype=dataset.dtype)
        pipeline = _pipeline(dataset)
        chunk_shape = tuple(dataset.chunks)
        offsets = list(_chunk_offsets(dataset.shape, chunk_shape))

        def read_encoded(offset):
            slices = _chunk_slices(offset, dataset.shape, chunk_shape)
            if dataset.id.get_chunk_info_by_coord(offset).size == 0:
                return slices, None, 0
            filter_mask, encoded = dataset.id.read_direct_chunk(offset)
            return slices, encoded, filter_mask

        def decode(encoded_chunk):
            slices, encoded, filter_mask = encoded_chunk
            if encoded is None:
                return slices, None
            chunk = np.empty(chunk_shape, dtype=dataset.dtype)
            decode_pipeline_into(
                encoded,
                chunk,
                itemsize=dataset.dtype.itemsize,
                pipeline=pipeline,
                filter_mask=filter_mask,
            )
            return slices, chunk

        if _parallel_chunks(dataset.nbytes, len(offsets)):
            encoded_chunks = (read_encoded(offset) for offset in offsets)
            decoded_chunks = _parallel_map(decode, encoded_chunks)
        else:
            decoded_chunks = (
                decode(read_encoded(offset)) for offset in offsets
            )
        for slices, chunk in decoded_chunks:
            if chunk is None:
                result[slices] = dataset.fillvalue
                continue
            local = tuple(slice(0, part.stop - part.start) for part in slices)
            result[slices] = chunk[local]
        return result

    def _write_all(self, value) -> None:
        dataset = self._dataset
        if not _raw_supported(dataset):
            dataset[...] = value
            return
        source = np.asarray(value, dtype=dataset.dtype)
        if source.shape != dataset.shape:
            source = np.broadcast_to(source, dataset.shape)
        pipeline = _pipeline(dataset)
        chunk_shape = tuple(dataset.chunks)
        offsets = list(_chunk_offsets(dataset.shape, chunk_shape))
        fillvalue = dataset.fillvalue

        def encode(offset):
            slices = _chunk_slices(offset, dataset.shape, chunk_shape)
            local = tuple(slice(0, part.stop - part.start) for part in slices)
            if all(part.stop - part.start == size for part, size in zip(slices, chunk_shape)):
                chunk = np.ascontiguousarray(source[slices])
            else:
                chunk = np.full(chunk_shape, fillvalue, dtype=dataset.dtype)
                chunk[local] = source[slices]
            encoded = encode_pipeline_buffer(
                memoryview(chunk).cast("B"),
                itemsize=dataset.dtype.itemsize,
                pipeline=pipeline,
            )
            return offset, encoded

        if _parallel_chunks(dataset.nbytes, len(offsets)):
            encoded_chunks = _parallel_map(encode, offsets)
        else:
            encoded_chunks = map(encode, offsets)
        for offset, encoded in encoded_chunks:
            dataset.id.write_direct_chunk(offset, encoded, filter_mask=0)

    def __getitem__(self, key):
        if _full_selection(key, self.ndim):
            return self._read_all()
        return self._dataset[key]

    def __setitem__(self, key, value):
        if _full_selection(key, self.ndim):
            self._write_all(value)
        else:
            self._dataset[key] = value

    def __array__(self, dtype=None, copy=None):
        array = self._read_all()
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        if copy:
            array = array.copy()
        return array

    def __len__(self):
        return len(self._dataset)

    def __iter__(self):
        return iter(self._dataset)

    def __getattr__(self, name):
        return getattr(self._dataset, name)

    def read_direct(self, dest, source_sel=None, dest_sel=None):
        if (
            source_sel is None
            and dest_sel is None
            and isinstance(dest, np.ndarray)
            and dest.shape == self.shape
            and dest.dtype == self.dtype
            and dest.flags.c_contiguous
            and dest.flags.writeable
        ):
            np.copyto(dest, self._read_all(), casting="no")
            return None
        return self._dataset.read_direct(dest, source_sel=source_sel, dest_sel=dest_sel)

    def write_direct(self, source, source_sel=None, dest_sel=None):
        if source_sel is None and dest_sel is None and np.shape(source) == self.shape:
            self._write_all(source)
            return None
        return self._dataset.write_direct(source, source_sel=source_sel, dest_sel=dest_sel)

    def __repr__(self):
        return repr(self._dataset)


def _wrap(value):
    if isinstance(value, _h5py.Dataset):
        return Dataset(value)
    if isinstance(value, _h5py.Group):
        return Group(value)
    return value


class Group:
    """Proxy for :class:`h5py.Group` preserving dataset wrapping."""

    def __init__(self, group):
        self._group = group

    def __getitem__(self, name):
        return _wrap(self._group[name])

    def __contains__(self, name):
        return name in self._group

    def __iter__(self):
        return iter(self._group)

    def __len__(self):
        return len(self._group)

    def __getattr__(self, name):
        return getattr(self._group, name)

    def create_group(self, name, track_order=None):
        kwargs = {} if track_order is None else {"track_order": track_order}
        return Group(self._group.create_group(name, **kwargs))

    def require_group(self, name):
        return Group(self._group.require_group(name))

    def create_dataset(self, name, shape=None, dtype=None, data=None, **kwds):
        if data is None:
            return Dataset(self._group.create_dataset(name, shape=shape, dtype=dtype, **kwds))
        array = np.asarray(data, dtype=dtype)
        if shape is None:
            target_shape = array.shape
        else:
            try:
                target_shape = (index(shape),)
            except TypeError:
                target_shape = tuple(shape)
        target_dtype = array.dtype if dtype is None else np.dtype(dtype)
        dataset = Dataset(
            self._group.create_dataset(
                name, shape=target_shape, dtype=target_dtype, data=None, **kwds
            )
        )
        dataset[...] = array
        return dataset

    def require_dataset(self, name, shape, dtype, exact=False, **kwds):
        return Dataset(
            self._group.require_dataset(name, shape, dtype, exact=exact, **kwds)
        )


class File(Group):
    """Drop-in covered subset of :class:`h5py.File`."""

    def __init__(self, name, mode=None, driver=None, libver=None, userblock_size=None, **kwds):
        args = [name]
        if mode is not None:
            args.append(mode)
        file = _h5py.File(
            *args,
            driver=driver,
            libver=libver,
            userblock_size=userblock_size,
            **kwds,
        )
        super().__init__(file)
        self._file = file

    def __enter__(self):
        self._file.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._file.__exit__(exc_type, exc_value, traceback)

    def close(self):
        return self._file.close()

    def __repr__(self):
        return repr(self._file)
