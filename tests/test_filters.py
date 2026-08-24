import struct
import tempfile

import h5py
import numpy as np
import pytest

import mojo_h5py as mh
from mojo_h5py.filters import decode_pipeline, decode_pipeline_into


def reference_shuffle(data: bytes, itemsize: int) -> bytes:
    array = np.frombuffer(data, dtype=np.uint8).reshape(-1, itemsize)
    return array.T.copy().tobytes()


def reference_fletcher32(data: bytes) -> int:
    sum1 = 0
    sum2 = 0
    words = (len(data) + 1) // 2
    for block_start in range(0, words, 360):
        for word_index in range(block_start, min(block_start + 360, words)):
            byte_index = 2 * word_index
            word = data[byte_index]
            if byte_index + 1 < len(data):
                word |= data[byte_index + 1] << 8
            sum1 += word
            sum2 += sum1
        sum1 = (sum1 & 0xFFFF) + (sum1 >> 16)
        sum2 = (sum2 & 0xFFFF) + (sum2 >> 16)
    sum1 = (sum1 & 0xFFFF) + (sum1 >> 16)
    sum2 = (sum2 & 0xFFFF) + (sum2 >> 16)
    return ((sum1 & 0xFFFF) << 16) | (sum2 & 0xFFFF)


@pytest.mark.parametrize("itemsize", [1, 2, 4, 8, 16])
def test_shuffle_matches_reference(itemsize):
    data = np.arange(127 * itemsize, dtype=np.uint8).tobytes()
    expected = reference_shuffle(data, itemsize)
    assert mh.shuffle(data, itemsize) == expected
    assert mh.unshuffle(expected, itemsize) == data


def test_float64_shuffle_simd_tail_accepts_unaligned_input():
    data = bytes((index * 37 + 11) & 0xFF for index in range(37 * 8))
    unaligned = memoryview(b"x" + data)[1:]
    expected = reference_shuffle(data, 8)
    assert mh.shuffle(unaligned, 8) == expected
    assert mh.unshuffle(memoryview(b"x" + expected)[1:], 8) == data


@pytest.mark.parametrize("size", [2, 6, 8, 10, 37, 38])
def test_fused_float64_shuffle_fletcher32_simd_tail(size):
    array = np.arange(size, dtype=np.float64) / 7.0
    expected = mh.append_fletcher32(reference_shuffle(array.tobytes(), 8))
    assert mh.encode_chunk(array, shuffle=True, fletcher32=True) == expected


def test_empty_buffers_do_not_cross_ffi_as_null_pointers():
    assert mh.shuffle(b"", 8) == b""
    assert mh.unshuffle(b"", 8) == b""
    assert mh.fletcher32(b"") == 0


@pytest.mark.parametrize("size", [31, 32, 33, 719, 720, 721, 733])
def test_fletcher32_simd_and_block_tails(size):
    data = bytes((index * 29 + 7) & 0xFF for index in range(size))
    assert mh.fletcher32(data) == reference_fletcher32(data)


@pytest.mark.parametrize("bad_itemsize", [0, -1, 3])
def test_shuffle_rejects_invalid_element_boundaries(bad_itemsize):
    with pytest.raises(ValueError):
        mh.shuffle(b"01234567", bad_itemsize)


@pytest.mark.parametrize("dtype", ["u1", "<u2", ">u2", "<u4", ">u4", "<f8", ">f8"])
@pytest.mark.parametrize("size", [7, 8])
def test_fletcher32_bytes_match_hdf5(tmp_path, dtype, size):
    array = np.arange(size, dtype=dtype)
    path = tmp_path / "checksum.h5"
    with h5py.File(path, "w") as file:
        dataset = file.create_dataset(
            "values", data=array, chunks=(size,), fletcher32=True
        )
        _, encoded = dataset.id.read_direct_chunk((0,))
    assert mh.append_fletcher32(array.tobytes()) == encoded
    assert mh.verify_fletcher32(encoded) == array.tobytes()


def test_fletcher32_published_simple_vector():
    data = bytes(range(8))
    assert mh.fletcher32(data) == 0x100C1E14
    assert mh.append_fletcher32(data).hex() == "0001020304050607100c1e14"


def test_fletcher32_detects_corruption():
    encoded = bytearray(mh.append_fletcher32(bytes(range(31))))
    encoded[9] ^= 0x80
    with pytest.raises(OSError, match="checksum mismatch"):
        mh.verify_fletcher32(encoded)
    with pytest.raises(OSError, match="shorter"):
        mh.verify_fletcher32(b"abc")


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"shuffle": True},
        {"fletcher32": True},
        {"compression": "gzip", "compression_opts": 1},
        {
            "compression": "gzip",
            "compression_opts": 6,
            "shuffle": True,
            "fletcher32": True,
        },
    ],
)
def test_encoded_chunk_is_byte_identical_to_hdf5(tmp_path, options):
    array = np.arange(96, dtype="<i4").reshape(12, 8)
    path = tmp_path / "pipeline.h5"
    with h5py.File(path, "w") as file:
        dataset = file.create_dataset(
            "values", data=array, chunks=array.shape, **options
        )
        _, expected = dataset.id.read_direct_chunk((0, 0))
    assert mh.encode_chunk(array, **options) == expected
    assert np.array_equal(
        mh.decode_chunk(expected, array.shape, array.dtype, **options), array
    )


@pytest.mark.parametrize("level", range(10))
def test_all_claimed_gzip_levels_match_hdf5(tmp_path, level):
    array = np.arange(257, dtype=np.float64)
    path = tmp_path / f"gzip-{level}.h5"
    options = {"compression": "gzip", "compression_opts": level}
    with h5py.File(path, "w") as file:
        dataset = file.create_dataset("values", data=array, chunks=array.shape, **options)
        _, expected = dataset.id.read_direct_chunk((0,))
    assert mh.encode_chunk(array, **options) == expected


def test_decode_honors_filter_mask():
    array = np.arange(32, dtype="<u2")
    raw = mh.append_fletcher32(array.tobytes())
    pipeline = [(2, (2,)), (1, (6,)), (3, ())]
    decoded = decode_pipeline(
        raw,
        itemsize=2,
        pipeline=pipeline,
        filter_mask=(1 << 0) | (1 << 1),
    )
    assert decoded == array.tobytes()


def test_filter_option_validation():
    array = np.arange(5)
    with pytest.raises(ValueError, match="only compression"):
        mh.encode_chunk(array, compression="lzf")
    with pytest.raises(ValueError, match="0 to 9"):
        mh.encode_chunk(array, compression="gzip", compression_opts=10)
    with pytest.raises(TypeError, match="integer"):
        mh.encode_chunk(array, compression="gzip", compression_opts=1.5)
    with pytest.raises(TypeError, match="raw-filter compatible"):
        mh.encode_chunk(np.array([object()], dtype=object))


def test_decode_rejects_wrong_shape():
    array = np.arange(12, dtype=np.int16)
    encoded = mh.encode_chunk(array, shuffle=True)
    with pytest.raises(OSError, match="expected"):
        mh.decode_chunk(encoded, (13,), np.int16, shuffle=True)


def test_decode_into_rejects_noncontiguous_destination():
    destination = np.empty((4, 4), dtype=np.uint8)[:, ::2]
    with pytest.raises(ValueError, match="C-contiguous"):
        decode_pipeline_into(
            bytes(destination.size),
            destination,
            itemsize=1,
            pipeline=[(2, (1,))],
        )
