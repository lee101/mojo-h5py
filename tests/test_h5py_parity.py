import h5py
import numpy as np
import pytest

import mojo_h5py as mh
from mojo_h5py import highlevel


FILTER_CASES = [
    {},
    {"shuffle": True},
    {"fletcher32": True},
    {"compression": "gzip", "compression_opts": 1},
    {
        "shuffle": True,
        "compression": "gzip",
        "compression_opts": 6,
        "fletcher32": True,
    },
]


@pytest.mark.parametrize("options", FILTER_CASES)
def test_upstream_reads_mojo_written_edge_chunks(tmp_path, options):
    array = (np.arange(13 * 17, dtype=np.float64).reshape(13, 17) / 7.0)[:, ::-1]
    path = tmp_path / "ours.h5"
    with mh.File(path, "w") as file:
        dataset = file.create_dataset(
            "values", data=array, chunks=(5, 7), fillvalue=-1.0, **options
        )
        assert np.array_equal(dataset[...], array)
    with h5py.File(path, "r") as file:
        assert np.array_equal(file["values"][...], array)
        assert file["values"].chunks == (5, 7)


@pytest.mark.parametrize("options", FILTER_CASES)
def test_mojo_reads_upstream_written_edge_chunks(tmp_path, options):
    rng = np.random.default_rng(4)
    array = rng.integers(-1000, 1000, size=(11, 19), dtype=np.int32)
    path = tmp_path / "upstream.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("values", data=array, chunks=(4, 6), **options)
    with mh.File(path, "r") as file:
        dataset = file["values"]
        assert isinstance(dataset, mh.Dataset)
        assert np.array_equal(dataset[:], array)
        assert np.array_equal(np.asarray(dataset), array)


def test_big_endian_dtype_round_trip(tmp_path):
    array = np.arange(143, dtype=">i4").reshape(11, 13)
    path = tmp_path / "endian.h5"
    with mh.File(path, "w") as file:
        file.create_dataset(
            "values",
            data=array,
            chunks=(4, 5),
            shuffle=True,
            compression="gzip",
            fletcher32=True,
        )
    with h5py.File(path, "r") as file:
        actual = file["values"][:]
        assert actual.dtype == array.dtype
        assert np.array_equal(actual, array)


def test_three_dimensional_dataset_round_trip(tmp_path):
    array = np.arange(7 * 5 * 9, dtype=np.uint16).reshape(7, 5, 9)
    path = tmp_path / "three-dimensional.h5"
    with mh.File(path, "w") as file:
        file.create_dataset(
            "values",
            data=array,
            chunks=(3, 2, 4),
            shuffle=True,
            fletcher32=True,
        )
    with h5py.File(path, "r") as file:
        assert np.array_equal(file["values"][:], array)


def test_partial_selection_falls_back_with_upstream_semantics(tmp_path):
    array = np.arange(120, dtype=np.int64).reshape(10, 12)
    path = tmp_path / "selection.h5"
    with mh.File(path, "w") as file:
        dataset = file.create_dataset(
            "values", data=array, chunks=(3, 5), shuffle=True, fletcher32=True
        )
        dataset[2:8:2, 3:10] = -7
        array[2:8:2, 3:10] = -7
        assert np.array_equal(dataset[1:9, ::3], array[1:9, ::3])
    with h5py.File(path, "r") as file:
        assert np.array_equal(file["values"][:], array)


def test_invalid_multiple_ellipsis_is_not_accepted_as_full_selection(tmp_path):
    path = tmp_path / "invalid-selection.h5"
    with mh.File(path, "w") as file:
        dataset = file.create_dataset(
            "values", data=np.arange(12).reshape(3, 4), chunks=(2, 2)
        )
        with pytest.raises(ValueError, match="Only one ellipsis"):
            _ = dataset[..., ...]


def test_scalar_broadcast_whole_dataset_write(tmp_path):
    path = tmp_path / "broadcast.h5"
    with mh.File(path, "w") as file:
        dataset = file.create_dataset(
            "values",
            shape=(9, 10),
            dtype="f4",
            chunks=(4, 6),
            compression="gzip",
            shuffle=True,
        )
        dataset[...] = 2.5
    with h5py.File(path, "r") as file:
        assert np.array_equal(file["values"][:], np.full((9, 10), 2.5, dtype="f4"))


def test_integer_shape_with_data_matches_h5py(tmp_path):
    path = tmp_path / "integer-shape.h5"
    with mh.File(path, "w") as file:
        dataset = file.create_dataset(
            "values", shape=6, data=np.arange(6), chunks=3
        )
        assert np.array_equal(dataset[:], np.arange(6))


def test_unallocated_chunks_use_fill_value(tmp_path):
    path = tmp_path / "fill.h5"
    with h5py.File(path, "w") as file:
        dataset = file.create_dataset(
            "values",
            shape=(12, 12),
            dtype="i2",
            chunks=(4, 4),
            fillvalue=-9,
            shuffle=True,
        )
        dataset[0:4, 0:4] = 3
    with mh.File(path, "r") as file:
        expected = np.full((12, 12), -9, dtype="i2")
        expected[0:4, 0:4] = 3
        assert np.array_equal(file["values"][:], expected)


def test_read_direct_and_write_direct(tmp_path):
    source = np.arange(99, dtype=np.float64).reshape(9, 11)
    destination = np.empty_like(source)
    path = tmp_path / "direct.h5"
    with mh.File(path, "w") as file:
        dataset = file.create_dataset(
            "values",
            shape=source.shape,
            dtype=source.dtype,
            chunks=(4, 5),
            fletcher32=True,
        )
        assert dataset.write_direct(source) is None
        assert dataset.read_direct(destination) is None
    assert np.array_equal(destination, source)


def test_read_direct_dtype_conversion_delegates_to_h5py(tmp_path):
    source = np.arange(24, dtype=np.int64).reshape(4, 6)
    destination = np.empty(source.shape, dtype=np.int16)
    path = tmp_path / "direct-conversion.h5"
    with mh.File(path, "w") as file:
        dataset = file.create_dataset(
            "values", data=source, chunks=(2, 3), shuffle=True
        )
        dataset.read_direct(destination)
    assert np.array_equal(destination, source.astype(np.int16))


def test_allocated_chunk_read_failure_is_not_treated_as_fill(tmp_path):
    path = tmp_path / "corrupt.h5"
    with h5py.File(path, "w") as file:
        dataset = file.create_dataset(
            "values",
            data=np.arange(16, dtype=np.int32),
            chunks=(16,),
            fletcher32=True,
        )
        mask, encoded = dataset.id.read_direct_chunk((0,))
        corrupt = bytearray(encoded)
        corrupt[0] ^= 1
        dataset.id.write_direct_chunk((0,), corrupt, filter_mask=mask)
    with mh.File(path, "r") as file:
        with pytest.raises(OSError, match="checksum mismatch"):
            file["values"][:]


def test_groups_and_iter_chunks_match_h5py(tmp_path):
    path = tmp_path / "groups.h5"
    with mh.File(path, "w") as file:
        group = file.create_group("nested")
        dataset = group.create_dataset(
            "values", data=np.arange(35).reshape(5, 7), chunks=(3, 4)
        )
        ours = list(dataset.iter_chunks())
        assert "nested" in file
        assert isinstance(file["nested"], mh.Group)
    with h5py.File(path, "r") as file:
        assert ours == list(file["nested/values"].iter_chunks())


def test_contiguous_dataset_uses_compatible_fallback(tmp_path):
    array = np.arange(20, dtype=np.float64)
    path = tmp_path / "contiguous.h5"
    with mh.File(path, "w") as file:
        dataset = file.create_dataset("values", data=array)
        assert dataset.chunks is None
        assert np.array_equal(dataset[:], array)


def test_structured_dtype_uses_compatible_fallback(tmp_path):
    dtype = np.dtype([("x", "<i4"), ("y", "<f8")])
    array = np.zeros(20, dtype=dtype)
    array["x"] = np.arange(20)
    array["y"] = np.arange(20) / 3
    path = tmp_path / "structured.h5"
    with mh.File(path, "w") as file:
        dataset = file.create_dataset(
            "values", data=array, chunks=(7,), shuffle=True, fletcher32=True
        )
        assert np.array_equal(dataset[:], array)


def test_metadata_and_resize_delegate(tmp_path):
    path = tmp_path / "resize.h5"
    with mh.File(path, "w") as file:
        dataset = file.create_dataset(
            "values", data=np.arange(6), chunks=(4,), maxshape=(None,)
        )
        dataset.attrs["units"] = "count"
        dataset.resize((10,))
        dataset[6:] = np.arange(4) + 6
        assert dataset.attrs["units"] == "count"
        assert dataset.shape == (10,)
        assert np.array_equal(dataset[:], np.arange(10))


def test_parallel_chunk_threshold_boundary(monkeypatch):
    monkeypatch.setattr(highlevel, "_PARALLEL_MIN_BYTES", 1024)
    assert not highlevel._parallel_chunks(1023, highlevel._PARALLEL_WORKERS)
    assert highlevel._parallel_chunks(1024, highlevel._PARALLEL_WORKERS)
    assert not highlevel._parallel_chunks(1024, highlevel._PARALLEL_WORKERS - 1)


@pytest.mark.parametrize("parallel", [False, True])
def test_serial_and_parallel_chunk_paths(tmp_path, monkeypatch, parallel):
    real_executor = highlevel.ThreadPoolExecutor
    executor_calls = []

    def tracking_executor(*args, **kwargs):
        executor_calls.append(kwargs.get("max_workers"))
        return real_executor(*args, **kwargs)

    monkeypatch.setattr(highlevel, "ThreadPoolExecutor", tracking_executor)
    monkeypatch.setattr(
        highlevel, "_PARALLEL_MIN_BYTES", 1 if parallel else 1 << 60
    )
    array = np.arange(65 * 67, dtype=np.float64).reshape(65, 67)
    path = tmp_path / f"parallel-{parallel}.h5"
    with mh.File(path, "w") as file:
        dataset = file.create_dataset(
            "values",
            data=array,
            chunks=(16, 17),
            shuffle=True,
            fletcher32=True,
        )
        assert np.array_equal(dataset[:], array)
    assert bool(executor_calls) is parallel
