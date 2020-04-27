import pytest
import numpy as np
import dask.array as da
import numpy.testing as npt
import h5py
# import math
# import functools
import os

from pyxrf.core.map_processing import (
    _compute_optimal_chunk_size, _chunk_numpy_array, _array_numpy_to_dask,
    RawHDF5Dataset, compute_total_spectrum)

'''
def test_map_averaged_counts():
    """Basic functionality of 'map_averaged_counts'"""

    # Zero-dimensional input data
    data = 10.0
    total_count_expected, spectrum_expected = data, data
    total_count, spectrum = map_total_counts(data)
    npt.assert_almost_equal(total_count, total_count_expected,
                            err_msg="scalar: total count maps don't match")
    npt.assert_almost_equal(spectrum, spectrum_expected,
                            err_msg="scalar: averaged spectra don't match")

    # One dimensional input data
    data = np.random.rand(10)
    total_count_expected, spectrum_expected = np.sum(data), data
    total_count, spectrum = map_total_counts(data)
    npt.assert_almost_equal(total_count, total_count_expected,
                            err_msg="1D array: total count maps don't match")
    npt.assert_almost_equal(spectrum, spectrum_expected,
                            err_msg="1D array: averaged spectra don't match")

    # Two-dimensional input data
    data = np.random.rand(5, 10)
    total_count_expected, spectrum_expected = np.sum(data, axis=1), np.sum(data, axis=0)
    total_count, spectrum = map_total_counts(data)
    npt.assert_almost_equal(total_count, total_count_expected,
                            err_msg="2D array: total count maps don't match")
    npt.assert_almost_equal(spectrum, spectrum_expected,
                            err_msg="2D array: averaged spectra don't match")

    # Three-dimensional input data
    data = np.random.rand(4, 5, 10)
    total_count_expected, spectrum_expected = np.sum(data, axis=2), np.sum(np.sum(data, axis=0), axis=0)
    total_count, spectrum = map_total_counts(data)
    npt.assert_almost_equal(total_count, total_count_expected,
                            err_msg="3D array: total count maps don't match")
    npt.assert_almost_equal(spectrum, spectrum_expected,
                            err_msg="3D array: averaged spectra don't match")

    # Four-dimensional input data
    data = np.random.rand(3, 4, 5, 10)
    total_count_expected, spectrum_expected = np.sum(data, axis=3), \
        np.sum(np.sum(np.sum(data, axis=0), axis=0), axis=0)
    total_count, spectrum = map_total_counts(data)
    npt.assert_almost_equal(total_count, total_count_expected,
                            err_msg="4D array: total count maps don't match")
    npt.assert_almost_equal(spectrum, spectrum_expected,
                            err_msg="4D array: averaged spectra don't match")

    # Two-dimensional input data represented as a list
    data = np.random.rand(5, 10)
    # Convert to 2D list
    data_list = [list(_) for _ in list(data)]
    total_count_expected, spectrum_expected = np.sum(data, axis=1), np.sum(data, axis=0)
    total_count, spectrum = map_total_counts(data_list)
    npt.assert_almost_equal(total_count, total_count_expected,
                            err_msg="2D list: total count maps don't match")
    npt.assert_almost_equal(spectrum, spectrum_expected,
                            err_msg="2D list: averaged spectra don't match")


@pytest.mark.parametrize("n_rows_max", [1, 3])
@pytest.mark.parametrize("n_columns_max", [1, 3])
@pytest.mark.parametrize("array_size", [(10, 10, 7), (15, 3, 7), (3, 15, 7), (10, 1, 7), (1, 10, 7)])
@pytest.mark.parametrize("random_array", [
    np.random.random,
    da.random.random
])
def test_process_chunk(random_array, array_size, n_rows_max, n_columns_max):
    """Basic tests for `process_chunk` function"""

    data = random_array(size=array_size)
    nr, nc, _ = array_size
    n_row_start = math.floor(nr / 3)
    n_rows = min(nr - n_row_start, n_rows_max)
    n_col_start = math.floor(nc / 3)
    n_cols = min(nc - n_col_start, n_columns_max)

    def processing_func(data, coefficient, *, bias):
        return np. sum(data, axis=data.ndim - 1) * coefficient + bias

    coefficient, bias = 3.0, 5.0
    processing_func = functools.partial(processing_func, coefficient=coefficient, bias=bias)

    result = process_map_chunk(data,
                               (n_row_start, n_col_start, n_rows, n_cols),
                               func=processing_func)
    data_np = np.asarray(data)
    result_expected = np.sum(data_np[n_row_start: n_row_start + n_rows,
                                     n_col_start: n_col_start + n_cols, :],
                             axis=data_np.ndim - 1)
    result_expected = result_expected * coefficient + bias

    npt.assert_array_almost_equal(result, result_expected,
                                  err_msg="Computation result and the expected result don't match")


def test_process_chunk_failing():
    """Tests for failing cases of `process_chunk` function"""

    def processing_func(data):
        return np. sum(data, axis=data.ndim - 1)

    # Wrong dimensions of the data array, must be 3D array
    data = np.random.random(size=(5, 5))
    with pytest.raises(TypeError, match="The input parameter `data` must be 3D array"):
        process_map_chunk(data, selection=(1, 1, 2, 2), func=processing_func)

    # Wrong number of elements: 'selection' parameter
    data = np.random.random(size=(5, 5, 3))
    with pytest.raises(TypeError, match="Argument `selection` must be an iterable returning 4 elements"):
        process_map_chunk(data, selection=(1, 1, 2), func=processing_func)

    # Data chunk is incorrectly defined
    data = np.random.random(size=(5, 7, 3))
    test_cases = [(-1, 1, 2, 2), (5, 1, 2, 2), (6, 1, 2, 2),  # Index 0
                  (1, -1, 2, 2), (1, 7, 2, 2), (1, 8, 2, 2),  # Index 1
                  # Test wrong number of selected points
                  (0, 0, 6, 2), (0, 0, 0, 2),  # Index 2
                  (0, 0, 2, 8), (0, 0, 2, 0)]  # Index 3
    for selection in test_cases:
        with pytest.raises(TypeError,
                           match="Some points in the selected chunk are not contained in the `data` array:"):
            process_map_chunk(data, selection=selection, func=processing_func)
'''


def test_RawHDF5Dataset(tmpdir):
    """Class RawHDF5Dataset"""
    dir_list, fln, dset = ("dir1", "dir2"), "tmp.txt", "/some/dataset"
    file_dir = os.path.join(tmpdir, *dir_list)
    os.makedirs(file_dir, exist_ok=True)
    os.chdir(file_dir)
    # Create file
    with open(fln, "w"):
        pass

    # 'fln' should be converted to absolute
    rds = RawHDF5Dataset(fln, dset)

    assert os.path.isabs(rds.abs_path), "Path was not successfully converted to absolute path"
    # If file exists, then the file name was correctly set
    assert os.path.exists(rds.abs_path), "Path was not correctly set"
    assert rds.dset_name == dset, "Dataset name was not currectly set"


@pytest.mark.parametrize("data_chunksize, chunk_optimal, n_pixels, data_shape", [
    ((3, 5), (12, 10), 100, (20, 20)),
    ((5, 3), (10, 12), 100, (20, 20)),
    ((2, 5), (10, 10), 100, (20, 20)),
    ((11, 13), (11, 13), 100, (20, 20)),
    # The case of the stretched array
    ((2, 2), (6, 2), 10, (50, 2)),  # Stretched vertically
    ((2, 2), (2, 6), 10, (2, 50)),  # Stretched horizontally
    # Chunk size exceeds the size of the array: processed as one big chunk
    ((25, 25), (20, 20), 400, (20, 20)),
    # The number of pixels in the chunk exceeds the size of the array: same
    ((25, 25), (20, 20), 500, (20, 20)),
    # The original chunk size allows to divide the array by 4 chunks
    #   in order to parallelize processing (4 is the default value, but
    #   may be set)
    ((5, 2), (10, 10), 400, (20, 20)),
    ((15, 15), (15, 15), 400, (20, 20)),
])
def test_compute_optimal_chunk_size(data_chunksize, chunk_optimal, n_pixels, data_shape):
    """Basic functionality of the '_compute_optimal_chunk_size'"""

    # Call with kwargs
    res = _compute_optimal_chunk_size(chunk_pixels=n_pixels,
                                      data_chunksize=data_chunksize,
                                      data_shape=data_shape,
                                      n_chunks_min=4)
    assert res == chunk_optimal, "Computed optimal chunks size doesn't match the expected"

    # Call with args
    res = _compute_optimal_chunk_size(n_pixels, data_chunksize, data_shape, 4)
    assert res == chunk_optimal, "Computed optimal chunks size doesn't match the expected"


@pytest.mark.parametrize("data_chunksize, data_shape", [
    ((3,), (20, 20)),
    ((3, 5, 7), (20, 20)),
    (5, (20, 20)),
    ((5, 3), (20,)),
    ((5, 3), (20, 20, 20)),
    ((5, 3), 20),
])
def test_compute_optimal_chunk_size_fail(data_chunksize, data_shape):
    """Failing cases for `compute_optimal_chunk_size`"""
    with pytest.raises(ValueError, match="Unsupported value of parameter"):
        _compute_optimal_chunk_size(10, data_chunksize, data_shape, 4)


@pytest.mark.parametrize("chunk_target, data_shape", [
    ((2, 2), (10, 10, 3)),  # 3D array (primary use case)
    ((2, 3), (9, 11, 3)),
    ((3, 2), (11, 9, 3)),
    ((3, 3), (3, 3, 2)),
    ((1, 1), (1, 1, 3)),
    ((1, 5), (1, 10, 3)),
    ((5, 1), (10, 1, 3)),
    ((3, 4), (2, 3, 2)),
    ((2, 2), (9, 11, 5)),
    ((2, 2), (9, 11)),  # 2D array
    ((2, 2), (9, 11, 5, 3)),  # 4D array
    ((2, 2), (9, 11, 5, 3, 4)),  # 5D array

])
def test_chunk_numpy_array(chunk_target, data_shape):
    """Basic functionailty tests for '_chunk_xrf_map_numpy"""

    data = np.random.random(data_shape)
    data_dask = _chunk_numpy_array(data, chunk_target)

    chunksize_expected = tuple([
        min(chunk_target[0], data_shape[0]),
        min(chunk_target[1], data_shape[1]),
        *data_shape[2:]])

    assert data_dask.shape == data.shape, "The shape of the original and chunked array don't match"
    assert data_dask.chunksize == chunksize_expected, \
        "The chunk size of the Dask array doesn't match the desired chunk size"
    npt.assert_array_equal(data_dask.compute(), data,
                           err_msg="The chunked array is different from the original array")


@pytest.mark.parametrize("chunk_pixels, data_shape, res_chunk_size", [
    (4, (10, 10, 3), (2, 2, 3)),
    (5, (10, 10, 3), (2, 3, 3)),
    (5, (50, 1, 3), (5, 1, 3)),
    (5, (1, 50, 3), (1, 5, 3)),
    (4, (9, 11), (2, 2)),
    (4, (9, 11, 3, 4), (2, 2, 3, 4)),
    (4, (9, 11, 3, 4, 2), (2, 2, 3, 4, 2)),
])
def test_array_numpy_to_dask1(chunk_pixels, data_shape, res_chunk_size):
    """Basic functionality of `_xrf_map_numpy_to_dask`"""

    data = np.random.random(size=data_shape)

    res = _array_numpy_to_dask(data, chunk_pixels)

    assert res.shape == data.shape, "The shape of the original and chunked array don't match"
    assert res.chunksize == res_chunk_size, \
        "The chunk size of the Dask array doesn't match the desired chunk size"


def test_array_numpy_to_dask2():
    """Test if `chunk_pixels` is properly adjusted if it is large compared to the XRF map
    size. Explicitely set the minimum number of chunks."""

    data_shape = (8, 8, 2)
    chunk_pixels = 100  # Some arbitrary large number
    data = np.random.random(size=data_shape)
    res = _array_numpy_to_dask(data, chunk_pixels, n_chunks_min=16)
    res_chunk_size = (2, 2, 2)

    assert res.shape == data.shape, "The shape of the original and chunked array don't match"
    assert res.chunksize == res_chunk_size, \
        "The chunk size of the Dask array doesn't match the desired chunk size"


@pytest.mark.parametrize("data", [
    da.random.random((5, 3, 5)),  # Dask array is not accepted
    10.6,  # Wrong data type
    (5, 6, 7),
    "test",
    np.random.random((3,)),  # Wrong number of dimensions
])
def test_array_numpy_to_dask_fail(data):
    """Wrong type of `data` array"""
    with pytest.raises(ValueError, match="Parameter 'data' must numpy array with at least 2 dimensions"):
        _array_numpy_to_dask(data, (5, 2))


@pytest.mark.parametrize("data_representation", ["numpy_array", "dask_array", "hdf5_file_dset"])
@pytest.mark.parametrize("apply_mask", [False, True])
@pytest.mark.parametrize("select_area", [False, True])
def test_compute_total_spectrum(tmpdir, data_representation, apply_mask, select_area):

    # Start with dask array
    data_shape = (7, 12, 20)
    data_dask = da.random.random(data_shape, chunks=(2, 3, 4))
    data_numpy = data_dask.compute()

    # Mask may be float
    if apply_mask:
        mask = np.random.randint(0, 2, data_shape[0:2])
        mask[3, 4] = 0  # Make sure that at least 1 element is zero
        mask[2, 4] = 1  # Make sure that at least 1 element is non-zero
    else:
        mask = None

    # Selected area (between two points, the second point is not included)
    if select_area:
        selection = (2, 3, 4, 2)  # (y0, x0, ny, nx)
    else:
        selection = None

    # Compute the expected result
    data_tmp = data_numpy
    if mask is not None:
        mask_conv = (mask > 0).astype(dtype=int)
        mask_conv = np.broadcast_to(np.expand_dims(mask_conv, axis=2), data_tmp.shape)
        data_tmp = data_tmp * mask_conv
    if selection is not None:
        y0, x0, ny, nx = selection
        data_tmp = data_tmp[y0: y0 + ny, x0: x0 + nx, :]
    total_spectrum_expected = np.sum(np.sum(data_tmp, axis=0), axis=0)

    if data_representation == "numpy_array":
        data = data_dask.compute()
    elif data_representation == "dask_array":
        data = data_dask
    elif data_representation == "hdf5_file_dset":
        fln = "test_hdf5_file.h5"
        dset_name = "level1/level2"
        with h5py.File(fln, "w") as f:
            # In this test all computations are performed using 'float64' precision,
            #   so we create the dataset with dtype="float64" for consistency.
            dset = f.create_dataset(dset_name, shape=data_dask.shape,
                                    chunks=data_dask.chunksize, dtype="float64")
            dset[:, :, :] = data_dask.compute()
        data = RawHDF5Dataset(fln, dset_name)
    else:
        raise RuntimeError(f"Error in test parameter: unknown value of 'data_representation' = "
                           f"{data_representation}")

    total_spectrum = compute_total_spectrum(data, selection=selection, mask=mask,
                                            chunk_pixels=12)

    npt.assert_array_almost_equal(total_spectrum, total_spectrum_expected,
                                  err_msg="Total spectrum was computed incorrectly")


def test_run_processing_with_dask():
    ...
