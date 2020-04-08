import pytest
import numpy as np
import dask.array as da
import numpy.testing as npt
import math
import functools

from pyxrf.core.map_processing import map_total_counts, process_map_chunk


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
@pytest.mark.parametrize("array_size", [(10, 10, 7), (15, 3, 7), (10, 1, 7), (1, 10, 7)])
@pytest.mark.parametrize("random_array", [
    np.random.random,
    da.random.random
])
def test_process_chunk(random_array, array_size, n_rows_max):

    data = random_array(size=array_size)
    nr, _, _ = array_size
    n_row_start = math.floor(nr / 3)
    n_rows = min(nr - n_row_start, n_rows_max)

    def processing_func(data, coefficient, *, bias):
        return np. sum(data, axis=data.ndim - 1) * coefficient + bias

    coefficient, bias = 3.0, 5.0
    processing_func = functools.partial(processing_func, coefficient=coefficient, bias=bias)

    result = process_map_chunk(data,
                               n_row_start=n_row_start,
                               n_rows=n_rows,
                               func=processing_func)
    data_np = np.asarray(data)
    result_expected = np.sum(data_np[n_row_start: n_row_start + n_rows, :, :], axis=data_np.ndim - 1)
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
        process_map_chunk(data, n_row_start=1, n_rows=2, func=processing_func)

    # Data chunk is incorrectly defined
    data = np.random.random(size=(5, 5, 3))
    with pytest.raises(ValueError, match="The data chunk is defined incorrectly.*is outside the range 0"):
        process_map_chunk(data, n_row_start=10, n_rows=2, func=processing_func)
    data = np.random.random(size=(5, 5, 3))
    with pytest.raises(ValueError, match="The data chunk is defined incorrectly.*fall outside"):
        process_map_chunk(data, n_row_start=1, n_rows=10, func=processing_func)
