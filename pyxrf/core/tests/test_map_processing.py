import pytest
import numpy as np
import dask.array as da
import numpy.testing as npt
import h5py
import os
import uuid
from dask.distributed import Client

from pyxrf.core.map_processing import (
    TerminalProgressBar, wait_and_display_progress,
    _compute_optimal_chunk_size, _chunk_numpy_array, _array_numpy_to_dask,
    RawHDF5Dataset, _prepare_xrf_map, _prepare_xrf_mask, compute_total_spectrum,
    _fit_xrf_block, fit_xrf_map)

from pyxrf.core.tests.test_fitting import DataForFittingTest

import logging
logger = logging.getLogger()


def test_TerminalProgressBar():
    """Basic functionality of `TerminalProgressBar`"""

    # We just run the sequence of instructions. I didn't find a way to check the output,
    #   since 'progress' package requires TTY terminal. Otherwise output is blocked
    title = "Monitor progress"
    pbar = TerminalProgressBar(title)
    pbar.start()
    for n in range(10):
        pbar(n * 10)  # Go from 0 to 90%
    pbar.finish()  # This should set it to 100%


class _SampleProgressBar:
    """
    Progress bar class convenient for testing of the functions that
    display progress bar.
    """
    def __init__(self, title):
        self.title = title

        # The list of strings, which we expect to see in the output (capsys out)
        #   Used only for testing purposes only.
        self._expected_output = [
            f"Starting progress bar: {self.title}",
            f"Percent completed: 100.0",
            f"Finished: {self.title}"
        ]

    def start(self):
        print(f"Starting progress bar: {self.title}")

    def __call__(self, percent_completed):
        print(f"Percent completed: {percent_completed}")

    def finish(self):
        print(f"Finished: {self.title}")

    def _check_output(self, capsys_out):
        # For testing purposes only, not needed for actual progress bar implementation
        for s in self._expected_output:
            assert s in capsys_out, \
                f"Expected string {s} is missing in the progress bar object output:\n"\
                f"{capsys_out}"


@pytest.mark.parametrize("progress_bar", [
    _SampleProgressBar("Monitoring progress"),
    TerminalProgressBar("Monitoring progress: "),
    None])
def test_wait_and_display_progress(progress_bar, capsys):
    """Basic test for the function `wait_and_display_progress`"""

    # There is no way to monitor the output (no TTY device -> no output is generated)
    # So we just run a typical sequence of commands and make sure it doesn't crash

    client = Client()
    data = da.random.random(size=(100, 100), chunks=(10, 10))
    sm_fut = da.sum(data, axis=0).persist(scheduler=client)

    # Call the progress monitor
    wait_and_display_progress(sm_fut, progress_bar)

    sm = sm_fut.compute(scheduler=client)
    client.close()

    # Just in case check that the computations were done correctly.
    sm_expected = np.sum(data.compute(), axis=0)
    npt.assert_array_almost_equal(sm, sm_expected, err_msg="Computations are incorrect")

    # Make sure that the output contains all necessary components
    if isinstance(progress_bar, _SampleProgressBar):
        captured = capsys.readouterr()
        progress_bar._check_output(captured.out)


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
    assert not hasattr(rds, "shape"), "Attribute 'shape' exists, while it should not exist"

    # Try to set shape attribute now (use just some arbitrary value)
    rds = RawHDF5Dataset(fln, dset, shape=(10, 50))
    assert rds.shape == (10, 50), "Attribute 'shape' was not set correctly"


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


def _create_xrf_data(data_dask, data_representation, tmpdir):
    """Prepare data represented as numpy array, dask array (no change) or HDF5 dataset"""
    if data_representation == "numpy_array":
        data = data_dask.compute()
    elif data_representation == "dask_array":
        data = data_dask
    elif data_representation == "hdf5_file_dset":
        os.chdir(tmpdir)
        fln = f"test-{uuid.uuid4()}.h5"  # Include UUID in the file name
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

    return data


def _create_xrf_mask(data_shape, apply_mask, select_area):
    """
    Generate a mask for testing of XRF dataset processing functions.

    Parameters
    ----------
    data_shape: tuple or list
        (ny, nx, ...) - the shape of XRF dataset. All dimensions except 0 and 1 are ignored
    apply_mask: bool
        True: generate random mask,
    select_area: bool
        True: select area of the XRF map for processing.

    Returns
    -------
    mask: ndarray(int) or None
        mask is ndarray with shape (ny, nx). Integer values: 0 - pixel inactive,
        1, 2 - pixel active. Note: any positive integer marks pixel as active, but only
        values of 1 and 2 are generated.
    selection: tuple or None
        selected area is (y0, x0, ny_sel, nx_sel)

    If `select_area==True`, then all pixels in `mask` outside the selected area
    are disabled.
    """
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

    return mask, selection


@pytest.mark.parametrize("data_representation", ["numpy_array", "dask_array", "hdf5_file_dset"])
def test_prepare_xrf_data(tmpdir, data_representation):
    # Start with dask array
    data_shape = (7, 12, 20)
    data_dask = da.random.random(data_shape, chunks=(2, 3, 4))
    data_numpy = data_dask.compute()

    data = _create_xrf_data(data_dask, data_representation, tmpdir)
    data, file_obj = _prepare_xrf_map(data, chunk_pixels=12, n_chunks_min=4)

    assert data.chunksize[0] * data.chunksize[1] == 12, f"Dataset was not properly chunked: "\
                                                        f"data.chunksize={data.chunksize}"

    npt.assert_array_almost_equal(data.compute(), data_numpy,
                                  err_msg="Prepared dataset is different from the original")


def test_prepare_xrf_data_fail():
    """Failing test for `_prepare_xrf_mask_fail`"""
    data = 50.0  # Just a number
    with pytest.raises(TypeError, match="Type of parameter 'data' is not supported"):
        _prepare_xrf_map(data, chunk_pixels=12, n_chunks_min=4)


@pytest.mark.parametrize("apply_mask", [False, True])
@pytest.mark.parametrize("select_area", [False, True])
def test_prepare_xrf_mask(apply_mask, select_area):
    """Basic functionality of `_prepare_xrf_mask`"""
    data_shape = (7, 12, 20)
    data_dask = da.random.random(data_shape, chunks=(2, 3, 4))
    mask, selection = _create_xrf_mask(data_shape, apply_mask, select_area)

    # Apply selection to mask
    mask_expected = None
    if mask is not None:
        mask_expected = mask
    if selection:
        if mask_expected is None:
            mask_expected = np.ones(shape=data_shape[0:2])
        ny, nx = mask_expected.shape
        y0, x0 = selection[0], selection[1]
        y1, x1 = y0 + selection[2], x0 + selection[3]
        for y in range(ny):
            if (y < y0) or (y >= y1):
                mask_expected[y, :] = 0
        for x in range(nx):
            if (x < x0) or (x >= x1):
                mask_expected[:, x] = 0

    mask_prepared = _prepare_xrf_mask(data_dask, mask=mask, selection=selection)

    npt.assert_array_equal(mask_prepared, mask_expected,
                           err_msg="The prepared mask is not equal to expected")


@pytest.mark.parametrize("data_dask, mask, selection, msg", [
    (np.random.random((5, 5, 10)), np.random.randint(0, 3, size=(5, 5)), None,
     "Parameter 'data' must be a Dask array"),
    (da.random.random((5,)), np.random.randint(0, 3, size=(5, 5)), None,
     "Parameter 'data' must have at least 2 dimensions"),
    (da.random.random((5, 4, 10)), np.random.randint(0, 3, size=(5, 5)), None,
     "Dimensions 0 and 1 of parameters 'data' and 'mask' do not match"),
    (da.random.random((5, 5, 10)), np.random.randint(0, 3, size=(5, 5)), (1, 3, 4),
     "Parameter 'selection' must be iterable with 4 elements"),
])
def test_prepare_xrf_mask_fail(data_dask, mask, selection, msg):
    """Failing cases of `_prepare_xrf_mask`"""
    with pytest.raises(TypeError, match=msg):
        _prepare_xrf_mask(data_dask, mask=mask, selection=selection)


@pytest.mark.parametrize("data_representation", ["numpy_array", "dask_array", "hdf5_file_dset"])
@pytest.mark.parametrize("apply_mask", [False, True])
@pytest.mark.parametrize("select_area", [False, True])
def test_compute_total_spectrum1(tmpdir, data_representation, apply_mask, select_area):

    # Start with dask array
    data_shape = (7, 12, 20)
    data_dask = da.random.random(data_shape, chunks=(2, 3, 4))
    data_numpy = data_dask.compute()

    mask, selection = _create_xrf_mask(data_shape, apply_mask, select_area)

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

    data = _create_xrf_data(data_dask, data_representation, tmpdir)

    total_spectrum = compute_total_spectrum(data, selection=selection, mask=mask,
                                            chunk_pixels=12,
                                            # Also run all computations with the progress bar
                                            progress_bar=TerminalProgressBar("Monitoring progress: "))

    npt.assert_array_almost_equal(total_spectrum, total_spectrum_expected,
                                  err_msg="Total spectrum was computed incorrectly")


def test_compute_total_spectrum2(tmpdir):
    # Start with dask array
    data_shape = (7, 12, 20)
    data_dask = da.random.random(data_shape, chunks=(2, 3, 4))

    data_numpy = data_dask.compute()
    total_spectrum_expected = np.sum(np.sum(data_numpy, axis=0), axis=0)

    data = _create_xrf_data(data_dask, "dask_array", tmpdir=tmpdir)

    # Create 'external' client and send the reference to 'compute_total_spectrum'
    client = Client(processes=True, silence_logs=logging.ERROR)
    # Run computations without the progress bar
    total_spectrum = compute_total_spectrum(data, chunk_pixels=12, client=client)
    client.close()

    npt.assert_array_almost_equal(total_spectrum, total_spectrum_expected,
                                  err_msg="Total spectrum was computed incorrectly")


@pytest.mark.parametrize("mask", [da.random.random((7, 12)), (7, 12), 20, "abcde"])
def test_compute_total_spectrum_fail(mask):
    """Failing cases: incorrect type for the mask ndarray"""
    data_dask = da.random.random((7, 12, 20), chunks=(2, 3, 4))
    with pytest.raises(TypeError, match="Parameter 'mask' must be a numpy array or None"):
        compute_total_spectrum(data_dask, mask=mask, chunk_pixels=12)


class _FitXRFMapTesting:
    """
    The class implements methods for testing of XRF fitting algorithm.
    Used for testing `_fit_xrf_block` and `fit_xrf_map` functions.
    See the respective tests for examples
    """
    def __init__(self, *, dataset_params, use_snip, add_pts_before, add_pts_after):
        self.use_snip = use_snip
        self.add_pts_before = add_pts_before
        self.add_pts_after = add_pts_after

        self.fitting_data = DataForFittingTest(**dataset_params)

        # 'spectra' has dimensions (n_spec_points, n_lines), which is correct
        self.spectra = self.fitting_data.spectra
        self.n_spectrum_points, self.n_lines = self.spectra.shape

        # 'data_tmp' has dimensions (n_spec_points, ny, nx), so it needs to be rearranged
        self.data_tmp = self.fitting_data.data_input

        # Add some small background if snip is used
        if self.use_snip:
            self.data_tmp += 1.0

        # We want to also add points at the beginning and the end of the spectra
        # The original data is filled with random values. Those values should be either
        #   overwritten by actual spectral data or ignored during fitting.
        self.data_input = np.random.random(
            size=(self.data_tmp.shape[1], self.data_tmp.shape[2],
                  self.data_tmp.shape[0] + self.add_pts_before + self.add_pts_after))

        # Range of indices of the experimental spectrum that should be used for fitting
        #   (it contains the spectrum data). The rest of the points should be ignored
        ne_start = self.add_pts_before
        ne_stop = self.add_pts_before + self.data_tmp.shape[0]
        self.data_sel_indices = (ne_start, ne_stop)

        for ny in range(self.data_tmp.shape[1]):
            for nx in range(self.data_tmp.shape[2]):
                ne_start = self.add_pts_before
                ne_stop = self.add_pts_before + self.data_tmp.shape[0]
                self.data_input[ny, nx, ne_start: ne_stop] = self.data_tmp[:, ny, nx]

        # The snip parameters are set so that the snip width is about 10 points
        self.snip_param = {"e_offset": 0.0, "e_linear": 0.1,
                           "e_quadratic": 0.0, "b_width": 1}

    def verify_output(self, *, data_out):

        assert data_out.shape == (self.data_tmp.shape[1],
                                  self.data_tmp.shape[2],
                                  self.n_lines + 4), \
            f"The shape of 'data_out' is incorrect: data_out.shape={data_out.shape}"

        # Check weights (the axes has to be rearranged to the same order as in 'data_tmp'
        weights_estimated = np.moveaxis(data_out[:, :, 0: self.n_lines], 2, 0)

        # If background subtraction is used, then it's not so easy to test the fitting
        if not self.use_snip:
            self.fitting_data.validate_output_weights(weights_estimated, decimal=10)

        # Let's trust, that R-factor is correctly inserted into
        #   the 'data_out' array. Computation of R-factor is tested elsewhere,

        bg_sum = data_out[:, :, self.n_lines] / self.n_spectrum_points
        if self.use_snip:
            assert ((bg_sum - 1.0) < 0.3).all(), \
                f"Baseline is estimated incorrectly (snip method): \n"\
                f"bg_sum = {bg_sum}\nmin(bg_sum) = {np.min(bg_sum)}\nmax(bg_sum) = {np.max(bg_sum)}"
        else:
            assert (bg_sum == 0.0).all(), \
                f"Baseline estimate is non-zero when snip method is disabled: \n"\
                f"bg_sum = {bg_sum}\nmin(bg_sum) = {np.min(bg_sum)}\nmax(bg_sum) = {np.max(bg_sum)}"

        # Verify if the total count for the whole spectra and the selected region
        #   was computed correctly
        sm_total = np.sum(self.data_input, axis=2)
        sm_sel = np.sum(self.data_tmp, axis=0)  # Sum over the original dataset
        npt.assert_array_almost_equal(
            data_out[:, :, self.n_lines + 2], sm_sel,
            err_msg="Total count for the selected region is computed incorrectly")
        npt.assert_array_almost_equal(
            data_out[:, :, self.n_lines + 3], sm_total,
            err_msg="Total count is computed incorrectly")


@pytest.mark.parametrize("dataset_params", [
    {"n_data_dimensions": (8, 1)},
    {"n_data_dimensions": (1, 8)},
    {"n_data_dimensions": (4, 4)},
    {"n_data_dimensions": (3, 5)},
])
@pytest.mark.parametrize("add_pts_before, add_pts_after", [(0, 0), (50, 100)])
@pytest.mark.parametrize("use_snip", [False, True])
def test_fit_xrf_block(dataset_params, add_pts_before, add_pts_after, use_snip):

    ft = _FitXRFMapTesting(dataset_params=dataset_params,
                           use_snip=use_snip,
                           add_pts_before=add_pts_before,
                           add_pts_after=add_pts_after)

    data_out = _fit_xrf_block(ft.data_input, data_sel_indices=ft.data_sel_indices,
                              matv=ft.spectra, snip_param=ft.snip_param, use_snip=use_snip)

    ft.verify_output(data_out=data_out)


# Used in 'test_fit_xrf_map' tests


@pytest.mark.parametrize("data_representation", ["numpy_array", "dask_array", "hdf5_file_dset"])
@pytest.mark.parametrize("dataset_params", [
    {"n_data_dimensions": (10, 10)},
    {"n_data_dimensions": (9, 11)},
    {"n_data_dimensions": (1, 100)},
    {"n_data_dimensions": (100, 1)},
])
def test_fit_xrf_map1(data_representation, dataset_params, tmpdir):
    """
    Basic functionality of `fit_xrf_map`.
    Tests are run using global Dask clients to inprove testing speed.
    """

    def _run_test(data_representation, dataset_params, add_pts_before,
                  add_pts_after, use_snip, tmpdir, client):

        ft = _FitXRFMapTesting(dataset_params=dataset_params,
                               use_snip=use_snip,
                               add_pts_before=add_pts_before,
                               add_pts_after=add_pts_after)

        # Unfortunately 'data_input' is ndarray, but we need dask array to work with
        #   Select very small chunk size. This is not efficient, but works fine for testing.
        data_dask = _array_numpy_to_dask(ft.data_input, chunk_pixels=4, n_chunks_min=1)
        # Now create the dataset we need
        data = _create_xrf_data(data_dask, data_representation, tmpdir)

        # Run fitting
        data_out = fit_xrf_map(data,
                               data_sel_indices=ft.data_sel_indices,
                               matv=ft.spectra,
                               snip_param=ft.snip_param,
                               use_snip=use_snip,
                               chunk_pixels=10, n_chunks_min=4,
                               progress_bar=None, client=client)

        ft.verify_output(data_out=data_out)

    # Dask client object is used for multiple tests to save execution time
    global_client = Client(processes=True, silence_logs=logging.ERROR)

    for add_pts_before, add_pts_after in [(0, 0), (50, 100)]:
        for use_snip in [False, True]:
            _run_test(data_representation=data_representation,
                      dataset_params=dataset_params,
                      add_pts_before=add_pts_before,
                      add_pts_after=add_pts_after,
                      use_snip=use_snip,
                      tmpdir=tmpdir,
                      client=global_client)

    global_client.close()


def test_fit_xrf_map2():
    """
    Basic functionality of `fit_xrf_map`.
    Tests are run using global Dask clients to inprove testing speed.
    """

    dataset_params = {"n_data_dimensions": (50, 50)}
    add_pts_before, add_pts_after = 15, 10
    use_snip = False

    ft = _FitXRFMapTesting(dataset_params=dataset_params,
                           use_snip=use_snip,
                           add_pts_before=add_pts_before,
                           add_pts_after=add_pts_after)

    # Just run the test with input data represented as numpy array
    data = ft.data_input

    # Run fitting
    data_out = fit_xrf_map(data,
                           data_sel_indices=ft.data_sel_indices,
                           matv=ft.spectra,
                           snip_param=ft.snip_param,
                           use_snip=use_snip,
                           chunk_pixels=10, n_chunks_min=4,
                           progress_bar=None, client=None)

    ft.verify_output(data_out=data_out)
