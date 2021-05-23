import numpy as np
import math
import os
import h5py
import dask
import dask.array as da
import time as ttime
from numba import jit
from dask.distributed import Client, wait
from progress.bar import Bar
from .fitting import fit_spectrum

import logging

logger = logging.getLogger(__name__)


def dask_client_create(**kwargs):
    """
    Create Dask client object. The function is trivial and introduced so that
    Dask client is created in uniform way throughout the program.
    The client is configured to keep temporary data in `~/.dask` directory
    instead of the current directory.

    Creating new Dask client may be costly (a few extra seconds). If computations require
    multiple calls to functions that are running on Dask, overhead can be reduced by creating
    one common Dask client and supplying the reference to the client as a parameter in each
    function call.

    .. code:: python

        client = dask_client_create()  # Create Dask client
        # <-- code that runs computations -->
        client.close()  # Close Dask client

    Parameters
    ----------
    kwargs: dict, optional
        kwargs will be passed to the Dask client constructor. No extra parameters are needed
        in most cases.

    Returns
    -------
    client: dask.distributed.Client
        Dask client object
    """
    _kwargs = {"processes": True, "silence_logs": logging.ERROR}
    _kwargs.update(kwargs)

    dask.config.set(shuffle="disk")
    path_dask_data = os.path.expanduser("~/.dask")
    dask.config.set({"temporary_directory": path_dask_data})

    client = Client(**_kwargs)
    return client


class TerminalProgressBar:
    """
    Custom class that displays the progress bar in the terminal. Progress
    is displayed in %. Unfortunately it works only in the terminal (or emulated
    terminal) and nothing will be printed in stderr if TTY is disabled.

    Examples
    --------
    .. code-block:: python

        title = "Monitor progress"
        pbar = TerminalProgressBar(title)
        pbar.start()
        for n in range(10):
           pbar(n * 10)  # Go from 0 to 90%
        pbar.finish()  # This should set it to 100%
    """

    def __init__(self, title):
        self.title = title

    def start(self):
        self.bar = Bar(self.title, max=100, suffix="%(percent)d%%")

    def __call__(self, percent_completed):
        while self.bar.index < percent_completed:
            self.bar.next()

    def finish(self):
        while self.bar.index < 100.0:
            self.bar.next()
        self.bar.finish()


def wait_and_display_progress(fut, progress_bar=None):
    """
    Wait for the future to complete and display the progress bar.
    This method may be used to drive any custom progress bar, which
    displays progress in percent from 0 to 100.

    Parameters
    ----------
    fut: dask future
        future object for the batch of tasks submitted to the distributed
        client.
    progress_bar: callable or None
        callable function or callable object with methods `start()`,
        `__call__(float)` and `finish()`. The methods `start()` and
        `finish()` are optional. For example, this could be a reference
        to an instance of the object `TerminalProgressBar`

    Examples
    --------

    .. code-block::

        client = Client()
        data = da.random.random(size=(100, 100), chunks=(10, 10))
        sm_fut = da.sum(data, axis=0).persist(scheduler=client)

        # Call the progress monitor
        wait_and_display_progress(sm_fut, TerminalProgressBar("Monitoring progress: "))

        sm = sm_fut.compute(scheduler=client)
        client.close()
    """

    # If there is no progress bar, then just return without waiting for the future
    if progress_bar is None:
        return

    if hasattr(progress_bar, "start"):
        progress_bar.start()

    progress_bar(1.0)
    while True:
        done, not_done = wait(fut, return_when="FIRST_COMPLETED")
        n_completed, n_pending = len(done), len(not_done)
        n_total = n_completed + n_pending
        percent_completed = n_completed / n_total * 100.0 if n_total > 0 else 100.0

        # It is guaranteed that 'progress_bar' is called for 100% completion
        progress_bar(percent_completed)

        if not n_pending:
            break
        ttime.sleep(0.5)

    if hasattr(progress_bar, "finish"):
        progress_bar.finish()


class RawHDF5Dataset:
    """
    Instead of actual data we may store the HDF5 file name and dataset name within the
    HDF5 file. When access is needed, then data may be loaded from file. Typically
    this will be used for keeping reference to large dataset that will be loaded in
    'lazy' way during processing.

    Parameters
    ----------
    abs_path: str
        absolute path to the HDF5 file
    dset_name: str
        name of the dataset in the HDF5 file
    shape: tuple
        the object is expected to have additional attribute `shape`. Keeping valid shape
        information is very convenient. There is no check, so `shape` may be any value, but
        typically this should be a tuple with actual dataset shape.
    """

    def __init__(self, _abs_path, _dset_name, shape):
        self.abs_path = os.path.abspath(os.path.expanduser(_abs_path))
        self.dset_name = _dset_name
        self.shape = shape


def _compute_optimal_chunk_size(chunk_pixels, data_chunksize, data_shape, n_chunks_min=4):
    """
    Compute the best chunk size for the 'data' array based on existing size and
    chunk size of the `data` array and the desired number of pixels in the chunk.
    The new chunk size will contain whole number of original chunks of the `data`
    array and at least `chunk_pixels` pixels. Image pixels are determined by axes
    0 and 1. The remaining axes (typically axis 2) are not considered during this
    rechunking procedure.

    Parameters
    ----------
    chunk_pixels: int
        The desired number of pixels in the new chunks
    data_chunksize: tuple(int)
        (chunk_y, chunk_x) - original chunk size of the data array
    data_shape: tuple(int)
        (ny, nx) - the shape of the data array
    n_chunks_min: int
        The minimum number of chunks.

    Returns
    -------
    (chunk_y, chunk_x): tuple(int)
        The size of the new chunks along axes 0 and 1
    """

    if not isinstance(data_chunksize, tuple) or len(data_chunksize) != 2:
        raise ValueError(f"Unsupported value of parameter 'data_chunksize': {data_chunksize}")
    if not isinstance(data_shape, tuple) or len(data_shape) != 2:
        raise ValueError(f"Unsupported value of parameter 'data_shape': {data_shape}")

    # Compute chunk size based on the desired number of pixels and chunks in the 'data' array
    dset_chunks, dset_shape = data_chunksize, data_shape
    # Desired number of pixels in the chunk
    n_pixels_total = dset_shape[0] * dset_shape[1]
    if n_pixels_total > n_chunks_min:
        # We want to have at least 1 pixel in the chunk
        chunk_pixels = min(chunk_pixels, n_pixels_total // n_chunks_min)
    # Try to make chunks square (math.sqrt)
    chunk_x = int(math.ceil(math.sqrt(chunk_pixels) / dset_chunks[1]) * dset_chunks[1])
    chunk_x = min(chunk_x, dset_shape[1])  # The array may have few columns
    chunk_y = int(math.ceil(chunk_pixels / chunk_x / dset_chunks[0]) * dset_chunks[0])
    if chunk_y > dset_shape[0]:
        # Now explore the case if the array has small number of rows. We may want to stretch
        #   the chunk horizontally
        chunk_y = dset_shape[0]
        chunk_x = int(math.ceil(chunk_pixels / chunk_y / dset_chunks[1]) * dset_chunks[1])
        chunk_x = min(chunk_x, dset_shape[1])

    return chunk_y, chunk_x


def _chunk_numpy_array(data, chunk_size):
    """
    Convert a numpy array into Dask array with chunks of given size. The function
    splits the array into chunks along axes 0 and 1. If the array has more than 2 dimensions,
    then the remaining dimensions are not chunked. Note, that
    `dask_array = da.array(data, chunks=...)` will set the chunk size, but not split the
    data into chunks, therefore the array can not be loaded block by block by workers
    controlled by a distributed scheduler.

    Parameters
    ----------
    data: ndarray(float), 2 or more dimensions
        XRF map of the shape `(ny, nx, ne)`, where `ny` and `nx` represent the image size
        and `ne` is the number of points in spectra
    chunk_size: tuple(int, int) or list(int, int)
         Chunk size for axis 0 and 1: `(chunk_y, chunk_x`). The function will accept
         chunk size values that are larger then the respective `data` array dimensions.

    Returns
    -------
    data_dask: dask.array
        Dask array with the given chunk size
    """

    chunk_y, chunk_x = chunk_size
    ny, nx = data.shape[0:2]
    chunk_y, chunk_x = min(chunk_y, ny), min(chunk_x, nx)

    def _get_slice(n1, n2):
        data_slice = data[
            slice(n1 * chunk_y, min(n1 * chunk_y + chunk_y, ny)),
            slice(n2 * chunk_x, min(n2 * chunk_x + chunk_x, nx)),
        ]
        # Wrap the slice into a list wiht appropriate dimensions
        for _ in range(2, data.ndim):
            data_slice = [data_slice]
        return data_slice

    # Chunk the numpy array and assemble it as a dask array
    data_dask = da.block(
        [
            [_get_slice(_1, _2) for _2 in range(int(math.ceil(nx / chunk_x)))]
            for _1 in range(int(math.ceil(ny / chunk_y)))
        ]
    )

    return data_dask


def _array_numpy_to_dask(data, chunk_pixels, n_chunks_min=4):
    """
    Convert an array (e.g. XRF map) from numpy array to chunked Dask array. Select chunk
    size based on the desired number of pixels `chunk_pixels`. The array is considered
    as an image with pixels along axes 0 and 1. The array is chunked only along axes 0 and 1.

    Parameters
    ----------
    data: ndarray(float), 3D
        Numpy array of the shape `(ny, nx, ...)` with at least 2 dimensions. If `data` is
        an image, then `ny` and `nx` represent the image dimensions.
    chunk_pixels: int
        Desired number of pixels in a chunk. The actual number of pixels may differ from
        the desired number to accommodate minimum requirements on the number of chunks or
        limited size of the dataset.
    n_chunks_min: int
        minimum number of chunks, which should be selected based on the minimum number of
        workers that should be used to process the map. Each chunk will contain at least
        one pixel: if there is not enough pixels, then the number of chunks will be reduced.

    Results
    -------
    Dask array of the same shape as `data` with chunks selected based on the desired number
    of pixels `chunk_pixels`.
    """

    if not isinstance(data, np.ndarray) or (data.ndim < 2):
        raise ValueError(f"Parameter 'data' must numpy array with at least 2 dimensions: type(data)={type(data)}")

    ny, nx = data.shape[0:2]
    # Since numpy array is not chunked by default, set the original chunk size to (1,1)
    #   because here we are performing 'original' chunking
    chunk_y, chunk_x = _compute_optimal_chunk_size(
        chunk_pixels=chunk_pixels, data_chunksize=(1, 1), data_shape=(ny, nx), n_chunks_min=n_chunks_min
    )

    return _chunk_numpy_array(data, (chunk_y, chunk_x))


def prepare_xrf_map(data, chunk_pixels=5000, n_chunks_min=4):

    """
    Convert XRF map from it's initial representation to properly chunked Dask array.

    Parameters
    ----------
    data: da.core.Array, np.ndarray or RawHDF5Dataset (this is a custom type)
        Raw XRF map represented as Dask array, numpy array or reference to a dataset in
        HDF5 file. The XRF map must have dimensions `(ny, nx, ne)`, where `ny` and `nx`
        define image size and `ne` is the number of spectrum points
    chunk_pixels: int
        The number of pixels in a single chunk. The XRF map will be rechunked so that
        each block contains approximately `chunk_pixels` pixels and contain all `ne`
        spectrum points for each pixel.
    n_chunks_min: int
        Minimum number of chunks. The algorithm will try to split the map into the number
        of chunks equal or greater than `n_chunks_min`. If HDF5 dataset is not chunked,
        then the whole map is treated as one chunk. This should happen only to very small
        files, so parallelism is not important.

    Returns
    -------
    data: da.core.Array
        XRF map represented as Dask array with proper chunk size. The XRF map may be loaded
        block by block when processing using `dask.array.map_blocks` and `dask.array.blockwise`
        functions with Dask multiprocessing scheduler.
    file_obj: h5py.File object
        File object that points to HDF5 file. `None` if input parameter `data` is Dask or
        numpy array. Note, that `file_obj` must be kept alive until processing is completed.
        Closing the file will invalidate references to the dataset in the respective
        Dask array.

    Raises
    ------
    TypeError if input parameter `data` is not one of supported types.
    """

    file_obj = None  # It will remain None, unless 'data' is 'RawHDF5Dataset'

    if isinstance(data, da.core.Array):
        chunk_size = _compute_optimal_chunk_size(
            chunk_pixels=chunk_pixels,
            data_chunksize=data.chunksize[0:2],
            data_shape=data.shape[0:2],
            n_chunks_min=n_chunks_min,
        )
        data = data.rechunk(chunks=(*chunk_size, data.shape[2]))
    elif isinstance(data, np.ndarray):
        data = _array_numpy_to_dask(data, chunk_pixels=chunk_pixels, n_chunks_min=n_chunks_min)
    elif isinstance(data, RawHDF5Dataset):
        fpath, dset_name = data.abs_path, data.dset_name

        # Note, that the file needs to remain open until the processing is complete !!!
        file_obj = h5py.File(fpath, "r")
        dset = file_obj[dset_name]

        if dset.ndim != 3:
            raise TypeError(
                f"Dataset '{dset_name}' in file '{fpath}' has {dset.ndim} dimensions: 3D dataset is expected"
            )
        ny, nx, ne = dset.shape

        if dset.chunks:
            chunk_size = _compute_optimal_chunk_size(
                chunk_pixels=chunk_pixels,
                data_chunksize=dset.chunks[0:2],
                data_shape=(ny, nx),
                n_chunks_min=n_chunks_min,
            )
        else:
            # The data is not chunked. Process data as one chunk.
            chunk_size = (ny, nx)
        data = da.from_array(dset, chunks=(*chunk_size, ne))
    else:
        raise TypeError(f"Type of parameter 'data' is not supported: type(data)={type(data)}")

    return data, file_obj


def _prepare_xrf_mask(data, mask=None, selection=None):
    """
    Create mask for processing XRF maps based on the provided mask, selection
    and XRF dataset. If only `mask` is provided, then it is passed to the output.
    If only `selection` is provided, then the new mask is generated based on selected pixels.
    If both `mask` and `selection` are provided, then all pixels in the mask that fall outside
    the selected area are disabled. Input `mask` is a numpy array. Output `mask` is a Dask
    array with chunk size matching `data`.

    Parameters
    ----------
    data: da.core.Array
        dask array representing XRF dataset with dimensions (ny, nx, ne) and chunksize
        (chunk_y, chunk_x)
    mask: ndarray or None
        mask represented as numpy array with dimensions (ny, nx)
    selection: tuple or list or None
        selected area represented as (y0, x0, ny_sel, nx_sel)

    Returns
    -------
    mask: da.core.Array
        mask represented as Dask array of size (ny, nx) and chunk size (chunk_y, chunk_x)

    Raises
    ------
    TypeError if type of some input parameters is incorrect

    """

    if not isinstance(data, da.core.Array):
        raise TypeError(f"Parameter 'data' must be a Dask array: type(data) = {type(data)}")
    if data.ndim < 2:
        raise TypeError(f"Parameter 'data' must have at least 2 dimensions: data.ndim = {data.ndim}")
    if mask is not None:
        if mask.shape != data.shape[0:2]:
            raise TypeError(
                f"Dimensions 0 and 1 of parameters 'data' and 'mask' do not match: "
                f"data.shape={data.shape} mask.shape={mask.shape}"
            )
    if selection is not None:
        if len(selection) != 4:
            raise TypeError(f"Parameter 'selection' must be iterable with 4 elements: selection = {selection}")

    if selection is not None:
        y0, x0, ny, nx = selection
        mask_sel = np.zeros(shape=data.shape[0:2])
        mask_sel[y0 : y0 + ny, x0 : x0 + nx] = 1

        if mask is None:
            mask = mask_sel
        else:
            mask = mask_sel * mask  # We intentionally create the copy of 'mask'

    if mask is not None:
        mask = (mask > 0).astype(dtype=int)
        chunk_y, chunk_x = data.chunksize[0:2]
        mask = _chunk_numpy_array(mask, (chunk_y, chunk_x))

    return mask


def compute_total_spectrum(
    data, *, selection=None, mask=None, chunk_pixels=5000, n_chunks_min=4, progress_bar=None, client=None
):
    """
    Parameters
    ----------
    data: da.core.Array, np.ndarray or RawHDF5Dataset (this is a custom type)
        Raw XRF map represented as Dask array, numpy array or reference to a dataset in
        HDF5 file. The XRF map must have dimensions `(ny, nx, ne)`, where `ny` and `nx`
        define image size and `ne` is the number of spectrum points
    selection: tuple or list or None
        selected area represented as (y0, x0, ny_sel, nx_sel)
    mask: ndarray or None
        mask represented as numpy array with dimensions (ny, nx)
    chunk_pixels: int
        The number of pixels in a single chunk. The XRF map will be rechunked so that
        each block contains approximately `chunk_pixels` pixels and contain all `ne`
        spectrum points for each pixel.
    n_chunks_min: int
        Minimum number of chunks. The algorithm will try to split the map into the number
        of chunks equal or greater than `n_chunks_min`.
    progress_bar: callable or None
        reference to the callable object that implements progress bar. The example of
        such a class for progress bar object is `TerminalProgressBar`.
    client: dask.distributed.Client or None
        Dask client. If None, then local client will be created

    Returns
    -------
    result: ndarray
        Spectrum averaged over the XRF dataset taking into account mask and selectied area.
    """

    if not isinstance(mask, np.ndarray) and (mask is not None):
        raise TypeError(f"Parameter 'mask' must be a numpy array or None: type(mask) = {type(mask)}")

    data, file_obj = prepare_xrf_map(data, chunk_pixels=chunk_pixels, n_chunks_min=n_chunks_min)
    mask = _prepare_xrf_mask(data, mask=mask, selection=selection)

    if client is None:
        client = dask_client_create()
        client_is_local = True
    else:
        client_is_local = False

    n_workers = len(client.scheduler_info()["workers"])
    logger.info(f"Dask distributed client: {n_workers} workers")

    if mask is None:
        result_fut = da.sum(da.sum(data, axis=0), axis=0).persist(scheduler=client)
    else:

        def _masked_sum(data, mask):
            mask = np.broadcast_to(np.expand_dims(mask, axis=2), data.shape)
            sm = np.sum(np.sum(data * mask, axis=0), axis=0)
            return np.array([[sm]])

        result_fut = da.blockwise(_masked_sum, "ijk", data, "ijk", mask, "ij", dtype="float").persist(
            scheduler=client
        )

    # Call the progress monitor
    wait_and_display_progress(result_fut, progress_bar)

    result = result_fut.compute(scheduler=client)

    if client_is_local:
        client.close()

    if mask is not None:
        # The sum computed for each block still needs to be assembled,
        #   but 'result' is much smaller array than 'data'
        result = np.sum(np.sum(result, axis=0), axis=0)
    return result


def compute_total_spectrum_and_count(
    data, *, selection=None, mask=None, chunk_pixels=5000, n_chunks_min=4, progress_bar=None, client=None
):
    """
    The function is similar to `compute_total_spectrum`, but computes both total
    spectrum and total count map. Total count map is typically used as preview.

    Parameters
    ----------
    data: da.core.Array, np.ndarray or RawHDF5Dataset (this is a custom type)
        Raw XRF map represented as Dask array, numpy array or reference to a dataset in
        HDF5 file. The XRF map must have dimensions `(ny, nx, ne)`, where `ny` and `nx`
        define image size and `ne` is the number of spectrum points
    selection: tuple or list or None
        selected area represented as (y0, x0, ny_sel, nx_sel)
    mask: ndarray or None
        mask represented as numpy array with dimensions (ny, nx)
    chunk_pixels: int
        The number of pixels in a single chunk. The XRF map will be rechunked so that
        each block contains approximately `chunk_pixels` pixels and contain all `ne`
        spectrum points for each pixel.
    n_chunks_min: int
        Minimum number of chunks. The algorithm will try to split the map into the number
        of chunks equal or greater than `n_chunks_min`.
    progress_bar: callable or None
        reference to the callable object that implements progress bar. The example of
        such a class for progress bar object is `TerminalProgressBar`.
    client: dask.distributed.Client or None
        Dask client. If None, then local client will be created

    Returns
    -------
    result: ndarray
        Spectrum averaged over the XRF dataset taking into account mask and selectied area.
    """

    if not isinstance(mask, np.ndarray) and (mask is not None):
        raise TypeError(f"Parameter 'mask' must be a numpy array or None: type(mask) = {type(mask)}")

    data, file_obj = prepare_xrf_map(data, chunk_pixels=chunk_pixels, n_chunks_min=n_chunks_min)
    mask = _prepare_xrf_mask(data, mask=mask, selection=selection)

    if client is None:
        client = dask_client_create()
        client_is_local = True
    else:
        client_is_local = False

    n_workers = len(client.scheduler_info()["workers"])
    logger.info(f"Dask distributed client: {n_workers} workers")

    if mask is None:

        def _process_block(data):
            data = data[0]  # Data is passed as a list of ndarrays
            _spectrum = np.sum(np.sum(data, axis=0), axis=0)
            _count_total = np.sum(data, axis=2)
            return np.array([[{"spectrum": _spectrum, "count_total": _count_total}]])

        result_fut = da.blockwise(_process_block, "ij", data, "ijk", dtype=float).persist(scheduler=client)
    else:

        def _process_block(data, mask):
            data = data[0]  # Data is passed as a list of ndarrays
            mask = np.broadcast_to(np.expand_dims(mask, axis=2), data.shape)
            masked_data = data * mask
            _spectrum = np.sum(np.sum(masked_data, axis=0), axis=0)
            _count_total = np.sum(masked_data, axis=2)
            return np.array([[{"spectrum": _spectrum, "count_total": _count_total}]])

        result_fut = da.blockwise(_process_block, "ij", data, "ijk", mask, "ij", dtype=float).persist(
            scheduler=client
        )

    # Call the progress monitor
    wait_and_display_progress(result_fut, progress_bar)

    result = result_fut.compute(scheduler=client)

    if client_is_local:
        client.close()

    # Assemble results
    total_counts = np.block([[_2["count_total"] for _2 in _1] for _1 in result])
    total_spectrum = sum([_["spectrum"] for _ in result.flatten()])

    return total_spectrum, total_counts


def _fit_xrf_block(data, data_sel_indices, matv, snip_param, use_snip):
    """
    Spectrum fitting for a block of XRF dataset. The function is intended to be
    called using `map_blocks` function for parallel processing using Dask distributed
    package.

    Parameters
    ----------
    data : ndarray
        block of an XRF dataset. Shape=(ny, nx, ne).
    data_sel_indices: tuple
        tuple `(n_start, n_end)` which defines the indices along axis 2 of `data` array
        that are used for fitting. Note that `ne` (in `data`) and `ne_model` (in `matv`)
        are not equal. But `n_end - n_start` MUST be equal to `ne_model`! Indexes
        `n_start .. n_end - 1` will be selected from each pixel.
    matv: ndarray
        Matrix of spectra of the selected elements (emission lines). Shape=(ne_model, n_lines)
    snip_param: dict
        Dictionary of parameters forwarded to 'snip' method for background removal.
        Keys: `e_offset`, `e_linear`, `e_quadratic` (parameters of the energy axis approximation),
        `b_width` (width of the window that defines resolution of the snip algorithm).
    use_snip: bool, optional
        enable/disable background removal using snip algorithm

    Returns
    -------
    data_out: ndarray
        array with fitting results. Shape: `(ny, nx, ne_model + 4)`. For each pixel
        the output data contains: `ne_model` values that represent area under the emission
        line spectra; background area (only in the selected energy range), error (R-factor),
        total count in the selected energy range, total count of the full experimental spectrum.
    """
    spec = data
    spec_sel = spec[:, :, data_sel_indices[0] : data_sel_indices[1]]

    if use_snip:
        bg_sel = np.apply_along_axis(
            snip_method_numba,
            2,
            spec_sel,
            snip_param["e_offset"],
            snip_param["e_linear"],
            snip_param["e_quadratic"],
            width=snip_param["b_width"],
        )

        y = spec_sel - bg_sel
        bg_sum = np.sum(bg_sel, axis=2)

    else:
        y = spec_sel
        bg_sum = np.zeros(shape=data.shape[0:2])

    weights, rfactor, _ = fit_spectrum(y, matv, axis=2, method="nnls")

    total_cnt = np.sum(spec, axis=2)
    sel_cnt = np.sum(spec_sel, axis=2)

    # Stack depth-wise (along axis 2)
    data_out = np.dstack((weights, bg_sum, rfactor, sel_cnt, total_cnt))

    return data_out


def fit_xrf_map(
    data,
    data_sel_indices,
    matv,
    snip_param=None,
    use_snip=True,
    chunk_pixels=5000,
    n_chunks_min=4,
    progress_bar=None,
    client=None,
):
    """
    Fit XRF map.

    Parameters
    ----------
    data: da.core.Array, np.ndarray or RawHDF5Dataset (this is a custom type)
        Raw XRF map represented as Dask array, numpy array or reference to a dataset in
        HDF5 file. The XRF map must have dimensions `(ny, nx, ne)`, where `ny` and `nx`
        define image size and `ne` is the number of spectrum points
    data_sel_indices: tuple
        tuple `(n_start, n_end)` which defines the indices along axis 2 of `data` array
        that are used for fitting. Note that `ne` (in `data`) and `ne_model` (in `matv`)
        are not equal. But `n_end - n_start` MUST be equal to `ne_model`! Indexes
        `n_start .. n_end - 1` will be selected from each pixel.
    matv: array
        Matrix of spectra of the selected elements (emission lines). Shape=(ne_model, n_lines)
    snip_param: dict
        Dictionary of parameters forwarded to 'snip' method for background removal.
        Keys: `e_offset`, `e_linear`, `e_quadratic` (parameters of the energy axis approximation),
        `b_width` (width of the window that defines resolution of the snip algorithm).
        It may be an empty dictionary or None if `use_snip` is `False`.
    use_snip: bool, optional
        enable/disable background removal using snip algorithm
    chunk_pixels: int
        The number of pixels in a single chunk. The XRF map will be rechunked so that
        each block contains approximately `chunk_pixels` pixels and contain all `ne`
        spectrum points for each pixel.
    n_chunks_min: int
        Minimum number of chunks. The algorithm will try to split the map into the number
        of chunks equal or greater than `n_chunks_min`.
    progress_bar: callable or None
        reference to the callable object that implements progress bar. The example of
        such a class for progress bar object is `TerminalProgressBar`.
    client: dask.distributed.Client or None
        Dask client. If None, then local client will be created

    Returns
    -------
    results: ndarray
        array with fitting results. Shape: `(ny, nx, ne_model + 4)`. For each pixel
        the output data contains: `ne_model` values that represent area under the emission
        line spectra; background area (only in the selected energy range), error (R-factor),
        total count in the selected energy range, total count of the full experimental spectrum.
    """

    logger.info("Starting single-pixel fitting ...")
    logger.info(f"Baseline subtraction (SNIP): {'enabled' if use_snip else 'disabled'}.")

    if snip_param is None:
        snip_param = {}  # For consistency

    # Verify that input parameters are valid
    if not isinstance(data_sel_indices, (tuple, list)):
        raise TypeError(
            f"Parameter 'data_sel_indices' must be tuple or list: "
            f"type(data_sel_indices) = {type(data_sel_indices)}"
        )

    if not len(data_sel_indices) == 2:
        raise TypeError(
            f"Parameter 'data_sel_indices' must contain two elements: data_sel_indices = {data_sel_indices}"
        )

    if any([_ < 0 for _ in data_sel_indices]):
        raise ValueError(
            f"Some of the indices in 'data_sel_indices' are negative: data_sel_indices = {data_sel_indices}"
        )

    if data_sel_indices[1] <= data_sel_indices[0]:
        raise ValueError(
            f"Parameter 'data_sel_indices' must select at least 1 element: "
            f"data_sel_indices = {data_sel_indices}"
        )

    if not isinstance(matv, np.ndarray) or matv.ndim != 2:
        raise TypeError(f"Parameter 'matv' must be 2D ndarray: type(matv) = {type(matv)}, matv = {matv}")

    ne_spec, _ = matv.shape
    nsel = data_sel_indices[1] - data_sel_indices[0]
    if ne_spec != nsel:
        raise ValueError(
            f"The number of selected points ({nsel}) is not equal "
            f"to the number of points in reference spectrum ({ne_spec})"
        )

    if not isinstance(snip_param, dict):
        raise TypeError(f"Parameter 'snip_param' must be a dictionary: type(snip_param) = {type(snip_param)}")

    required_keys = ("e_offset", "e_linear", "e_quadratic", "b_width")
    if use_snip and not all([_ in snip_param.keys() for _ in required_keys]):
        raise TypeError(
            f"Parameter 'snip_param' must a dictionary with keys {required_keys}: "
            f"snip_param.keys() = {snip_param.keys()}"
        )

    # Convert data to Dask array
    data, file_obj = prepare_xrf_map(data, chunk_pixels=chunk_pixels, n_chunks_min=n_chunks_min)

    # Verify that selection makes sense (data is Dask array at this point)
    _, _, ne = data.shape
    if data_sel_indices[0] >= ne or data_sel_indices[1] > ne:
        raise ValueError(f"Selection indices {data_sel_indices} are outside the allowed range 0 .. {ne}")

    if client is None:
        client = dask_client_create()
        client_is_local = True
    else:
        client_is_local = False

    n_workers = len(client.scheduler_info()["workers"])
    logger.info(f"Dask distributed client: {n_workers} workers")

    matv_fut = client.scatter(matv)
    result_fut = da.map_blocks(
        _fit_xrf_block,
        data,
        # Parameters of the '_fit_xrf_block' function
        data_sel_indices=data_sel_indices,
        matv=matv_fut,
        snip_param=snip_param,
        use_snip=use_snip,
        # Output data type
        dtype="float",
    ).persist(scheduler=client)

    # Call the progress monitor
    wait_and_display_progress(result_fut, progress_bar)

    result = result_fut.compute(scheduler=client)

    if client_is_local:
        client.close()

    return result


def _compute_roi(data, data_sel_indices, roi_bands, snip_param, use_snip):
    """
    Compute intensity for ROIs (energy bands) in XRF datasets. The function is intended to be
    called using `map_blocks` function for parallel processing using Dask distributed
    package.

    Parameters
    ----------
    data : ndarray
        block of an XRF dataset. Shape=(ny, nx, ne).
    data_sel_indices: tuple
        tuple `(n_start, n_end)` which defines the indices along axis 2 of `data` array
        that are used for fitting. Note that `ne` (in `data`). Indexes
        `n_start .. n_end - 1` will be selected from each pixel.
    roi_bands: list(tuple)
        list of ROI bands, elements are tuples `(left_val, right_val)`, where `left_val` and
        `right_val` are energy values in keV that define the ROI band. If `left_val >= right_val`,
        then the width of the band is considered zero and and ROI will be zero.
    snip_param: dict
        Dictionary of parameters forwarded to 'snip' method for background removal.
        Keys: `e_offset`, `e_linear`, `e_quadratic` (parameters of the energy axis approximation),
        `b_width` (width of the window that defines resolution of the snip algorithm).
        The values of `e_offset` and `e_linear` are used to compute indices for ROIs, so they
        need to be always provided.
    use_snip: bool, optional
        enable/disable background removal using snip algorithm

    Returns
    -------
    data_out: ndarray
        array with ROI counts. Shape: `(ny, nx, len(roi_bands))`. For each pixel
        the output data contains `len(roi_bands)` values that represent area under
        the experimental spectrum inside the band.
    """
    spec = data
    spec_sel = spec[:, :, data_sel_indices[0] : data_sel_indices[1]]

    e_offset = snip_param["e_offset"]
    e_linear = snip_param["e_linear"]
    e_quadratic = snip_param["e_quadratic"]

    if use_snip:
        bg_sel = np.apply_along_axis(
            snip_method_numba, 2, spec_sel, e_offset, e_linear, e_quadratic, width=snip_param["b_width"]
        )
        y = spec_sel - bg_sel

    else:
        y = spec_sel

    # The number of available spectrum points
    ny, nx, n_pts = y.shape
    n_sel_start = data_sel_indices[0]

    def _energy_to_index(energy):
        # 'y' is truncated array, and energy axis is aligned with the full array
        n_index = int(round((energy - e_offset) / e_linear)) - n_sel_start
        n_index = int(np.clip(n_index, a_min=0, a_max=n_pts - 1))
        return n_index

    roi_data = np.zeros(shape=(ny, nx, len(roi_bands)))
    for n, band in enumerate(roi_bands):
        n_left = _energy_to_index(band[0])
        n_right = _energy_to_index(band[1])
        roi_data[:, :, n] = (
            np.sum(y[:, :, n_left:n_right], axis=2) if n_right > n_left else np.zeros(shape=(ny, nx))
        )

    return roi_data


def compute_selected_rois(
    data,
    data_sel_indices,
    roi_dict,
    snip_param=None,
    use_snip=True,
    chunk_pixels=5000,
    n_chunks_min=4,
    progress_bar=None,
    client=None,
):
    """
    Compute XRF map based on ROIs for XRF dataset.

    Parameters
    ----------
    data: da.core.Array, np.ndarray or RawHDF5Dataset (this is a custom type)
        Raw XRF map represented as Dask array, numpy array or reference to a dataset in
        HDF5 file. The XRF map must have dimensions `(ny, nx, ne)`, where `ny` and `nx`
        define image size and `ne` is the number of spectrum points
    data_sel_indices: tuple
        tuple `(n_start, n_end)` which defines the indices along axis 2 of `data` array
        that are used for fitting. Note that `ne` (in `data`). Indexes
        `n_start .. n_end - 1` will be selected from each pixel.
    roi_dict: dict
        Dictionary that specifies ROIs for the selected emission lines:
        key - emission line, value - tuple (left_val, right_val).
        Energy values are in keV.
    snip_param: dict
        Dictionary of parameters forwarded to 'snip' method for background removal.
        Keys: `e_offset`, `e_linear`, `e_quadratic` (parameters of the energy axis approximation),
        `b_width` (width of the window that defines resolution of the snip algorithm).
        The values of `e_offset` and `e_linear` are used to compute indices for ROIs, so they
        need to be always provided.
    use_snip: bool, optional
        enable/disable background removal using snip algorithm
    chunk_pixels: int
        The number of pixels in a single chunk. The XRF map will be rechunked so that
        each block contains approximately `chunk_pixels` pixels and contain all `ne`
        spectrum points for each pixel.
    n_chunks_min: int
        Minimum number of chunks. The algorithm will try to split the map into the number
        of chunks equal or greater than `n_chunks_min`.
    progress_bar: callable or None
        reference to the callable object that implements progress bar. The example of
        such a class for progress bar object is `TerminalProgressBar`.
    client: dask.distributed.Client or None
        Dask client. If None, then local client will be created

    Returns
    -------
    roi_dict_computed: dict
        Dictionary with XRF maps computed for ROIs specified by `roi_dict`.
        Key: emission line. Value: numpy array with shape `(ny, nx)`.
        XRF map values represent area under of the experimental spectrum computed
        over ROI.
    """

    logger.info("Starting ROI computation ...")
    logger.info(f"Baseline subtraction (SNIP): {'enabled' if use_snip else 'disabled'}.")

    # Verify that input parameters are valid
    if not isinstance(data_sel_indices, (tuple, list)):
        raise TypeError(
            f"Parameter 'data_sel_indices' must be tuple or list: "
            f"type(data_sel_indices) = {type(data_sel_indices)}"
        )

    if not len(data_sel_indices) == 2:
        raise TypeError(
            f"Parameter 'data_sel_indices' must contain two elements: data_sel_indices = {data_sel_indices}"
        )

    if any([_ < 0 for _ in data_sel_indices]):
        raise ValueError(
            f"Some of the indices in 'data_sel_indices' are negative: data_sel_indices = {data_sel_indices}"
        )

    if data_sel_indices[1] <= data_sel_indices[0]:
        raise ValueError(
            f"Parameter 'data_sel_indices' must select at least 1 element: "
            f"data_sel_indices = {data_sel_indices}"
        )

    if not isinstance(snip_param, dict):
        raise TypeError(f"Parameter 'snip_param' must be a dictionary: type(snip_param) = {type(snip_param)}")

    required_keys = ("e_offset", "e_linear", "e_quadratic", "b_width")
    if not all([_ in snip_param.keys() for _ in required_keys]):
        raise TypeError(
            f"Parameter 'snip_param' must a dictionary with keys {required_keys}: "
            f"snip_param.keys() = {snip_param.keys()}"
        )

    # Convert data to Dask array
    data, file_obj = prepare_xrf_map(data, chunk_pixels=chunk_pixels, n_chunks_min=n_chunks_min)

    # Verify that selection makes sense (data is Dask array at this point)
    _, _, ne = data.shape
    if data_sel_indices[0] >= ne or data_sel_indices[1] > ne:
        raise ValueError(f"Selection indices {data_sel_indices} are outside the allowed range 0 .. {ne}")

    if client is None:
        client = dask_client_create()
        client_is_local = True
    else:
        client_is_local = False

    n_workers = len(client.scheduler_info()["workers"])
    logger.info(f"Dask distributed client: {n_workers} workers")

    # Prepare ROI bands in the form of a list
    roi_band_keys = []
    roi_bands = []
    for k, v in roi_dict.items():
        roi_band_keys.append(k)
        roi_bands.append(v)

    result_fut = da.map_blocks(
        _compute_roi,
        data,
        # Parameters of the '_fit_xrf_block' function
        data_sel_indices=data_sel_indices,
        roi_bands=roi_bands,
        snip_param=snip_param,
        use_snip=use_snip,
        # Output data type
        dtype="float",
    ).persist(scheduler=client)

    # Call the progress monitor
    wait_and_display_progress(result_fut, progress_bar)

    result = result_fut.compute(scheduler=client)

    roi_dict_computed = {roi_band_keys[_]: result[:, :, _] for _ in range(len(roi_band_keys))}

    if client_is_local:
        client.close()

    return roi_dict_computed


# The following function `snip_method_numba` is a copy of the function
# 'snip_method' from scikit-beam, converted to work with numba.
# It may be considered to move this function to scikit-beam if there
# is interest. The output of this function is tested to match the
# output of 'snip_method' to make sure the functions are equivalent.

_default_con_val_no_bin = 3
_default_con_val_bin = 5
_default_iter_num_no_bin = 3
_default_iter_num_bin = 5


@jit(nopython=True, nogil=True)
def snip_method_numba(
    spectrum,
    e_off,
    e_lin,
    e_quad,
    xmin=0,
    xmax=4096,
    epsilon=2.96,
    width=0.5,
    decrease_factor=np.sqrt(2),
    spectral_binning=None,
    con_val=None,
    iter_num=None,
    width_threshold=0.5,
):
    """
    Use snip algorithm to obtain background.

    The code of this function is borrowed from scikit-beam and changed to
    to work with numba.

    Parameters
    ----------
    spectrum : array
        intensity spectrum
    e_off : float
        energy calibration, such as e_off + e_lin * energy + e_quad * energy^2
    e_lin : float
        energy calibration, such as e_off + e_lin * energy + e_quad * energy^2
    e_quad : float
        energy calibration, such as e_off + e_lin * energy + e_quad * energy^2
    xmin : float, optional
        smallest index to define the range
    xmax : float, optional
        largest index to define the range
    epsilon : float, optional
        energy to create a hole-electron pair
        for Ge 2.96, for Si 3.61 at 300K
        needs to double check this value
    width : int, optional
        window size to adjust how much to shift background
    decrease_factor : float, optional
        gradually decrease of window size, default as sqrt(2)
    spectral_binning : float, optional
        bin the data into different size
    con_val : int, optional
        size of scipy.signal.boxcar to convolve the spectrum.

        Default value is controlled by the keys `con_val_no_bin`
        and `con_val_bin` in the defaults dictionary, depending
        on if spectral_binning is used or not

    iter_num : int, optional
        Number of iterations.

        Default value is controlled by the keys `iter_num_no_bin`
        and `iter_num_bin` in the defaults dictionary, depending
        on if spectral_binning is used or not

    width_threshold : float, optional
        stop point of the algorithm

    Returns
    -------
    background : array
        output results with peak removed

    References
    ----------

    .. [1] C.G. Ryan etc, "SNIP, a statistics-sensitive background
           treatment for the quantitative analysis of PIXE spectra in
           geoscience applications", Nuclear Instruments and Methods in
           Physics Research Section B, vol. 34, 1998.
    """
    # clean input a bit
    if con_val is None:
        if spectral_binning is None:
            con_val = _default_con_val_no_bin
        else:
            con_val = _default_con_val_bin

    if iter_num is None:
        if spectral_binning is None:
            iter_num = _default_iter_num_no_bin
        else:
            iter_num = _default_iter_num_bin

    # np.array(spectrum) is not supported by numba so we have to us this:
    background = np.asarray(spectrum).copy()
    n_background = background.size

    energy = np.arange(n_background, dtype=np.float64)

    if spectral_binning is not None:
        energy = energy * spectral_binning

    energy = e_off + energy * e_lin + energy ** 2 * e_quad

    # transfer from std to fwhm
    std_fwhm = 2 * np.sqrt(2 * np.log(2))
    tmp = (e_off / std_fwhm) ** 2 + energy * epsilon * e_lin
    tmp[tmp < 0] = 0
    fwhm = std_fwhm * np.sqrt(tmp)

    # smooth the background
    s = np.ones(con_val)

    # For background remove, we only care about the central parts
    # where there are peaks. On the boundary part, we don't care
    # the accuracy so much. But we need to pay attention to edge
    # effects in general convolution.
    A = s.sum()

    background = np.convolve(background, s) / A
    # Trim 'background' array to imitate the np.convolve option 'mode="same"'
    mg = len(s) - 1
    n_beg = mg // 2
    n_end = n_beg - mg  # Negative
    background = background[n_beg:n_end]

    window_p = width * fwhm / e_lin
    if spectral_binning is not None and spectral_binning > 0:
        window_p = window_p / 2.0

    background = np.log(np.log(background + 1) + 1)

    index = np.arange(n_background)

    def _clip(arr, vmin, vmax):
        """`np.clip` is not supported by numba, but the following works"""
        arr[arr < vmin] = vmin
        arr[arr > vmax] = vmax
        return arr

    v_xmin, v_xmax = max(xmin, 0), min(xmax, n_background - 1)

    # FIRST SNIPPING
    for j in range(iter_num):
        lo_index = _clip(index - window_p, v_xmin, v_xmax)
        hi_index = _clip(index + window_p, v_xmin, v_xmax)

        temp = (background[lo_index.astype(np.int32)] + background[hi_index.astype(np.int32)]) / 2.0

        bg_index = background > temp
        background[bg_index] = temp[bg_index]

    current_width = window_p
    max_current_width = np.amax(current_width)

    while max_current_width >= width_threshold:
        lo_index = _clip(index - current_width, v_xmin, v_xmax)
        hi_index = _clip(index + current_width, v_xmin, v_xmax)

        temp = (background[lo_index.astype(np.int32)] + background[hi_index.astype(np.int32)]) / 2.0

        bg_index = background > temp
        background[bg_index] = temp[bg_index]

        # decrease the width and repeat
        current_width = current_width / decrease_factor
        max_current_width = np.amax(current_width)

    background = np.exp(np.exp(background) - 1) - 1

    inf_ind = np.where(~np.isfinite(background))
    background[inf_ind] = 0.0

    return background
