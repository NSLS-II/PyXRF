import numpy as np
import math
import os
import h5py
import dask.array as da
from dask.distributed import Client  # , wait
# from progress.bar import ChargingBar, Bar


import logging
logger = logging.getLogger()

'''
def map_total_counts(data):
    """
    Computes the map of total counts and the averaged spectrum for the XRF experiment data.

    Parameters
    ----------
    data: ndarray
        Spectrum is located along the axis with the largest index. For example, the data with
        `data.shape` of `(3, 5, 4096)`, contains 3x5 map with each pixel represented with 4096
        spectral points. The function will return 3x5 map of total count values and 4096 point
        average spectrum. If `data` is a list, then it is converted to ndarray. The function
        will also work correctly if `data` is a scalar, but in general it doesn't make much sense.

    Returns
    -------
    a tuple `(total_count_map, spectrum)`
    """

    data = np.asarray(data)
    n_dim = data.ndim

    if n_dim == 0:
        # `data` is a scalar
        total_count_map, spectrum = data, data
    else:
        # Total count map
        total_count_map = np.sum(data, n_dim - 1)
        if n_dim == 1:
            spectrum = data
        else:
            spectrum = np.sum(data, axis=0)
            for i in range(1, n_dim - 1):
                spectrum = np.sum(spectrum, axis=0)

    return total_count_map, spectrum


def process_map_chunk(data, selection, func):
    """
    Process a chunk of XRF map. The map `data` is 3D array with dimensions (ny, nx, n_spectrum).
    The chunk is determined as the range of rows: `n_row_start .. n_row_start + n_rows`.
    If 'data' is a dask array, then the slice of the array is filled and forwarded
    to the function `func`.

    Parameters
    ----------
    data: ndarray or dask array
        3D XRF map with the shape `(ny, nx, n_spectrum)`
    selection: iterable(int) that contains 4 elements
        Iterable (typically tuple or list) that contains the coordinates of the selected region
        of the `data` array: (ny_start, nx_start, ny_points, nx_points)
    func: callable
        The function that is called for processing of each chunk. The function signature:
        `func(data_chunk, *args, **kwargs)`. The `args` and `kwargs` can be passed to the
        function by applying `functools.partial` to the function.

    Returns
    -------
    Forwards the output of the function `func`.
    """

    # Always convert to tuple for consistency
    selection = tuple(selection)
    if len(selection) != 4:
        raise TypeError(f"Argument `selection` must be an iterable returning 4 elements: "
                        f"type(selection)={type(selection)}")

    ny_start, nx_start, ny_points, nx_points = selection
    ny_stop, nx_stop = ny_start + ny_points, nx_start + nx_points

    if data.ndim != 3:
        raise TypeError(f"The input parameter `data` must be 3D array: number of dimensions = {data.ndim}")

    n_rows, n_columns, _ = data.shape  # The number of rows
    if ((ny_start < 0) or (ny_start >= n_rows) or
        (ny_stop <= 0) or (ny_stop > n_rows) or
        (nx_start < 0) or (nx_start >= n_columns) or
        (nx_stop <= 0) or (nx_stop > n_columns)):
            raise TypeError(f"Some points in the selected chunk are not contained in the `data` array: "
                            f"selection={tuple(selection)}, data dimensions={(n_rows, n_columns)}")

    data_chunk = data[ny_start: ny_stop, nx_start: nx_stop, :]

    if hasattr(data, "compute"):
        # This is a dask array. Fill the array and convert it to ndarray
        data_chunk = data_chunk.compute(scheduler="synchronous")
        data_chunk = np.asarray(data_chunk)

    return func(data_chunk)
'''


class RawHDF5Dataset():
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
    """
    def __init__(self, _abs_path, _dset_name):
        self.abs_path = os.path.abspath(os.path.expanduser(_abs_path))
        self.dset_name = _dset_name


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
        data_slice = data[slice(n1 * chunk_y, min(n1 * chunk_y + chunk_y, ny)),
                          slice(n2 * chunk_x, min(n2 * chunk_x + chunk_x, nx))]
        # Wrap the slice into a list wiht appropriate dimensions
        for _ in range(2, data.ndim):
            data_slice = [data_slice]
        return data_slice

    # Chunk the numpy array and assemble it as a dask array
    data_dask = da.block([
        [
            _get_slice(_1, _2)
            for _2 in range(int(math.ceil(nx / chunk_x)))
        ]
        for _1 in range(int(math.ceil(ny / chunk_y)))
    ])

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
        raise ValueError(f"Parameter 'data' must numpy array with at least 2 dimensions: "
                         f"type(data)={type(data)}")

    ny, nx = data.shape[0:2]
    # Since numpy array is not chunked by default, set the original chunk size to (1,1)
    #   because here we are performing 'original' chunking
    chunk_y, chunk_x = _compute_optimal_chunk_size(chunk_pixels=chunk_pixels,
                                                   data_chunksize=(1, 1),
                                                   data_shape=(ny, nx),
                                                   n_chunks_min=n_chunks_min)

    return _chunk_numpy_array(data, (chunk_y, chunk_x))


def _prepare_xrf_map(data, chunk_pixels=5000, n_chunks_min=4):

    """

    `file_obj` must be kept alive until processing is completed. Closing the file will
    invalidate references in the respective Dask array.
    """

    file_obj = None  # It will remain None, unless 'data' is 'RawHDF5Dataset'

    if isinstance(data, da.core.Array):
        chunk_size = _compute_optimal_chunk_size(chunk_pixels=chunk_pixels,
                                                 data_chunksize=data.chunksize[0:2],
                                                 data_shape=data.shape[0:2],
                                                 n_chunks_min=n_chunks_min)
        data = data.rechunk(chunks=(*chunk_size, data.shape[2]))
    elif isinstance(data, np.ndarray):
        data = _array_numpy_to_dask(data, chunk_pixels=chunk_pixels, n_chunks_min=n_chunks_min)
    elif isinstance(data, RawHDF5Dataset):
        fpath, dset_name = data.abs_path, data.dset_name

        # Note, that the file needs to remain open until the processing is complete !!!
        file_obj = h5py.File(fpath, "r")
        dset = file_obj[dset_name]

        if dset.ndim != 3:
            raise TypeError(f"Dataset '{dset_name}' in file '{fpath}' has {dset.ndim} dimensions: "
                            f"3D dataset is expected")
        ny, nx, ne = dset.shape
        chunk_size = _compute_optimal_chunk_size(chunk_pixels=chunk_pixels,
                                                 data_chunksize=dset.chunks[0:2],
                                                 data_shape=(ny, nx),
                                                 n_chunks_min=n_chunks_min)
        data = da.from_array(dset, chunks=(*chunk_size, ne))
    else:
        raise TypeError(f"Type of parameter 'data' is not supported: type(data)={type(data)}")

    return data, file_obj


def _prepare_xrf_mask(data, mask=None, selection=None):

    if selection is not None:
        y0, x0, ny, nx = selection
        mask_sel = np.zeros(shape=data.shape[0:2])
        mask_sel[y0: y0 + ny, x0: x0 + nx] = 1

        if mask is None:
            mask = mask_sel
        else:
            mask = mask_sel * mask  # We intentionally create the copy of 'mask'

    if mask is not None:
        mask = (mask > 0).astype(dtype=int)
        chunk_y, chunk_x = data.chunksize[0:2]
        mask = _chunk_numpy_array(mask, (chunk_y, chunk_x))

    return mask


def compute_total_spectrum(data, *, selection=None, mask=None,
                           chunk_pixels=5000, n_chunks_min=4, client=None):

    if not isinstance(mask, np.ndarray) and (mask is not None):
        raise TypeError(f"Parameter 'mask' must be a numpy array or None: type(mask) = {type(mask)}")

    data, file_obj = _prepare_xrf_map(data, chunk_pixels=5000, n_chunks_min=4)
    mask = _prepare_xrf_mask(data, mask=mask, selection=selection)

    client = Client(processes=True, silence_logs=logging.ERROR)
    if mask is None:
        result_fut = da.sum(da.sum(data, axis=0), axis=0).persist(scheduler=client)
    else:
        def _masked_sum(data, mask):
            mask = np.broadcast_to(np.expand_dims(mask, axis=2), data.shape)
            sm = np.sum(np.sum(data * mask, axis=0), axis=0)
            return np.array([[sm]])
        result_fut = da.blockwise(_masked_sum, 'ijk', data, "ijk", mask, "ij", dtype="float")

    result = result_fut.compute(scheduler=client)
    client.close()
    if mask is not None:
        # The sum computed for each block still needs to be assembled,
        #   but 'result' is much smaller array than 'data'
        result = np.sum(np.sum(result, axis=0), axis=0)
    return result


def run_processing_with_dask(data, *, func=None, chunks_pixels=5000, client=None):

    # If true, then the client will be created and then closed in the functions
    using_local_client = bool(client is None)
    if using_local_client:
        client = Client(processes=True, silence_logs=logging.ERROR)
        logger.debug("Starting processing with local Dask client")
    else:
        logger.debug("Starting processing with externally provided Dask client")

    # Get the number of workers
    n_workers = len(client.scheduler_info()["workers"])
    logger.debug(f"The number of workers: {n_workers}")

    if hasattr(data, "compute"):
        # 'data' is Dask array
        pass
    else:
        # create chunked dask array
        pass

    if using_local_client:
        client.close()
