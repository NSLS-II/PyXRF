import numpy as np


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
