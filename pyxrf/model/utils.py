import numpy as np
import scipy

import logging
logger = logging.getLogger()


# =================================================================================
#  The following set of functions are separated from the rest of the program
#  and prepared to be moved to scikit-beam (skbeam.core.fitting.xrf_model)


def grid_interpolate(data, xx, yy):
    '''
    Interpolate unevenly sampled data to even grid. The new even grid has the same
    dimensions as the original data and covers full range of original X and Y axes.

    Parameters
    ----------

    data : ndarray
        2D array with data values (`xx`, `yy` and `data` must have the same shape)
    xx : ndarray
        2D array with measured values of X coordinates of data points (the values may be unevenly spaced)
    yy : ndarray
        2D array with measured values of Y coordinates of data points (the values may be unevenly spaced)

    Returns
    -------
    data_uniform : ndarray
        2D array with data fitted to even grid (same shape as `data`)
    xx_uniform : ndarray
        2D array with evenly spaced X axis values (same shape as `data`)
    yy_uniform : ndarray
        2D array with evenly spaced Y axis values (same shape as `data`)
    '''
    if (data.shape != xx.shape) or (data.shape != yy.shape):
        logger.debug("Function utils.grid_interpolate: shapes of data and coordinate arrays do not match. "
                     "Grid interpolation is skipped")
        return data, xx, yy
    ny, nx = data.shape
    # Data must be 2-dimensional to use the interpolation procedure
    if (nx <= 1) or (ny <= 1):
        logger.debug("Function utils.grid_interpolate: single row or column scan. "
                     "Grid interpolation is skipped")
        return data, xx, yy
    xx = xx.flatten()
    yy = yy.flatten()
    data = data.flatten()

    # Find the range of axes
    x_min, x_max = np.min(xx), np.max(xx)
    if xx[0] > xx[-1]:
        x_min, x_max = x_max, x_min
    y_min, y_max = np.min(yy), np.max(yy)
    if yy[0] > yy[-1]:
        y_min, y_max = y_max, y_min
    # Create uniform grid
    yy_uniform, xx_uniform = np.mgrid[y_min: y_max: ny * 1j, x_min: x_max: nx * 1j]
    xxyy = np.stack((xx, yy)).T
    # Do the fitting
    data_uniform = scipy.interpolate.griddata(xxyy, data, (xx_uniform, yy_uniform),
                                              method='linear', fill_value=0)
    return data_uniform, xx_uniform, yy_uniform


def normalize_data_by_scaler(data_in, scaler, *, data_name=None, name_not_scalable=None):
    '''
    Normalize data based on the availability of scaler

    Parameters
    ----------

    data_in : ndarray
        numpy array of input data
    scaler : ndarray
        numpy array of scaling data, the same size as data_in
    data_name : str
        name of the data set ('time' or 'i0' etc.)
    name_not_scalable : list
        names of not scalable datasets (['time', 'i0_time'])

    Returns
    -------
    ndarray with normalized data, the same shape as data_in
        The returned array is the reference to 'data_in' if no normalization
        is applied to data or reference to modified copy of 'data_in' if
        normalization was applied.

    ::note::

        Normalization will not be performed if the following is true:

        - scaler is None

        - scaler is not the same shape as data_in

        - scaler contains all elements equal to zero

        If normalization is not performed then REFERENCE to data_in is returned.

    '''

    if data_in is None or scaler is None:  # Nothing to scale
        logger.debug("Function utils.gnormalize_data_by_scaler: data and/or scaler arrays are None. "
                     "Data scaling is skipped.")
        return data_in

    if data_in.shape != scaler.shape:
        logger.debug("Function utils.gnormalize_data_by_scaler: data and scaler arrays have different shape. "
                     "Data scaling is skipped.")
        return data_in

    do_scaling = False
    # Check if data name is in the list of non-scalable items
    # If data name or the list does not exits, then do the scaling
    if name_not_scalable is None or \
            data_name is None or \
            data_name not in name_not_scalable:
        do_scaling = True

    # If scaler is all zeros, then don't scale the data:
    #   check if there is at least one nonzero element
    n_nonzero = np.count_nonzero(scaler)
    if not n_nonzero:
        logger.debug("Function utils.gnormalize_data_by_scaler: scaler is all-zeros array. "
                     "Data scaling is skipped.")
        do_scaling = False

    if do_scaling:
        # If scaler contains some zeros, set those zeros to mean value
        if data_in.size != n_nonzero:
            s_mean = np.mean(scaler[scaler != 0])
            # Avoid division by very small number (or zero)
            if np.abs(s_mean) < 1e-10:
                s_mean = 1e-10 if np.sign(s_mean) >= 0 else -1e-10
            scaler = scaler.copy()
            scaler[scaler == 0.0] = s_mean
        data_out = data_in / scaler
    else:
        data_out = data_in

    return data_out


# ===============================================================================


# ===============================================================================
# The following functions are prepared to be moved to scikit-beam

def _get_2_sqrt_2_log2():
    return 2 * np.sqrt(2 * np.log(2))


def gaussian_sigma_to_fwhm(sigma):
    """
    Converts parameters of Gaussian curve: 'sigma' to 'fwhm'

    Parameters
    ----------

    sigma : float
        sigma of the Gaussian curve

    Returns
    -------
    FWHM of the Gaussian curve
    """
    return sigma * _get_2_sqrt_2_log2()


def gaussian_fwhm_to_sigma(fwhm):
    """
    Converts parameters of Gaussian curve: 'fwhm' to 'sigma'

    Parameters
    ----------

    fwhm : float
        Full Width at Half Maximum of the Gaussian curve

    Returns
    -------
    sigma of the Gaussian curve
    """
    return fwhm / _get_2_sqrt_2_log2()


def _get_sqrt_2_pi():
    return np.sqrt(2 * np.pi)


def gaussian_max_to_area(peak_max, peak_sigma):
    """
    Computes the area under Gaussian curve based on maximum and sigma

    Parameters
    ----------

    peak_max : float
        maximum of the Gaussian curve
    peak_sigma : float
        sigma of the Gaussian curve

    Returns
    -------
    area under the Gaussian curve
    """
    return peak_max * peak_sigma * _get_sqrt_2_pi()


def gaussian_area_to_max(peak_area, peak_sigma):
    """
    Computes the maximum of the Gaussian curve based on area
    under the curve and sigma

    Parameters
    ----------

    peak_area : float
       area under the Gaussian curve
    peak_sigma : float
        sigma of the Gaussian curve

    Returns
    -------
    area under the Gaussian curve
    """
    if peak_sigma == 0:
        return 0
    else:
        return peak_area / peak_sigma / _get_sqrt_2_pi()

# ==================================================================================


def convert_list_to_string(value):
    """
    Convert iterable object to a string (custom operation).
    The function is intended to be used for converting lists (tuples, ndarrays)
    to strings before recording them as attribute values in HDF5 files.
    (Older versions of ``h5py`` 2.6.0 do not support ndarrays as attribute arguments)

    The input container may contain elements of different types. The output string
    is enclosed in square brackets (indicates that the string represent a list).
    The strings are enclosed in single quotes (even if they are already enclosed in
    single quotes). Single quotes are always removed during conversion back to list.
    Integers and floats are replaced by their printed version (``%d`` and ``%.15e``).
    Scientific notation is selected for floats to represent wide range of values with
    good precision.

    Example:

    The list with elements ``abcd``, ``56``, and ``3.05`` is converted to the string
    ``['abcd', 56, 3.0500000000000e00]``.

    Parameters
    ----------

    value : list, tuple, ndarray
        input list of strings, ints or floats

    Returns
    -------

    string representation of the input list
    """

    # The elements of value may change
    value = value.copy()

    for n, v in enumerate(value):
        if isinstance(v, str):
            value[n] = f"'{v}'"
        elif isinstance(v, float):
            value[n] = f"{v:.15e}"
        else:
            # This may be an integer, just use the default printing format
            value[n] = f"{v}"
    value = f"[{', '.join(value)}]"

    return value


def convert_string_to_list(value):
    """
    Convert list represented as string to list representation.
    The operation is reverse to the operation performed by ``convert_list_to_string``.
    The function is intended to be used for converting between lists (tuples, ndarrays)
    and their string representations in working with attribute values in HDF5 files.
    (Older versions of ``h5py`` 2.6.0 do not support ndarrays as attribute arguments)
    See the docstring from ``convert_list_to_string`` for more details.

    The function returns the original value of ``value`` if it is not a list
    (not enclosed in ``[`` and ``]``).

    Parameters
    ----------

    value : str
        string representation of a list (created by function ``convert_list_to_string``

    Returns
    -------

    list containing values from string representation of the list
    """

    # Still check whether it is a string representation of the list
    if isinstance(value, str) and value and value[0] == "[" and value[-1] == "]":
        # The value represents a list, so the list must be retrieved
        value = value.strip("[]").split(", ")
        for n, v in enumerate(value.copy()):
            if v and v[0] == "'" and v[-1] == "'":
                # The list element is a string, so remove single quotes
                value[n] = v.strip("'")
            else:
                try:
                    # Try converting to int
                    value[n] = int(v)
                except:
                    try:
                        # Try converting to float
                        value[n] = float(v)
                    except:
                        pass
                # If everything fails, then leave the element as is

    return value

