import numpy as np
import scipy
import time as ttime

import logging

logger = logging.getLogger(__name__)


# =================================================================================
#  The following set of functions are separated from the rest of the program
#  and prepared to be moved to scikit-beam (skbeam.core.fitting.xrf_model)


def grid_interpolate(data, xx, yy, xx_uniform=None, yy_uniform=None):
    """
    Interpolate unevenly sampled data to even grid. The new even grid has the same
    dimensions as the original data and covers full range of original X and Y axes.

    Parameters
    ----------

    data : ndarray
        2D array with data values (`xx`, `yy` and `data` must have the same shape)
        ``data`` may be None. In this case interpolation will not be performed, but uniform
        grid will be generated. Use this feature to generate uniform grid.
    xx : ndarray
        2D array with measured values of X coordinates of data points (the values may be unevenly spaced)
    yy : ndarray
        2D array with measured values of Y coordinates of data points (the values may be unevenly spaced)
    xx_uniform : ndarray
        2D array with evenly spaced X axis values (same shape as `data`). If not provided, then
        generated automatically and returned by the function.
    yy_uniform : ndarray
        2D array with evenly spaced Y axis values (same shape as `data`). If not provided, then
        generated automatically and returned by the function.

    Returns
    -------
    data_uniform : ndarray
        2D array with data fitted to even grid (same shape as `data`)
    xx_uniform : ndarray
        2D array with evenly spaced X axis values (same shape as `data`)
    yy_uniform : ndarray
        2D array with evenly spaced Y axis values (same shape as `data`)
    """

    # Check if data shape and shape of coordinate arrays match
    if data is not None:
        if data.shape != xx.shape:
            msg = "Shapes of data and coordinate arrays do not match. (function 'grid_interpolate')"
            raise ValueError(msg)
    if xx.shape != yy.shape:
        msg = "Shapes of coordinate arrays 'xx' and 'yy' do not match. (function 'grid_interpolate')"
        raise ValueError(msg)
    if (xx_uniform is not None) and (xx_uniform.shape != xx.shape):
        msg = (
            "Shapes of data and array of uniform coordinates 'xx_uniform' do not match. "
            "(function 'grid_interpolate')"
        )
        raise ValueError(msg)
    if (yy_uniform is not None) and (xx_uniform.shape != xx.shape):
        msg = (
            "Shapes of data and array of uniform coordinates 'yy_uniform' do not match. "
            "(function 'grid_interpolate')"
        )
        raise ValueError(msg)

    ny, nx = xx.shape
    # Data must be 2-dimensional to use the following interpolation procedure.
    if (nx <= 1) or (ny <= 1):
        logger.debug("Function utils.grid_interpolate: single row or column scan. Grid interpolation is skipped")
        return data, xx, yy

    def _get_range(vv):
        """
        Returns the range of the data coordinates along X or Y axis. Coordinate
        data for a single axis is represented as a 2D array ``vv``. The array
        will have all rows or all columns identical or almost identical.
        The range is returned as ``vv_min`` (leftmost or topmost value)
        and ``vv_max`` (rightmost or bottommost value). Note, that ``vv_min`` may
        be greater than ``vv_max``

        Parameters
        ----------
        vv : ndarray
            2-d array of coordinates

        Returns
        -------
        vv_min : float
            starting point of the range
        vv_max : float
            end of the range
        """
        # The assumption is that X values are mostly changing along the dimension 1 and
        #   Y values change along the dimension 0 of the 2D array and only slightly change
        #   along the alternative dimension. Determine, if the range is for X or Y
        #   axis based on the dimension in which value change is the largest.
        if abs(vv[0, 0] - vv[0, -1]) > abs(vv[0, 0] - vv[-1, 0]):
            vv_min = np.median(vv[:, 0])
            vv_max = np.median(vv[:, -1])
        else:
            vv_min = np.median(vv[0, :])
            vv_max = np.median(vv[-1, :])

        return vv_min, vv_max

    if xx_uniform is None or yy_uniform is None:
        # Find the range of axes
        x_min, x_max = _get_range(xx)
        y_min, y_max = _get_range(yy)
        _yy_uniform, _xx_uniform = np.mgrid[y_min : y_max : ny * 1j, x_min : x_max : nx * 1j]

    if xx_uniform is None:
        xx_uniform = _xx_uniform
    if yy_uniform is None:
        yy_uniform = _yy_uniform

    xx = xx.flatten()
    yy = yy.flatten()
    xxyy = np.stack((xx, yy)).T

    if data is not None:
        # Do the interpolation only if data is provided
        data = data.flatten()
        # Do the interpolation
        data_uniform = scipy.interpolate.griddata(
            xxyy, data, (xx_uniform, yy_uniform), method="linear", fill_value=0
        )
    else:
        data_uniform = None

    return data_uniform, xx_uniform, yy_uniform


def normalize_data_by_scaler(data_in, scaler, *, data_name=None, name_not_scalable=None):
    """
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

    """

    if data_in is None or scaler is None:  # Nothing to scale
        logger.debug(
            "Function utils.normalize_data_by_scaler: data and/or scaler arrays are None. "
            "Data scaling is skipped."
        )
        return data_in

    if data_in.shape != scaler.shape:
        logger.debug(
            "Function utils.normalize_data_by_scaler: data and scaler arrays have different shape. "
            "Data scaling is skipped."
        )
        return data_in

    do_scaling = False
    # Check if data name is in the list of non-scalable items
    # If data name or the list does not exits, then do the scaling
    if name_not_scalable is None or data_name is None or data_name not in name_not_scalable:
        do_scaling = True

    # If scaler is all zeros, then don't scale the data:
    #   check if there is at least one nonzero element
    n_nonzero = np.count_nonzero(scaler)
    if not n_nonzero:
        logger.debug(
            "Function utils.normalize_data_by_scaler: scaler is all-zeros array. Data scaling is skipped."
        )
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


def convert_time_to_nexus_string(t):
    """
    Convert time to a string according to NEXUS format

    Parameters
    ----------

    t : time.struct_time
        Time in the format returned by ``time.localtime`` or ``time.gmtime``

    Returns
    -------

    t : str
        A string represetation of time according to NEXUS standard
    """
    # Convert to sting format recommented for NEXUS files
    t = ttime.strftime("%Y-%m-%dT%H:%M:%S+00:00", t)
    return t


def convert_time_from_nexus_string(t):
    """
    Convert time from NEXUS string to ``time.struct_time``

    Parameters
    ----------

    t : str
        A string represetation of time according to NEXUS standard

    Returns
    -------

    t : time.struct_time
        Time in the format returned by ``time.localtime`` or ``time.gmtime``
    """
    # Convert to sting format recommented for NEXUS files
    t = ttime.strptime(t, "%Y-%m-%dT%H:%M:%S+00:00")
    return t
