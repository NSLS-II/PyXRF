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
