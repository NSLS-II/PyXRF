import numpy as np
from scipy.optimize import nnls


def rfactor_compute(spectrum, fit_results, ref_spectra):
    r"""
    Computes R-factor for the fitting results

    Parameters
    ----------
    spectrum : ndarray
        spectrum data on which fitting is performed (N elements)

    fit_results : ndarray
        results of fitting (coefficients, K elements)

    ref_spectra : 2D ndarray
        reference spectra used for fitting (NxK element array)

    Returns
    -------
        float, the value of R-factor
    """

    # Check if input parameters are valid
    assert (
        spectrum.ndim == 1 or spectrum.ndim == 2
    ), "Parameter 'spectrum' must be 1D or 2D array, ({spectrum.ndim})"
    assert spectrum.ndim == fit_results.ndim, (
        f"Spectrum data (ndim = {spectrum.ndim}) and fitting results "
        f"(ndim = {fit_results.ndim}) must have the same number of dimensions"
    )
    assert ref_spectra.ndim == 2, "Parameter 'ref_spectra' must be 2D array, ({ref_spectra.ndim})"
    assert spectrum.shape[0] == ref_spectra.shape[0], (
        f"Arrays 'spectrum' ({spectrum.shape}) and 'ref_spectra' ({ref_spectra.shape}) "
        "must have the same number of data points"
    )
    assert fit_results.shape[0] == ref_spectra.shape[1], (
        f"Arrays 'fit_results' ({fit_results.shape}) and 'ref_spectra' ({ref_spectra.shape}) "
        "must have the same number of spectrum points"
    )
    if spectrum.ndim == 2:  # Only if multiple spectra are processed
        assert spectrum.shape[1] == fit_results.shape[1], (
            f"Arrays 'spectrum' {spectrum.shape} and 'fit_results' {fit_results.shape}"
            "must have the same number of columns"
        )

    spectrum_fit = np.matmul(ref_spectra, fit_results)
    return rfactor(spectrum, spectrum_fit)


def rfactor(spectrum_experimental, spectrum_fit):
    r"""
    Computes R-factor based on two spectra

    Parameters
    ----------
    spectrum_experimental : ndarray
        spectrum data on which fitting is performed (N elements)

    spectrum_fit : ndarray
        fitted spectrum (weighted sum of spectrum components, N elements)

    Returns
    -------
        float, the value of R-factor
    """

    # Compute R-factor
    dif = spectrum_experimental - spectrum_fit
    dif_sum = np.sum(np.abs(dif), axis=0)
    data_sum = np.sum(np.abs(spectrum_experimental), axis=0)

    # Avoid accidental division by zero (or a very small number)
    data_sum = np.clip(data_sum, a_min=1e-30, a_max=None)

    return dif_sum / data_sum


def fit_spectrum(data, ref_spectra, *, method="nnls", axis=0, maxiter=100, rate=0.2, epsilon=1e-30):
    r"""
    Perform fitting of a single or or multiple spectra. A single spectra is represented as
    a 1D ndarray. Multiple spectra collected from scan of a line, 2D or 3D image may be represented
    as multidimensional array containing spectral information along axis ``axis``. The returned
    fitting result have the same dimensionality as the input data with fitted coefficients
    located along the same axis as the spectrum data.

    Parameters
    ----------
    data : ndarray
        single or multidimensional spectral data. If fitting done for a single spectrum with K energy points,
        ``data`` is a 1D ndarray with K points. If 'data' contains spectra obtained by scanning NxM map,
        ``data`` is 3D ndarray with dimensions (K,N,M), (N,K,M) or (N,M,K). The axis containing spectra
        is specified by the parameter ``axis``, which should be set to 0 (default), 1 or 2 for the for the
        3D example above. Dimensionality of ``data`` array is not restricted and the function will perform
        properly if ``axis`` points to the correct dimension.

    ref_spectra : ndarray (2D)
        array with columns representing the reference spectra. If fitting is performed for Q reference
        spectra, then ``ref_spectra`` has dimensions (K,Q), where K is the number of energy points.

    method : str
        optimization method used for fitting. Currently supported methods are "nnls" and "admm".

    axis : int
        the number of the axis in the ``data`` array that hold the spectral information. If ``data``
        array has ``n`` dimensions, then ``axis`` may take values in the range ``-n .. n-1``. The
        fitting results will be placed in the output data array along the same axis.

    maxiter : int
        maximum number of iterations. Optimization may stop prematurely if convergence criteria are met.

    rate : float
        descent rate for optimization algorithm. Currently is used only for ADMM fitting (1/lambda).

    epsilon : float
        small value used in stopping criterion of ADMM optimization algorithm.

    Returns
    -------
    weights : ndarray
        array with the same number of dimensions as ``data``. Weights are placed along the axis ``axis``.
        For example, if ``data`` has the shape (N,K,M) and ``axis=1``, then ``weights`` has
        the shape (N,Q,M), where Q is the number of spectrum references.

    rfactor : ndarray
        array that contains R-factor value for each fitted spectrum. For example, if ``data`` has shape
        (N,K,M) and ``axis=1``, then ``rfactor`` has shape (N,M)

    results_dict : dict
        dictionary that contains additional information returned by a fitting routine. The contents of
        the dictionary depends on the optimization routine. Common information: ``method`` - string
        that contains the name of the optimization method (``nnls`` or ``admm``).

        NNLS optimization:

            - ``residual`` - an array that contains values of least-squares difference between the observed
            and fitted spectra. The same shape as ``rfactor``.

        ADMM optimization:

            - ``convergence`` - 1D array of values that represent change of weights at each iteration of
            the optimization. The number of elements is equal to the number of iterations.

            - ``feasibility`` - 1D array, same shape as ``convergence``
    """
    # Explicitly check if one of the supported optimisation method is specified
    method = method.lower()
    supported_fitting_methods = ("nnls", "admm")
    assert (
        method in supported_fitting_methods
    ), f"Fitting method '{method}' is not supported. Supported methods: {supported_fitting_methods}"

    data = np.asarray(data)
    ref_spectra = np.asarray(ref_spectra)

    assert ref_spectra.ndim == 2, f"The array 'ref_spectra' must have 2 dimensions instead of {ref_spectra.ndim}"

    assert (axis >= -data.ndim) and (
        axis < data.ndim
    ), f"Specified axis {axis} does not exist in data array. Allowed values: {-data.ndim} .. {data.ndim - 1}"

    # Switch to standard data view (spectrum points along axis==0)
    data = np.moveaxis(data, axis, 0)

    n_pts = data.shape[0]
    data_dims = data.shape[1:]
    n_pts_2 = ref_spectra.shape[0]
    n_refs = ref_spectra.shape[1]

    assert (
        n_pts == n_pts_2
    ), f"The number of spectrum points in data ({n_pts}) and references ({n_pts_2}) do not match."

    assert rate > 0.0, f"The parameter 'rate' is zero or negative ({rate:.6g})"

    assert maxiter > 0, f"The parameter 'maxiter' is zero or negative ({rate})"

    assert epsilon > 0.0, f"The parameter 'epsilon' is zero or negative ({rate:.6g})"

    # Depending on 'data_dim', there could be three cases
    #   'data_dim' is empty - data is 1D array representing a single point, 1D array of weights
    #                         will be returned, data must be converted to 2D array for processing
    #   'data_dim' has one element - data is 2D array, representing one line of pixels, process as is
    #   'data_dim' has more than one element - data is multidimensional array representing
    #                        multidimensional image, reshape to 1D data (2D array) for processing
    #                        and the convert back to multidimensional representation

    if not data_dims:
        data_1D = np.expand_dims(data, axis=1)
    elif len(data_dims) > 1:
        data_1D = np.reshape(data, [n_pts, np.prod(data_dims)])
    else:
        data_1D = data

    if method == "admm":
        weights, rfactor, convergence, feasibility = _fitting_admm(
            data_1D, ref_spectra, rate=rate, maxiter=maxiter, epsilon=epsilon
        )
    else:
        # Call the default "nnls" method, since this is the only choice left.
        weights, rfactor, residual = _fitting_nnls(data_1D, ref_spectra, maxiter=maxiter)

    # Reshape the fitting results in case of a single pixel or multidimensional map
    if not data_dims:
        weights = np.squeeze(weights, axis=1)
        rfactor = rfactor[0]
        if method == "nnls":
            residual = residual[0]
    elif len(data_dims) > 1:
        weights = np.reshape(weights, np.insert(data_dims, 0, n_refs))
        rfactor = np.reshape(rfactor, data_dims)
        if method == "nnls":
            residual = np.reshape(residual, data_dims)

    # Now swap back the results of fitting (coefficients are along the same axis as the spectrum points)
    weights = np.moveaxis(weights, 0, axis)

    # The results returned for each optimization method include 'weights' and 'rfactor'
    #   Additional results are different for each optimisation method and are returned as
    #   elements of the dictionary ``results_dict``.
    if method == "admm":
        results_dict = {
            "method": method,
            "convergence": convergence,
            "feasibility": feasibility,
        }
    else:
        results_dict = {
            "method": method,
            "residual": residual,
        }

    return weights, rfactor, results_dict


def _fitting_nnls(data, ref_spectra, *, maxiter=100):
    r"""
    Fitting of multiple spectra using NNLS method.

    Parameters
    ----------

    data : ndarray(float), 2D
        array holding multiple observed spectra, shape (K, N), where K is the number of energy points,
        and N is the number of spectra

    absorption_refs : ndarray(float), 2D
        array of references, shape (K, Q), where Q is the number of references.

    maxiter : int
        maximum number of iterations. Optimization may stop prematurely if convergence criteria are met.

    Returns
    -------

    map_data_fitted : ndarray(float), 2D
        fitting results, shape (Q, N), where Q is the number of references and N is the number of spectra.

    map_rfactor : ndarray(float), 2D
        map that represents R-factor for the fitting, shape (M,N).

    map_residual : ndarray(float), 2D
        residual returned by NNLS algorithm
    """
    assert data.ndim == 2, "Data array 'data' must have 2 dimensions"
    assert ref_spectra.ndim == 2, "Data array 'ref_spectra' must have 2 dimensions"

    n_pts = data.shape[0]
    n_pixels = data.shape[1]
    n_pts_2 = ref_spectra.shape[0]
    n_refs = ref_spectra.shape[1]

    assert (
        n_pts == n_pts_2
    ), f"The number of spectrum points in data ({n_pts}) and references ({n_pts_2}) do not match."

    assert maxiter > 0, f"The parameter 'maxiter' is zero or negative ({maxiter})"

    map_data_fitted = np.zeros(shape=[n_refs, n_pixels])
    map_rfactor = np.zeros(shape=[n_pixels])
    map_residual = np.zeros(shape=[n_pixels])
    for n in range(n_pixels):
        map_sel = data[:, n]
        result, residual = nnls(ref_spectra, map_sel, maxiter=maxiter)

        rfactor = rfactor_compute(map_sel, result, ref_spectra)

        map_data_fitted[:, n] = result
        map_rfactor[n] = rfactor
        map_residual[n] = residual

    return map_data_fitted, map_rfactor, map_residual


def _fitting_admm(data, ref_spectra, *, rate=0.2, maxiter=100, epsilon=1e-30, non_negative=True):
    r"""
    Fitting of multiple spectra using ADMM method.

    Parameters
    ----------

    data : ndarray(float), 2D
        array holding multiple observed spectra, shape (K, N), where K is the number of energy points,
        and N is the number of spectra

    absorption_refs : ndarray(float), 2D
        array of references, shape (K, Q), where Q is the number of references.

    maxiter : int
        maximum number of iterations. Optimization may stop prematurely if convergence criteria are met.

    rate : float
        descent rate for optimization algorithm. Currently is used only for ADMM fitting (1/lambda).

    epsilon : float
        small value used in stopping criterion of ADMM optimization algorithm.

    non_negative : bool
        if True, then the solution is guaranteed to be non-negative

    Returns
    -------

    map_data_fitted : ndarray(float), 2D
        fitting results, shape (Q, N), where Q is the number of references and N is the number of spectra.

    map_rfactor : ndarray(float), 2D
        map that represents R-factor for the fitting, shape (M,N).

    convergence : ndarray(float), 1D
        convergence data returned by ADMM algorithm

    feasibility : ndarray(float), 1D
        feasibility data returned by ADMM algorithm

    The prototype for the ADMM fitting function was implemented by Hanfei Yan in Matlab.
    """
    assert data.ndim == 2, "Data array 'data' must have 2 dimensions"
    assert ref_spectra.ndim == 2, "Data array 'ref_spectra' must have 2 dimensions"

    n_pts = data.shape[0]
    n_pixels = data.shape[1]
    n_pts_2 = ref_spectra.shape[0]
    n_refs = ref_spectra.shape[1]

    assert (
        n_pts == n_pts_2
    ), f"ADMM fitting: number of spectrum points in data ({n_pts}) and references ({n_pts_2}) do not match."

    assert rate > 0.0, f"ADMM fitting: parameter 'rate' is zero or negative ({rate:.6g})"

    assert maxiter > 0, f"ADMM fitting: parameter 'maxiter' is zero or negative ({rate})"

    assert epsilon > 0.0, f"ADMM fitting: parameter 'epsilon' is zero or negative ({rate:.6g})"

    y = data
    # Calculate some quantity to be used in the iteration
    A = ref_spectra
    At = np.transpose(A)

    z = np.matmul(At, y)
    c = np.matmul(At, A)

    # Initialize variables
    w = np.ones(shape=[n_refs, n_pixels])
    u = np.zeros(shape=[n_refs, n_pixels])

    # Feasibility test: x == w
    convergence = np.zeros(shape=[maxiter])
    feasibility = np.zeros(shape=[maxiter])

    dg = np.eye(n_refs, dtype=float) * rate
    m1 = np.linalg.inv((c + dg))

    n_iter = 0
    for i in range(maxiter):
        m2 = z + (w - u) * rate
        x = np.matmul(m1, m2)
        w_updated = x + u
        if non_negative:
            w_updated = w_updated.clip(min=0)
        u = u + x - w_updated

        conv = np.linalg.norm(w_updated - w) / np.linalg.norm(w_updated)
        convergence[i] = conv
        feasibility[i] = np.linalg.norm(x - w_updated)

        w = w_updated

        if conv < epsilon:
            n_iter = i + 1
            break

    # Compute R-factor
    rfactor = rfactor_compute(data, w, ref_spectra)

    convergence = convergence[:n_iter]
    feasibility = feasibility[:n_iter]

    return w, rfactor, convergence, feasibility
