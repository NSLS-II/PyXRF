import pytest
import numpy as np

from ..model.utils import fitting_admm

# Tolerance used for comparing floating point numbers
tolerance_compare_floats = 1e-30

# ------------------------------------------------------------------------------
#  useful functions for generating of datasets for testing of fitting algorithms

def _generate_gaussian_spectra(x_values, gaussian_centers, gaussian_std):

    assert len(gaussian_centers) == len(gaussian_std), \
        "The number of center values must be equal to the number of STD values"
    nn = len(x_values)
    n_spectra = len(gaussian_centers)

    spectra = np.zeros(shape=[nn, len(gaussian_centers)])
    for n in range(n_spectra):
        p, std = gaussian_centers[n], gaussian_std[n]
        spectra[:, n] = np.exp(-np.square(x_values - p) / (2 * std ** 2))

    return spectra

class DataForAdmmFittingTest:
    """
    The class that generates and stores dataset used for testing of fitting algorithms
    """
    def __init__(self, **kwargs):
        self.spectra = None
        self.weights = None
        self.data_input = None

        self.generate_dataset(**kwargs)

    def generate_dataset(self, *, n_pts = 101, pts_range=(0, 100),
                         n_spectra = 3, n_gaus_centers_range=(20,80), gauss_std_range=(10, 20),
                         weights_range=(0.1, 1), n_data_dimensions = (8,)):

        if n_data_dimensions:
            data_dim = n_data_dimensions
        else:
            data_dim = (1,)

        # Values for 'energy' axis
        self.x_values = np.mgrid[pts_range[0]: pts_range[1]: n_pts * 1j]

        # Centers of gaussians are evenly spread in the range
        gaussian_centers = np.mgrid[n_gaus_centers_range[0]: n_gaus_centers_range[1]: n_spectra * 1j]
        # Standard deviations are uniformly distributed in the range
        gaussian_std = np.random.rand(n_spectra) * \
                       (gauss_std_range[1] - gauss_std_range[0]) + gauss_std_range[0]

        self.spectra = _generate_gaussian_spectra(x_values=self.x_values,
                                                 gaussian_centers=gaussian_centers,
                                                 gaussian_std=gaussian_std)

        # The number of pixels in the flattened multidimensional image
        dims = np.prod(data_dim)
        # Generate data for every pixel of the multidimensional image
        self.weights = np.random.rand(n_spectra, dims) * \
                       (weights_range[1] - weights_range[0]) + weights_range[0]
        self.data_input = np.matmul(self.spectra, self.weights)

        if n_data_dimensions:
            # Convert weights and data from 2D to multidimensional arrays
            self.weights = np.reshape(self.weights, np.insert(data_dim, 0, n_spectra))
            self.data_input = np.reshape(self.data_input, np.insert(data_dim, 0, n_pts))
        else:
            # Convert weights and data to 1D arrays representing a single point
            self.weights = np.squeeze(self.weights, axis=1)
            self.data_input = np.squeeze(self.data_input, axis=1)

    def validate_output_weights(self, weights_output, atol=tolerance_compare_floats):

        assert weights_output.shape == self.weights.shape, \
            f"Shapes of the output weight array {weights_output.shape} and "\
            f" input weight array {self.weights.shape} do not match. Can not compare the arrays"

        # Check if the weights match
        return np.all(np.isclose(weights_output, self.weights, atol=atol))


data_for_admm_fitting_test = DataForAdmmFittingTest()

# Trivial dataset that may be reused for testing fitting algorithms
test_data_admm = [DataForAdmmFittingTest(n_data_dimensions=(8,)),
                  DataForAdmmFittingTest(n_data_dimensions=(9,)),
                  DataForAdmmFittingTest(n_data_dimensions=(1,)),
                  DataForAdmmFittingTest(n_data_dimensions=(3, 8)),
                  DataForAdmmFittingTest(n_data_dimensions=(4, 7)),
                  DataForAdmmFittingTest(n_data_dimensions=(1, 8)),
                  DataForAdmmFittingTest(n_data_dimensions=(8, 1)),
                  DataForAdmmFittingTest(n_data_dimensions=())
                  ]

@pytest.mark.parametrize("fitting_data", test_data_admm)
def test_admm_normal_use(fitting_data):

    spectra = fitting_data.spectra
    data_input = fitting_data.data_input

    # -------------- Test regular fitting ---------------
    weights_estimated, convergence, feasibility = fitting_admm(data_input, spectra)

    # Check if the weights match
    assert fitting_data.validate_output_weights(weights_estimated), \
        "The difference between the initial and estimated weights are too large"

    # Check the convergence data
    assert (convergence.ndim == 1) and (convergence.size >= 1) \
           and convergence[-1] < tolerance_compare_floats, \
           "Convergence array has incorrect dimensions or the alogrithm did not converge"

    # Check feasibility array dimensions
    assert (feasibility.ndim == 1) and (feasibility.size >= 1), \
        "Feasibility array has incorrect dimensions"


@pytest.mark.parametrize("fitting_data", test_data_admm)
def test_admm_try_wrong_input_dimensions(fitting_data):

    spectra = fitting_data.spectra
    data_input = fitting_data.data_input

    # -------------- Try feeding input data with wrong dimensions ----------------
    # Remove one data point (along axis 0)
    data_input_wrong_dimensions = np.delete(data_input, -1, axis=0)
    with pytest.raises(AssertionError,
                       match=r"number of spectrum points in data \(\d+\) " \
                             r"and references \(\d+\) do not match"):
        fitting_admm(data_input_wrong_dimensions, spectra)


@pytest.mark.parametrize("fitting_data", test_data_admm)
def test_admm_try_wrong_fitting_param_value(fitting_data):

    spectra = fitting_data.spectra
    data_input = fitting_data.data_input

    # -------------- Set 'rate' to 0.0 ----------------
    with pytest.raises(AssertionError,
                       match=r"parameter 'rate' is zero or negative"):
        fitting_admm(data_input, spectra, rate=0.0)

    # -------------- Set 'maxiter' to 0 ----------------
    with pytest.raises(AssertionError,
                       match=r"parameter 'maxiter' is zero or negative"):
        fitting_admm(data_input, spectra, maxiter=0)

    # -------------- Set 'epsilon' to 0.0 ----------------
    with pytest.raises(AssertionError,
                       match=r"parameter 'epsilon' is zero or negative"):
        fitting_admm(data_input, spectra, epsilon=0.0)
