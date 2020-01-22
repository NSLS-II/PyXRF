from .model.fileio import (stitch_fitted_results,  spec_to_hdf, create_movie,  # noqa: F401
                           combine_data_to_recon, h5file_for_recon, export_to_view,  # noqa: F401
                           make_hdf_stitched)  # noqa: F401
from .model.load_data_from_db import make_hdf, export1d  # noqa: F401
from .model.command_tools import fit_pixel_data_and_save, pyxrf_batch  # noqa: F401
from .xanes_maps.xanes_maps_api import build_xanes_map  # noqa: F401
from .simulation.sim_xrf_scan_data import gen_hdf5_qa_dataset, gen_hdf5_qa_dataset_preset_1  # noqa: F401

# from .model.command_tools import pyxrf_batch  # noqa: F401

# Note:  the statement '# noqa: F401' is telling flake8 to ignore violation F401 at the given line
#     Violation F401 - the package is imported but unused

import logging
logger = logging.getLogger()

logger.setLevel(logging.INFO)

formatter = logging.Formatter(fmt='%(asctime)s : %(levelname)s : %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)

try:
    from .model.load_data_from_db import db
except ImportError:
    db = None
    logger.error("Databroker is not available.")

try:
    from .model.load_data_from_db import db_analysis
except ImportError:
    db_analysis = None
    # We don't use 'analysis' databroker, so disable the message for now
    # logger.error("'Analysis' databroker is not available.")


def pyxrf_api():
    r"""
    =======================================================================================
    Module ``pyxrf.api`` supports the following functions:

        Loading data:
          make_hdf - load XRF mapping data from databroker

        Data processing:
          pyxrf_batch - batch processing of XRF maps
          build_xanes_map - generation and processing of XANES maps

        Simulation of datasets:
          gen_hdf5_qa_dataset - generate quantitative analysis dataset
          gen_hdf5_qa_dataset_preset_1 - generate the dataset based on preset parameters

        VIEW THIS MESSAGE AT ANY TIME: pyxrf_api()

    For more detailed descriptions of the supported functions, type ``help(<function-name>)``
    in IPython command prompt.
    =========================================================================================
    """
    print(pyxrf_api.__doc__)


pyxrf_api()
