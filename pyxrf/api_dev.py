# Use this file if you need to import PyXRF APIs into a custom script.
# Use 'pyxrf.api' if you are interactively importing APIs into an IPython session.

from .model.fileio import stitch_fitted_results  # noqa: F401
from .model.fileio import spec_to_hdf  # noqa: F401
from .model.fileio import create_movie  # noqa: F401
from .model.fileio import combine_data_to_recon  # noqa: F401
from .model.fileio import h5file_for_recon  # noqa: F401
from .model.fileio import export_to_view  # noqa: F401
from .model.fileio import make_hdf_stitched  # noqa: F401

from .model.load_data_from_db import make_hdf, export1d  # noqa: F401
from .model.command_tools import fit_pixel_data_and_save, pyxrf_batch  # noqa: F401
from .xanes_maps.xanes_maps_api import build_xanes_map  # noqa: F401
from .simulation.sim_xrf_scan_data import gen_hdf5_qa_dataset, gen_hdf5_qa_dataset_preset_1  # noqa: F401
from .core.map_processing import dask_client_create  # noqa: F401

from .model.load_data_from_db import save_data_to_hdf5  # noqa: F401, E402
from .model.fileio import read_data_from_hdf5  # noqa: F401, E402

# Note:  the statement '# noqa: F401' is telling flake8 to ignore violation F401 at the given line
#     Violation F401 - the package is imported but unused

import logging

logger = logging.getLogger("pyxrf")

logger.setLevel(logging.INFO)

formatter = logging.Formatter(fmt="%(asctime)s : %(levelname)s : %(message)s")

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
