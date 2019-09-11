from .model.fileio import (stitch_fitted_results,  spec_to_hdf, create_movie,  # noqa: F401
                           combine_data_to_recon, h5file_for_recon, export_to_view,  # noqa: F401
                           make_hdf_stitched)  # noqa: F401
from .model.load_data_from_db import make_hdf, export1d  # noqa: F401
from .model.command_tools import fit_pixel_data_and_save, pyxrf_batch  # noqa: F401

# Note:  the statement '# noqa: F401' is telling flake8 to ignore violation F401 at the given line
#     Violation F401 - the package is imported but unused

import logging
logger = logging.getLogger()

try:
    from .model.load_data_from_db import db, db_analysis
except ImportError:
    db = None
    db_analysis = None
    logger.error('databroker is not available.')
