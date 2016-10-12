from .model.fileio import (stitch_fitted_results, make_hdf,
                           make_hdf_stitched, export_hdf, export1d)

try:
    from .model.fileio import db
except ImportError as e:
    db = None
    logger.error('databroker is not available: %s', e)
