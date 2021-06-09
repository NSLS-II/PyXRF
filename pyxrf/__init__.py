import logging
from logging import NullHandler
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

logger = logging.getLogger("pyxrf")
logger.addHandler(NullHandler())

from .model.load_data_from_db import save_data_to_hdf5  # noqa: F401, E402
from .model.fileio import read_data_from_hdf5  # noqa: F401, E402
