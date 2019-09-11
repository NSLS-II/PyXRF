import logging
from logging import NullHandler
from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

logger = logging.getLogger()
logger.addHandler(NullHandler())
