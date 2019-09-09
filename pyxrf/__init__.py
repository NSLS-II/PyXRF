import logging
logger = logging.getLogger()

from logging import NullHandler
logger.addHandler(NullHandler())

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
