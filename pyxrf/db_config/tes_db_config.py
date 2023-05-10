import h5py

try:
    from databroker.v0 import Broker
except ModuleNotFoundError:
    from databroker import Broker

import logging

from databroker._core import register_builtin_handlers

#  srx detector, to be moved to filestore
# from databroker.assets.handlers import Xspress3HDF5Handler
from databroker.assets.handlers import HandlerBase

logger = logging.getLogger(__name__)

db = Broker.named("tes")
try:
    register_builtin_handlers(db.reg)
except Exception as ex:
    logger.error(f"Error while registering default SRX handlers: {ex}")


class BulkXSPRESS(HandlerBase):
    specs = {
        "XPS3_FLY",  # Old incorrect name
        "XSP3_FLY",
    }

    def __init__(self, resource_fn):
        self._handle = h5py.File(resource_fn, "r")

    def __call__(self):
        return self._handle["entry/instrument/detector/data"][:]


for spec in BulkXSPRESS.specs:
    db.reg.register_handler(spec, BulkXSPRESS, overwrite=True)
