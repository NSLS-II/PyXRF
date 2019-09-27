import h5py
from databroker import Broker
from databroker._core import register_builtin_handlers

#  srx detector, to be moved to filestore
# from databroker.assets.handlers import Xspress3HDF5Handler
from databroker.assets.handlers import HandlerBase


db = Broker.named('tes')
register_builtin_handlers(db.reg)


class BulkXSPRESS(HandlerBase):
    HANDLER_NAME = 'XPS3_FLY'

    def __init__(self, resource_fn):
        self._handle = h5py.File(resource_fn, 'r')

    def __call__(self):
        return self._handle['entry/instrument/detector/data'][:]


db.reg.register_handler(BulkXSPRESS.HANDLER_NAME, BulkXSPRESS,
                        overwrite=True)
