import h5py
from metadatastore.mds import MDS
from databroker import Broker
from databroker.core import register_builtin_handlers
from filestore.fs import FileStore

# pull from /etc/metadatastore/connection.yaml
mds = MDS({'host': 'xf05id-ca1',
           'database': 'datastore',
           'port': 27017,
           'timezone': 'US/Eastern'}, auth=False)

# pull configuration from /etc/filestore/connection.yaml
db = Broker(mds, FileStore({'host': 'xf05id-ca1',
                            'database': 'filestore',
                            'port': 27017,
                            'timezone': 'US/Eastern',
                            }))
register_builtin_handlers(db.fs)


# srx detector, to be moved to filestore
from filestore.handlers import Xspress3HDF5Handler, HandlerBase
class BulkXSPRESS(HandlerBase):
    HANDLER_NAME = 'XPS3_FLY'
    def __init__(self, resource_fn):
        self._handle = h5py.File(resource_fn, 'r')

    def __call__(self):
        return self._handle['entry/instrument/detector/data'][:]

db.fs.register_handler(BulkXSPRESS.HANDLER_NAME, BulkXSPRESS,
                       overwrite=True)

class ZebraHDF5Handler(HandlerBase):
    HANDLER_NAME = 'ZEBRA_HDF51'
    def __init__(self, resource_fn):
        self._handle = h5py.File(resource_fn, 'r')

    def __call__(self, column):
        return self._handle[column][:]

class SISHDF5Handler(HandlerBase):
    HANDLER_NAME = 'SIS_HDF51'
    def __init__(self, resource_fn):
        self._handle = h5py.File(resource_fn, 'r')

    def __call__(self, column):
        return self._handle[column][:]

db.fs.register_handler('SIS_HDF51', SISHDF5Handler, overwrite=True)
db.fs.register_handler('ZEBRA_HDF51', ZebraHDF5Handler, overwrite=True)
