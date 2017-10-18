from metadatastore.mds import MDS
from databroker import Broker
from databroker.core import register_builtin_handlers
from filestore.fs import FileStore

_mds_config = {'host': 'xf03id-ca1',
               'port': 27017,
               'database': 'datastore-new',
               'timezone': 'US/Eastern'}
mds = MDS(_mds_config, auth=False)

_fs_config = {'host': 'xf03id-ca1',
              'port': 27017,
              'database': 'filestore-new'}
db_new = Broker(mds, FileStore(_fs_config))

_mds_config_old = {'host': 'xf03id-ca1',
               'port': 27017,
               'database': 'datastore',
               'timezone': 'US/Eastern'}
mds_old = MDS(_mds_config_old, auth=False)

_fs_config_old = {'host': 'xf03id-ca1',
              'port': 27017,
              'database': 'filestore'}
db_old = Broker(mds_old, FileStore(_fs_config_old))


from hxntools.handlers.xspress3 import Xspress3HDF5Handler
from hxntools.handlers.timepix import TimepixHDF5Handler

register_builtin_handlers(db_new.fs)

db_new.fs.register_handler(Xspress3HDF5Handler.HANDLER_NAME,
                           Xspress3HDF5Handler)
db_new.fs.register_handler(TimepixHDF5Handler._handler_name,
                           TimepixHDF5Handler, overwrite=True)


register_builtin_handlers(db_old.fs)
db_old.fs.register_handler(Xspress3HDF5Handler.HANDLER_NAME,
                           Xspress3HDF5Handler)
db_old.fs.register_handler(TimepixHDF5Handler._handler_name,
                           TimepixHDF5Handler, overwrite=True)


# wrapper for two databases
class Broker_New(Broker):

    def __getitem__(self, key):
        try:
            return db_new[key]
        except ValueError:
            return db_old[key]

    def get_table(self, *args, **kwargs):
        result = db_new.get_table(*args, **kwargs)
        if len(result) == 0:
            result = db_old.get_table(*args, **kwargs)
        return result

    def get_images(self, *args, **kwargs):
        result = db_new.get_images(*args, **kwargs)
        if len(result) == 0:
            result = db_old.get_images(*args, **kwargs)
        return result


db = Broker_New(mds, FileStore(_fs_config))

import suitcase.hdf5 as sc
