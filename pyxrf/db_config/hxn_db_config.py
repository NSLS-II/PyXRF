from databroker import Broker

db = Broker.named('hxn')

from hxntools.handlers.xspress3 import Xspress3HDF5Handler
from hxntools.handlers.timepix import TimepixHDF5Handler

db.fs.register_handler(Xspress3HDF5Handler.HANDLER_NAME,
                       Xspress3HDF5Handler)
db.fs.register_handler(TimepixHDF5Handler._handler_name,
                       TimepixHDF5Handler, overwrite=True)

import suitcase.hdf5 as sc
