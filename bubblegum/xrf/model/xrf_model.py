__author__ = 'edill'

from pprint import pprint
from atom.api import Atom, Str, observe, Typed
import numpy as np

class XRF(Atom):
    folder_name = Str()
    file_name = Str()
    data = Typed(object)

    @observe('folder_name', 'file_name')
    def update(self, changed):
        pprint(changed)
        if changed['type'] == 'create':
            return
        print('{} was changed from {} to {}'.format(changed['name'],
                                                    changed['oldvalue'],
                                                    changed['value']))
        if changed['name'] == 'file_name':
            self.load_data()

    def load_data(self):
        self.data = np.loadtxt(self.file_name)

    @observe('data')
    def data_changed(self, data):
        print('The data was changed. First five lines of new data:\n{}'
              ''.format(self.data[:5]))

