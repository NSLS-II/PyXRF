from pprint import pprint
from atom.api import Atom, Str, observe, Typed, Int, Unicode
import numpy as np
import os


class FileIOModel(Atom):
    tool_name = Str('PyXRF: X-ray Fluorescence Analysis Tool')
    folder_name = Str('')
    file_name = Str('')
    data = Typed(object)
    file_path = Str()
    load_status = Str()
    #tool_name = 'PyXRF'

    @observe('folder_name', 'file_name')
    def update(self, changed):
        pprint(changed)
        if changed['type'] == 'create':
            return
        print('{} was changed from {} to {}'.format(changed['name'],
                                                    changed['oldvalue'],
                                                    changed['value']))
        #if changed['name'] == 'file_name':
        #    self.load_data()

    @observe('data')
    def data_changed(self, data):
        print('The data was changed. First five lines of new data:\n{}'
              ''.format(self.data[:5]))

    def set_path(self):
        self.file_path = os.path.join(self.folder_name, self.file_name)
        if os.path.exists(self.file_path):
            self.load_status = 'File {0} is loaded successfully.'.format(self.file_name)
            self.data = np.loadtxt(self.file_path)
        else:
            self.load_status = 'File {0} does not exist.'.format(self.file_name)

    @observe('load_status')
    def _new_status(self, changed):
        pprint('status changes.')
