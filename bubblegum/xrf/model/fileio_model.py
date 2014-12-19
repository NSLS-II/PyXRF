from pprint import pprint
from atom.api import Atom, Str, observe, Typed, Int, Unicode
import numpy as np
import os

# The following lines need to be updated.
# A better design to hook up with meta data store needs to be done.
folder = '/Users/Li/Research/X-ray/Research_work/all_code/nsls2_gui/nsls2_gui'
file = 'NSLS_X27.txt'
data_path = os.path.join(folder, file)
the_data = np.loadtxt(data_path)


class FileIOModel(Atom):
    tool_name = Str('PyXRF: X-ray Fluorescence Analysis Tool')
    folder_name = Str(folder)
    file_name = Str(file)
    data = Typed(np.ndarray)
    file_path = Str(data_path)
    load_status = Str()

    def __init__(self):
        self.load_data()

    @observe('folder_name', 'file_name')
    def update(self, changed):
        pprint(changed)
        if changed['type'] == 'create':
            return
        print('{} was changed from {} to {}'.format(changed['name'],
                                                    changed['oldvalue'],
                                                    changed['value']))
        self.file_path = os.path.join(self.folder_name, self.file_name)

    @observe('file_path')
    def path_changed(self, changed):
        self.load_data()

    @observe('data')
    def data_changed(self, data):
        print('The data was changed. First five lines of new data:\n{}'
              ''.format(self.data[:5]))

    def load_data(self):
        try:
            self.data = np.loadtxt(self.file_path)
            self.load_status = 'File {} is loaded successfully.'.format(self.file_name)
        except IOError:
            self.load_status = 'File {} doesn\'t exist.'.format(self.file_name)
        except ValueError:
            self.load_status = 'File {} can\'t be loaded. '.format(self.file_name)
