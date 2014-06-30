"""
Example usage of StackScanner
'
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys

from matplotlib.backends.qt4_compat import QtGui, QtCore
import numpy as np

from vistools.qt_widgets.CrossSection2DWidget import CrossSection2DMainWindow


class data_gen(object):
    def __init__(self, length, func=None):
        self._len = length
        self._x, self._y = [_ * 2 * np.pi / 500 for _ in
                            np.ogrid[-500:500, -500:500]]
        self._rep = int(np.sqrt(length))

    def __len__(self):
        return self._len

    def __getitem__(self, k):
        kx = k // self._rep + 1
        ky = k % self._rep
        return np.sin(kx * self._x) * np.cos(ky * self._y) + 1.05

    @property
    def ndim(self):
        return 2

    @property
    def shape(self):
        len(self._x), len(self._y)

def data_gen(length):
       x, y = [_ * 2 * np.pi / 500 for _ in
                            np.ogrid[-500:500, -500:500]]
       rep = int(np.sqrt(length))
       data = {}
       lbls = []
       for idx in range(length):
            lbls.append(str(idx))
            kx = idx // rep + 1
            ky = idx % rep
            data[str(idx)] = np.sin(kx * x) * np.cos(ky * y) + 1.05
       return lbls, data


class StackExplorer(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent=parent)
        self.setWindowTitle('StackExplorer')
        key_list, data_dict = data_gen(25)
        self._main_window = CrossSection2DMainWindow(data_dict=data_dict,
                                                     key_list=key_list)

        self._main_window.setFocus()
        self.setCentralWidget(self._main_window)

app = QtGui.QApplication(sys.argv)
tt = StackExplorer()
tt.show()
sys.exit(app.exec_())
