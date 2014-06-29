"""
Example usage of StackScanner
'
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys

from matplotlib.backends.qt4_compat import QtGui, QtCore
import numpy as np

from vistools.qt_widgets.CrossSection2DWidget import CrossSection2DWidget


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


class StackExplorer(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.setWindowTitle('StackExplorer')

        self._stack = CrossSection2DWidget(data_gen(25))

        self._stack.setFocus()
        self.setCentralWidget(self._stack)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea,
                           self._stack.ctrl_box_2)

app = QtGui.QApplication(sys.argv)
tt = StackExplorer()
tt.show()
sys.exit(app.exec_())
