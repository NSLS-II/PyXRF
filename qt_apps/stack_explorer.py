"""
Example usage of StackScanner
'
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.backends.qt4_compat import QtGui
import numpy as np
import vistools.qt_widgets as qt_widgets
import sys


def _gen_test_data(n):
    out = []
    x, y = np.ogrid[-500:500, -500:500]
    for j in xrange(n):
        out.append(np.exp(-((x - 250) ** 2 + (y - 250) ** 2) / (20 + 5 * j) ** 2))
    return np.array(out)


class StackExplorer(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.setWindowTitle('StackExplorer')
        self._stack = qt_widgets.StackScanner(_gen_test_data(15))

        self.main_widget = QtGui.QWidget(self)

        l = QtGui.QVBoxLayout(self.main_widget)

        l.addWidget(self._stack)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)


app = QtGui.QApplication(sys.argv)
tt = StackExplorer()
tt.show()
sys.exit(app.exec_())
