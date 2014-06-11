"""
Example usage of StackScanner
'
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.backends.qt4_compat import QtGui, QtCore
from matplotlib.ticker import NullLocator
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas  # noqa
import numpy as np
from matplotlib.figure import Figure

import sys

class data_gen(object):
    def __init__(self, length, func=None):
        self._len = length
        self._x = [1, 2, 3, 8, 4, 5]
        self._rep = int(np.sqrt(length))

    def __len__(self):
        return self._len

    def __getitem__(self, k):
        kx = k // self._rep + 1
        return kx * self._x

    @property
    def ndim(self):
        return 2

    @property
    def shape(self):
        len(self._x)

class OneDimCrossSectionViewer(object):
    # The default number of bins to use in the _percentile_limit method
    _DEFAULT_NUM_BINS = 1000

    def __init__(self, fig, init_image,
                 cmap=None,
                 norm=None,
                 limit_func=None,
                 limit_args=None):

        # stash the figure
        self.fig = fig
        # clean it
        self.fig.clf()

        # make the main axes
        # (in matplotlib speak the 'main axes' is the 2d
        # image in the middle of the canvas)
        self._im_ax = fig.add_subplot(1, 1, 1)
        self._im_ax.set_aspect('equal')
        self._im_ax.xaxis.set_major_locator(NullLocator())
        self._im_ax.yaxis.set_major_locator(NullLocator())
        self._imdata = init_image
        self._im = self._im_ax.plot(init_image)
        # self._im = self._im_ax.imshow(init_image, cmap=cmap, norm=norm,
        # interpolation='none', aspect='equal')

class OneDimCrossSectionCanvas(FigureCanvas):
    """
    This is a thin wrapper around images.CrossSectionViewer which
    manages the Qt side of the figure creation and provides slots
    to pass commands down to the gui-independent layer
    """
    def __init__(self, init_image, parent=None):
        width = height = 24
        self.fig = Figure(figsize=(width, height))
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        self._xsection = OneDimCrossSectionViewer(self.fig, init_image)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

class OneDimScannerWidget(QtGui.QWidget):

    def __init__(self, stack, page_size=10, parent=None):
        QtGui.QWidget.__init__(self, parent)
        # v_box_layout = QtGui.QVBoxLayout()

        self._stack = stack

        self._len = len(stack)

        # create the viewer widget
        self._widget = OneDimCrossSectionCanvas(stack[0])
        
        # create a layout manager for the widget
        v_box_layout = QtGui.QVBoxLayout()
        
        # add the 1D widget to the layout
        v_box_layout.addWidget(self._widget)
        
        self.setLayout(v_box_layout)


class OneDimStackExplorer(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.setWindowTitle('1-D Stack Plotting')

        # Need generate data
        self._stack = OneDimScannerWidget(data_gen(25))

        self._stack.setFocus()
        self.setCentralWidget(self._stack)

app = QtGui.QApplication(sys.argv)
tt = OneDimStackExplorer()
tt.show()
sys.exit(app.exec_())
