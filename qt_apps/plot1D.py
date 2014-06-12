"""
Example usage of StackScanner
'
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.backends.qt4_compat import QtGui, QtCore
from matplotlib.ticker import NullLocator
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas  # noqa
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar  # noqa
import numpy as np
from matplotlib.figure import Figure
from vistools.qt_widgets.common import LimitSpinners

import sys


def data_gen(num_sets, phase_shift=0.1, vertical_shift=0.1):
    """
    Generate some data

    Parameters
    ----------
    num_sets: int
        number of 1-D data sets to generate

    Returns
    -------
    x : np.ndarray
        x-coordinates
    y : list of np.ndarray
        y-coordinates
    """
    x = np.arange(0, 25, .01)
    y = []
    for idx in range(num_sets):
        y.append(np.sin(x + idx * phase_shift) + idx * vertical_shift)

    return x, y


class OneDimCrossSectionViewer(object):

    def __init__(self, fig, x, y_data,
                 cmap=None,
                 norm=None):

        # stash the figure
        self._fig = fig
        # save the x-axis
        self._x = x
        # save the y-data
        self._y = y_data
        # save the colormap
        self._cmap = cmap
        # save the normalization
        self._norm = norm
        self._x_offset = 0
        self._y_offset = 0
        self.replot()

    def set_y_offset(self, y_offset):
        self._y_offset = y_offset
        self.replot()

    def set_x_offset(self, x_offset):
        self._x_offset = x_offset
        self.replot()

    def replot(self):
        print("replotting with x_offset={0} and y_offset={1}".format(self._x_offset, self._y_offset))
        self._fig.clf()

        # make the main axes
        # (in matplotlib speak the 'main axes' is the 2d
        # image in the middle of the canvas)
        self._im_ax = self._fig.add_subplot(1, 1, 1)
        self._im_ax.set_aspect('equal')
        for idx in range(len(self._y)):
            self._im_ax.plot(self._x + idx * self._x_offset,
                             self._y[idx] + idx * self._y_offset)


def plot_diff(ax, data, norm=None):
    """
    Parameters
    ----------
    ax : Axes
        The axes to plot into
    data : ndarray
        MxN array of data
    q_vec : ndarray
        length M array of q-values
    c : ndarray
        scalars to be used for color mapping
    norm : scalar, ndarray or None
        if None, defaults to 1.  Valure used to normalize data.

    Returns
    -------
    lns : list
        list of line artists returned
    """
    if norm is None:
        norm = 1
    lns = []
    data = data / norm

    for d in data:
        ln, = ax.plot(d)
        lns.append(ln)
    return lns


class ControlWidget(QtGui.QDockWidget):
    """
    Control widget class docstring
    """
    def __init__(self, name):
        """
        init docstring

        Parameters
        ----------
        name : String
            Name of the control widget
        """
        QtGui.QDockWidget.__init__(self, name)
        # make the control widget float
        self.setFloating(True)
        # add a widget that lives in the floating control widget
        self._widget = QtGui.QWidget(self)

        self._x_shift_spinbox = QtGui.QDoubleSpinBox(parent=self)
        self._y_shift_spinbox = QtGui.QDoubleSpinBox(parent=self)

        layout = QtGui.QFormLayout()
        layout.addRow("x_shift", self._x_shift_spinbox)
        layout.addRow("y_shift", self._y_shift_spinbox)
        # set the widget layout
        self._widget.setLayout(layout)
        self.setWidget(self._widget)


class OneDimCrossSectionCanvas(FigureCanvas):
    """
    This is a thin wrapper around images.CrossSectionViewer which
    manages the Qt side of the figure creation and provides slots
    to pass commands down to the gui-independent layer
    """
    def __init__(self, x, y, parent=None):
        width = height = 24
        self.fig = Figure(figsize=(width, height))
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        self._view = OneDimCrossSectionViewer(self.fig, x, y)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    @QtCore.Slot(float)
    def sl_update_x_offset(self, x_offset):
        self._view.set_x_offset(x_offset)

    @QtCore.Slot(float)
    def sl_update_y_offset(self, y_offset):
        self._view.set_y_offset(y_offset)


class OneDimScannerWidget(QtGui.QWidget):

    def __init__(self, x_axis, y_data, page_size=10, parent=None):
        QtGui.QWidget.__init__(self, parent)
        # v_box_layout = QtGui.QVBoxLayout()

        self._x_axis = x_axis
        self._y_data = y_data

        self._len = len(y_data)

        # create the viewer widget
        self._canvas = OneDimCrossSectionCanvas(x_axis, y_data)

        # create a layout manager for the widget
        v_box_layout = QtGui.QVBoxLayout()

        self.mpl_toolbar = NavigationToolbar(self._canvas, self)
        # add toolbar
        v_box_layout.addWidget(self.mpl_toolbar)
        # add the 1D widget to the layout
        v_box_layout.addWidget(self._canvas)

        self.setLayout(v_box_layout)


class OneDimStackExplorer(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.setWindowTitle('1-D Stack Plotting')

        x, y = data_gen(25)
        # Need generate data
        self._widget = OneDimScannerWidget(x, y)
        self._ctrl = ControlWidget("1-D Stack Controls")
        self._ctrl._x_shift_spinbox.valueChanged.connect(self._widget._canvas.sl_update_x_offset)
        self._ctrl._y_shift_spinbox.valueChanged.connect(self._widget._canvas.sl_update_y_offset)

        self._widget.setFocus()
        self.setCentralWidget(self._widget)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea,
                           self._ctrl)

app = QtGui.QApplication(sys.argv)
tt = OneDimStackExplorer()
tt.show()
sys.exit(app.exec_())
