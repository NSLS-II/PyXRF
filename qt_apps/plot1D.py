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


def data_gen(num_sets, phase_shift=0.1, vert_shift=0.1, horz_shift=0.1):
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
    x_axis = np.arange(0, 25, .01)
    x = []
    y = []
    for idx in range(num_sets):
        x.append(x_axis + idx * horz_shift)
        y.append(np.sin(x_axis + idx * phase_shift) + idx * vert_shift)

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
        self._autoscale = False
        self._x_offset = 0
        self._y_offset = 0
        self.plot()
        self._artists = []

    def set_y_offset(self, y_offset):
        self._y_offset = y_offset
        self.replot()

    def set_x_offset(self, x_offset):
        self._x_offset = x_offset
        self.replot()

    def add_line(self, x, y):
        self._x.append(x)
        self._y.append(y)

    def replot(self):
        print("replotting with x_offset={0} and y_offset={1}".
              format(self._x_offset, self._y_offset))

        for idx in range(len(self._im_ax.lines)):

            self._im_ax.lines[idx].set_xdata(self._x[idx] +
                                             idx * self._x_offset)
            self._im_ax.lines[idx].set_ydata(self._y[idx] +
                                             idx * self._y_offset)

        if(self._autoscale):
            (min_x, max_x, min_y, max_y) = self.find_range()
            self._im_ax.set_xlim(min_x, max_x)
            self._im_ax.set_ylim(min_y, max_y)

    def set_auto_scale(self, is_autoscaling):
        print("autoscaling: {0}".format(is_autoscaling))
        self._autoscale = is_autoscaling
        self.replot()

    def find_range(self):
        """
        Find the min/max in x and y

        Returns
        -------
        (min_x, max_x, min_y, max_y)
        """
        # find min/max in x and y
        min_x = np.zeros(len(self._im_ax.lines))
        max_x = np.zeros(len(self._im_ax.lines))
        min_y = np.zeros(len(self._im_ax.lines))
        max_y = np.zeros(len(self._im_ax.lines))

        for idx in range(len(self._im_ax.lines)):
            min_x[idx] = np.min(self._im_ax.lines[idx].get_xdata())
            max_x[idx] = np.max(self._im_ax.lines[idx].get_xdata())
            min_y[idx] = np.min(self._im_ax.lines[idx].get_ydata())
            max_y[idx] = np.max(self._im_ax.lines[idx].get_ydata())

        return (np.min(min_x), np.max(max_x), np.min(min_y), np.max(max_y))

    def plot(self):
        # (in matplotlib speak the 'main axes' is the 2d
        # image in the middle of the canvas)
        self._im_ax = self._fig.add_subplot(1, 1, 1)
        self._im_ax.set_aspect('equal')
        for idx in range(len(self._y)):
            self._im_ax.plot(self._x[idx] + idx * self._x_offset,
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

        # create the offset spin boxes
        self._x_shift_spinbox = QtGui.QDoubleSpinBox(parent=self)
        self._y_shift_spinbox = QtGui.QDoubleSpinBox(parent=self)

        # create the offset step size spin boxes
        self._x_shift_step_spinbox = QtGui.QDoubleSpinBox(parent=self)
        self._y_shift_step_spinbox = QtGui.QDoubleSpinBox(parent=self)

        # set the min/max limits
        default_min = -100
        default_max = 100
        default_step = 0.01
        self._x_shift_spinbox.setMinimum(default_min)
        self._y_shift_spinbox.setMinimum(default_min)
        self._x_shift_step_spinbox.setMinimum(default_min)
        self._y_shift_step_spinbox.setMinimum(default_min)
        self._x_shift_spinbox.setMaximum(default_max)
        self._y_shift_spinbox.setMaximum(default_max)
        self._x_shift_step_spinbox.setMaximum(default_max)
        self._y_shift_step_spinbox.setMaximum(default_max)
        self._x_shift_spinbox.setSingleStep(default_step)
        self._y_shift_spinbox.setSingleStep(default_step)
        self._x_shift_step_spinbox.setSingleStep(default_step)
        self._y_shift_step_spinbox.setSingleStep(default_step)

        self._x_shift_step_spinbox.setValue(default_step)
        self._y_shift_step_spinbox.setValue(default_step)

        # connect the signals
        self._x_shift_step_spinbox.valueChanged.connect(self._x_shift_spinbox.setSingleStep)
        self._y_shift_step_spinbox.valueChanged.connect(self._y_shift_spinbox.setSingleStep)

        self._autoscale_box = QtGui.QCheckBox(parent=self)

        # create the layout
        layout = QtGui.QFormLayout()

        # add the controls to the layout
        layout.addRow("--- Curve Shift ---", None)
        layout.addRow("x shift", self._x_shift_spinbox)
        layout.addRow("y shift", self._y_shift_spinbox)
        layout.addRow("x step", self._x_shift_step_spinbox)
        layout.addRow("y step", self._y_shift_step_spinbox)
        layout.addRow("--- Misc. ---", None)
        layout.addRow("autoscale data view", self._autoscale_box)
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
        self.draw()

    @QtCore.Slot(float)
    def sl_update_y_offset(self, y_offset):
        self._view.set_y_offset(y_offset)
        self.draw()

    @QtCore.Slot(bool)
    def sl_update_autoscaling(self, is_autoscaling):
        self._view.set_auto_scale(is_autoscaling)
        self.draw()


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
        # Generate data
        x, y = data_gen(100, phase_shift=0, horz_shift=0, vert_shift=0)
        # create view widget and control widget
        self._widget = OneDimScannerWidget(x, y)
        self._ctrl = ControlWidget("1-D Stack Controls")

        # connect signals/slots between view widget and control widget
        self._ctrl._x_shift_spinbox.valueChanged.connect(self._widget._canvas.sl_update_x_offset)
        self._ctrl._y_shift_spinbox.valueChanged.connect(self._widget._canvas.sl_update_y_offset)
        self._ctrl._autoscale_box.clicked.connect(self._widget._canvas.sl_update_autoscaling)

        self._widget.setFocus()
        self.setCentralWidget(self._widget)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea,
                           self._ctrl)

app = QtGui.QApplication(sys.argv)
tt = OneDimStackExplorer()
tt.show()
sys.exit(app.exec_())
