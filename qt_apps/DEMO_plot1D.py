"""
Example usage of StackScanner
'
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


from matplotlib.backends.qt4_compat import QtGui, QtCore
import numpy as np
from vistools.qt_widgets.OneDimStack import OneDimStackMainWindow
from collections import OrderedDict
import sys


def data_gen(num_sets=1, phase_shift=0.1, vert_shift=0.1, horz_shift=0.1):
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
        x.append(x_axis + horz_shift)
        y.append(np.sin(x_axis + idx * phase_shift) + idx * vert_shift)

    return x, y


class demo_1d(QtGui.QMainWindow):

    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        # Generate data
        num_sets = 100
        x_data, y_data = data_gen(num_sets=num_sets, phase_shift=0,
                        horz_shift=0, vert_shift=0)
        od = OrderedDict()
        for (lbl, x, y) in zip(range(num_sets), x_data, y_data):
            od[lbl] = (x, y)
        # init the 1d stack main window
        self._widget = OneDimStackMainWindow(data_dict=od)
        # add the demo buttons

        # declare button to generate data for testing/example purposes
        btn_datagen = QtGui.QPushButton("add data set",
                                        parent=self._widget._ctrl_widget)
        # declare button to append data to existing data set
        btn_append = QtGui.QPushButton("append data",
                                       parent=self._widget._ctrl_widget)

        btn_datagen.clicked.connect(self.datagen)
        btn_append.clicked.connect(self.append_data)
        layout = self._widget._ctrl_widget._widget.layout()

        layout.addRow("--- Demo Buttons ---", None)
        layout.addRow(btn_append, btn_datagen)

        # connect signals to test harness
        self.sig_append_demo_data.connect(
            self._widget._widget._canvas.sl_append_data)
        self.sig_add_demo_data.connect(
            self._widget._widget._canvas.sl_add_data)

        self.setCentralWidget(self._widget)

    # Qt Signals for Demo
    sig_append_demo_data = QtCore.Signal(list, list, list)
    sig_add_demo_data = QtCore.Signal(list, list, list)

    @QtCore.Slot()
    def append_data(self):
        num_sets = 7
        # get some fake data
        x, y = data_gen(num_sets, phase_shift=0, horz_shift=25, vert_shift=0)
        # emit the signal
        self.sig_append_demo_data.emit(range(num_sets), x, y)

    @QtCore.Slot()
    def datagen(self):
        num_data = 10
        self.sig_add_demo_data.emit(range(num_data),
                               *data_gen(num_data, phase_shift=0,
                                         horz_shift=0, vert_shift=0))

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    tt = demo_1d()
    tt.show()
    sys.exit(app.exec_())
