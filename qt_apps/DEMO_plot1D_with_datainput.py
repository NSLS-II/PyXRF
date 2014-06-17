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
import sys, os


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


def get_files():
    files = QtGui.QFileDialog.getOpenFileName(parent=None,
                                              caption="File opener")
    for file in files:
        print(str(file))
    return files


def parse_files(files):
    lbls = []
    x_datasets = []
    y_datasets = []
    if os.path.isdir(files):
        for afile in files:
            f = open(afile, 'r')
            x, y = read_file(f)
            lbls.append(str(afile))
            x_datasets.append(x)
            y_datasets.append(y)
    else:
        f = open(files, 'r')
        x, y = read_file(f)
        lbls.append(str(files))
        x_datasets.append(x)
        y_datasets.append(y)

    return lbls, x_datasets, y_datasets


def read_file(afile):
    # probably need to call a specific file reader here

    # read data
    x = None
    y = None

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
        self.btn_loaddata = QtGui.QPushButton("load data set",
                                        parent=self._widget._ctrl_widget)

        self.btn_loaddata.clicked.connect(self.open_data)
        layout = self._widget._ctrl_widget._widget.layout()

        layout.addRow("--- File IO Buttons ---", None)
        layout.addRow(self.btn_loaddata, None)

        # connect signals to test harness
        self.sig_add_real_data.connect(
            self._widget._widget._canvas.sl_add_data)

        self.setCentralWidget(self._widget)

    # Qt Signals for Data loading
    sig_add_real_data = QtCore.Signal(list, list, list)

    @QtCore.Slot()
    def open_data(self):
        files = get_files()
        self.sig_add_real_data.emit(*parse_files(files))


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    tt = demo_1d()
    tt.show()
    sys.exit(app.exec_())
