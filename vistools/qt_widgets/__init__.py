from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# grab the version from mpl which has done the work of smoothing over
# the differences
from matplotlib.backends.qt4_compat import QtGui, QtCore
from ..messenger.mpl.stack_1d import Stack1DMessenger
from ..messenger.mpl.cross_section_2d import CrossSection2DMessenger


class CrossSectionMainWindow(QtGui.QMainWindow):
    """
    MainWindow
    """

    def __init__(self, title=None, parent=None,
                 data_list=None, key_list=None):
        QtGui.QMainWindow.__init__(self, parent)
        if title is None:
            title = "2D Cross Section"
        self.setWindowTitle(title)
        # create view widget, control widget and messenger pass-through
        self._messenger = CrossSection2DMessenger(data_list=data_list,
                                             key_list=key_list)

        self._ctrl_widget = self._messenger._ctrl_widget
        self._display = self._messenger._display

        # finish the init
        self._display.setFocus()
        self.setCentralWidget(self._display)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea,
                           self._ctrl_widget)

        
class Stack1DMainWindow(QtGui.QMainWindow):
    """
    MainWindow
    """

    def __init__(self, title=None, parent=None,
                 data_list=None, key_list=None):
        QtGui.QMainWindow.__init__(self, parent)
        if title is None:
            title = "1D Stack"
        self.setWindowTitle(title)
        # create view widget, control widget and messenger pass-through
        self._messenger = Stack1DMessenger(data_list=data_list,
                                             key_list=key_list)

        self._ctrl_widget = self._messenger._ctrl_widget
        self._display = self._messenger._display

        # finish the init
        self._display.setFocus()
        self.setCentralWidget(self._display)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea,
                           self._ctrl_widget)