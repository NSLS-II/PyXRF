from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# grab the version from mpl which has done the work of smoothing over
# the differences
from matplotlib.backends.qt4_compat import QtGui, QtCore
from ..messenger.mpl.Stack1DMessenger import Stack1DMessenger
from ..messenger.mpl.CrossSection2DMessenger import CrossSection2DMessenger


class MainWindow(QtGui.QMainWindow):
    """
    OneDimStackMainWindow
    """
    messenger_classes = [Stack1DMessenger, CrossSection2DMessenger]

    def __init__(self, messenger_class, title=None, parent=None,
                 data_dict=None, key_list=None):
        QtGui.QMainWindow.__init__(self, parent)
        if title is None:
            title = str(messenger_class)
        self.setWindowTitle(title)
        # create view widget, control widget and messenger pass-through
        self._messenger = messenger_class(data_dict=data_dict,
                                             key_list=key_list)

        self._ctrl_widget = self._messenger._ctrl_widget
        self._display = self._messenger._display

        # finish the init
        self._display.setFocus()
        self.setCentralWidget(self._display)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea,
                           self._ctrl_widget)