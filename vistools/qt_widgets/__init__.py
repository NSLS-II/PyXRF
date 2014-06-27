
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six

from matplotlib.backends.qt4_compat import QtGui, QtCore


class AbstractDisplayWidget(QtGui.QWidget):
    """
    AbstractDisplayWidget class docstring.
    The purpose of this class and its daughter classes is simply to render the
    figure that the various plotting libraries use to present themselves
    """
    def __init__(self, parent=None):
        # init the QWidget
        super(AbstractDisplayWidget, self).__init__(parent=parent)
        # do nothing else