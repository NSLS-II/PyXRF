'''
Created on Jun 16, 2014

@author: edill
'''

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.backends.qt4_compat import QtGui

from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar  # noqa
from matplotlib.figure import Figure


class AbstractDisplayWidget(QtGui.QWidget):
    """
    AbstractDisplayWidget class docstring.
    The purpose of this class and its daughter classes is simply to render the
    figure that the various plotting libraries use to present themselves
    """
    def __init__(self, parent=None):
        # init the QWidget
        QtGui.QWidget.__init__(self, parent)

        # do nothing else


class MPLDisplayWidget(AbstractDisplayWidget):
    """
    AbstractDatatWidget class docstring
    """
    default_height = 24
    default_width = 24

    def __init__(self, parent=None, data=None):
        # init the QWidget
        AbstractDisplayWidget.__init__(parent=parent)

        # create a figure to display the mpl axes
        fig = Figure(figsize=(self.default_height, self.default_width))
        self._drawable = fig

        # create the mpl toolbar
        self._mpl_toolbar = NavigationToolbar(self._drawable.canvas, self)
        # create a layout manager
        layout = QtGui.QVBoxLayout()
        # add the mpl toolbar to the layout
        layout.addWidget(self._mpl_toolbar)
        # add the mpl canvas to the layout
        layout.addWidget(self._drawable.canvas)
        # add the layout to the widget
        self.setLayout(layout)

    def draw(self):
        self._drawable.canvas.draw()
