from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from . import AbstractDisplayWidget

import six

from matplotlib.backends.qt4_compat import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar  # noqa
from matplotlib.figure import Figure

__author__ = 'Eric-hafxb'


class MPLDisplayWidget(AbstractDisplayWidget):
    """
    AbstractDatatWidget class docstring
    """
    default_height = 24
    default_width = 24

    def __init__(self, parent=None, *args, **kwargs):
        super(MPLDisplayWidget, self).__init__(parent=parent, *args, **kwargs)

        # create a figure to display the mpl axes
        self._fig = Figure(figsize=(self.default_height, self.default_width))

        canvas = FigureCanvas(self._fig)
        FigureCanvas.setSizePolicy(canvas,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(canvas)

        # create the mpl toolbar
        self._mpl_toolbar = NavigationToolbar(canvas=self._fig.canvas,
                                              parent=self)
        # create a layout manager
        layout = QtGui.QVBoxLayout()
        # add the mpl toolbar to the layout
        layout.addWidget(self._mpl_toolbar)
        # add the mpl canvas to the layout
        layout.addWidget(self._fig.canvas)
        # add the layout to the widget
        self.setLayout(layout)

    def draw(self):
        self._fig.canvas.draw()