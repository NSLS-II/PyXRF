'''
Created on Jun 16, 2014

@author: edill
'''

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six

from matplotlib.backends.qt4_compat import QtGui, QtCore

from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar  # noqa
from matplotlib.figure import Figure

from . import AbstractDisplayWidget



class PlotWidget(QtGui.QMainWindow):
    """
    Top level container for one control widget and one data view widget
    """
    def __init__(self, parent=None, embed=True):
        """
        init doc-string

        Parameters
        ----------
        parent : QWidget
            parent widget


        embed : bool
            if widget should be embeddable in another widget (if true)
            or a stand-alone window (false)
        """
        # explictily call up the stack
        # TODO sort out qt5 super rules
        QtGui.QMainWindow.__init__(self, parent)
        # set flag is this should be widget or a main window
        if embed:
            self.setWindowFlags(QtCore.Qt.Widget)

        # dummy figure size in inches
        width = height = 12
        # create the Figure object
        self._fig = Figure(figsize=(width, height))
        # create the canvas (and connect it to the figure)
        self._canvas = FigureCanvas(self.fig, parent=self)
        # set the canvas to fill available space
        self._canvas.setSizePolicy(QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        self._canvas.updateGeometry()

        # set the central widget to be the canvas
        self.setCentralWidget(self._canvas)

        # create the control box
        control_box_name = 'controls'
        self._config = QtGui.QDockWidget(control_box_name)
        self._ctl_widget = ControlContainer(self.ctrl_box_2)

    @property
    def fig(self):
        """
        The `Figure` object for this widget.
        """
        return self._fig

    @property
    def control(self):
        return self._ctl_widget
