from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.backends.qt4_compat import QtCore, QtGui
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar  # noqa
from matplotlib.figure import Figure
from matplotlib import colors
from ...backend.mpl import AbstractMPLDataView
from .. import AbstractMessenger
from .. import AbstractDisplayWidget


class AbstractMPLMessenger(AbstractMessenger):
    """
    docstring
    """

    def __init__(self, *args, **kwargs):
        # call up the inheritance toolchain
        super(AbstractMPLMessenger, self).__init__(*args, **kwargs)
        # init a display
        self._display = MPLDisplayWidget()
        self._fig = self._display._fig
        # set a default view
        self._view = AbstractMPLDataView(fig=self._fig)

    #@QtCore.Slot(colors.Normalize)
    def sl_update_norm(self, new_norm):
        """
        Updates the normalization function used for the color mapping
        """
        self._view.update_norm(new_norm)
        self.sl_update_view()

    @QtCore.Slot(colors.Colormap)
    def sl_update_cmap(self, cmap):
        """
        Updates the color map.  Currently takes a string, should probably be
        redone to take a cmap object and push the look up function up a layer
        so that we do not need the try..except block.
        """
        try:
            self._view.update_cmap(str(cmap))
        except ValueError:
            # do nothing and return
            return
        self.sl_update_view()

    @QtCore.Slot()
    def sl_update_view(self):
        self._view.replot()
        self._view._fig.canvas.draw()


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
