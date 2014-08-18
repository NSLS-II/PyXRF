# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# Redistribution and use in source and binary forms, with or without   #
# modification, are permitted provided that the following conditions   #
# are met:                                                             #
#                                                                      #
# * Redistributions of source code must retain the above copyright     #
#   notice, this list of conditions and the following disclaimer.      #
#                                                                      #
# * Redistributions in binary form must reproduce the above copyright  #
#   notice this list of conditions and the following disclaimer in     #
#   the documentation and/or other materials provided with the         #
#   distribution.                                                      #
#                                                                      #
# * Neither the name of the Brookhaven Science Associates, Brookhaven  #
#   National Laboratory nor the names of its contributors may be used  #
#   to endorse or promote products derived from this software without  #
#   specific prior written permission.                                 #
#                                                                      #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS  #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT    #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS    #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE       #
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,           #
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES   #
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR   #
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)   #
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,  #
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OTHERWISE) ARISING   #
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE   #
# POSSIBILITY OF SUCH DAMAGE.                                          #
########################################################################
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from .. import QtCore, QtGui
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar  # noqa
from matplotlib.figure import Figure
from matplotlib import colors
from ...backend.mpl import AbstractMPLDataView
from .. import AbstractMessenger
from .. import AbstractDisplayWidget


import logging
logger = logging.getLogger(__name__)


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
