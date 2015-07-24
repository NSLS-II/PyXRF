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

__author__ = 'Li Li'

import six
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm

from mpl_toolkits.axes_grid1 import ImageGrid
from atom.api import Atom, Str, observe, Typed, Int, List, Dict, Bool


import logging
logger = logging.getLogger(__name__)


class DrawImageAdvanced(Atom):
    """
    This class performs 2D image rendering, such as showing multiple
    2D roi images based on user's selection.

    Attributes
    ----------
    img_data : dict
        dict of 2D array
    fig : object
        matplotlib Figure
    file_name : str
    stat_dict : dict
        determine which image to show
    data_dict : dict
        save multiple data
    file_opt : int
        which file is chosen
    plot_opt : int
        show plot or not
    single_file : dict
        image data for one given file
    """

    img_data = Typed(object)
    fig = Typed(Figure)
    file_name = Str()
    stat_dict = Dict()
    data_dict = Dict()
    file_opt = Int(0)
    plot_opt = Int(0)
    single_file = Dict()
    scale_opt = Str('Linear')
    color_opt = Str('Orange')

    group_names = List()
    group_name = Str()
    items_in_group = List()

    scaler_group_name = Str()
    scaler_items = List()
    scaler_name = Str()
    scaler_data = Typed(object)

    select_stat_dict = Dict()

    plot_all = Bool(False)

    def __init__(self):
        self.fig = plt.figure()

    @observe('data_dict')
    def init_plot_status(self, change):
        logger.info('2D image display: {}'.format(self.data_dict.keys()))

        self.set_initial_stat()

        self.group_names = [' '] + self.data_dict.keys()
        self.group_name = self.group_names[1]
        scaler_groups = [v for v in self.data_dict.keys() if 'scaler' in v]

        self.scaler_group_name = scaler_groups[0]
        self.scaler_items = [' '] + self.data_dict[self.scaler_group_name].keys()
        self.scaler_data = None

        print(self.stat_dict)
        self.select_stat_dict = self.stat_dict[self.group_name]

    @observe('group_name')
    def _change_img_group(self, change):
        if self.group_name != ' ':
            self.items_in_group = []
            self.items_in_group = self.data_dict[self.group_name].keys()
            self.select_stat_dict = self.stat_dict[self.group_name]
            print('update {}'.format(self.select_stat_dict))

    @observe('scaler_name')
    def _get_scaler_data(self, change):
        if self.scaler_name == ' ':
            self.scaler_data = None
        else:
            self.scaler_data = self.data_dict[self.scaler_group_name][self.scaler_name]
            print('scaler data shape: {}'.format(self.scaler_data.shape))

    @observe('scale_opt', 'color_opt')
    def _update_scale(self, change):
        if change['type'] != 'create':
            self.show_image()

    @observe('plot_all')
    def _update_all_plot(self, change):
        if self.plot_all == True:
            for k in six.iterkeys(self.stat_dict[self.group_name]):
                self.stat_dict[self.group_name][k] = True
        else:
            for k in six.iterkeys(self.stat_dict[self.group_name]):
                self.stat_dict[self.group_name][k] = False
        self.show_image()

    def set_initial_stat(self):
        """
        Set up initial plotting status for all the 2D images.
        """
        stat_dict = {}
        for k, v in six.iteritems(self.data_dict):
            if 'roi' in k and 'each' not in k:
                temp = {m: True for m in six.iterkeys(v)}
            else:
                temp = {m: False for m in six.iterkeys(v)}
            stat_dict.update({k: temp})
        self.stat_dict = stat_dict

    def update_plot(self):
        self.fig.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2)
        self.fig.canvas.draw_idle()

    def show_image(self):
        self.fig.clf()
        stat_temp = self.get_activated_num()

        low_lim = 1e-4  # define the low limit for log image
        plot_interp = 'Nearest'

        if self.color_opt == 'Orange':
            grey_use = cm.Oranges
        elif self.color_opt == 'Color':
            grey_use = None
        else:
            grey_use = cm.Greys_r

        ncol = int(np.ceil(np.sqrt(len(stat_temp))))
        nrow = int(np.ceil(len(stat_temp)/float(ncol)))

        grid = ImageGrid(self.fig, 111,
                         nrows_ncols=(nrow, ncol),
                         axes_pad=(0.5, 0.5),
                         cbar_location='right',
                         cbar_mode='each',
                         cbar_size='7%',
                         cbar_pad='2%',
                         share_all=True)

        for i, (k, v) in enumerate(sorted(stat_temp)):
            if self.scale_opt == 'Linear':
                if self.scaler_data is not None:
                    data_dict = self.data_dict[k][v]/self.scaler_data
                else:
                    data_dict = self.data_dict[k][v]
                im = grid[i].imshow(data_dict,
                                    cmap=grey_use,
                                    interpolation=plot_interp)

            else:
                maxz = np.max(self.data_dict[k][v])
                im = grid[i].imshow(self.data_dict[k][v],
                                    norm=LogNorm(vmin=low_lim*maxz,
                                                 vmax=maxz),
                                    cmap=grey_use)
            grid_title = '{}'.format(k+'_'+v)
            grid[i].text(0, -1, grid_title)
            grid.cbar_axes[i].colorbar(im)
        self.update_plot()

    def get_activated_num(self):
        data_temp = []
        for k, v in six.iteritems(self.stat_dict):
            for m in six.iterkeys(v):
                if v[m]:
                    data_temp.append((k, m))
        return data_temp
