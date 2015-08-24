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
from collections import OrderedDict


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
    data_dict_keys = List()
    file_opt = Int(0)
    plot_opt = Int(0)
    plot_item = Str()
    dict_to_plot = Dict()

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

    x_pos = Typed(object)
    y_pos = Typed(object)

    pixel_or_pos = Int(0)
    pixel_or_pos_for_plot = Typed(object)

    def __init__(self):
        self.fig = plt.figure()
        self.x_pos = []
        self.y_pos = []
        self.pixel_or_pos_for_plot = None

    def data_dict_update(self, change):
        """
        Observer function to be connected to the fileio model
        in the top-level gui.py startup

        Parameters
        ----------
        changed : dict
            This is the dictionary that gets passed to a function
            with the @observe decorator
        """
        self.data_dict = change['value']

    @observe('data_dict', 'data_dict_keys')
    def init_plot_status(self, change):
        self.data_dict_keys = self.data_dict.keys()
        logger.info('2D image display: {}'.format(self.data_dict.keys()))

        scaler_groups = [v for v in self.data_dict.keys() if 'scaler' in v]
        if len(scaler_groups) > 0:
            self.scaler_group_name = scaler_groups[0]
            self.scaler_items = [' '] + self.data_dict[self.scaler_group_name].keys()
            self.scaler_data = None

        if 'positions' in self.data_dict:
            print('get pos {}'.format(self.data_dict['positions'].keys()))
            self.x_pos = self.data_dict['positions']['x_pos'][0, :]
            self.y_pos = self.data_dict['positions']['y_pos'][:, 0]

    @observe('file_opt')
    def _update_file(self, change):
        if self.file_opt > 0:
            print('file opt:{}'.format(self.file_opt))
            namelist = self.data_dict.keys()
            self.file_name = namelist[self.file_opt]

    @observe('plot_item')
    def _update_file(self, change):
        if self.file_opt > 0:
            self.items_in_group = []
            self.dict_to_plot = self.data_dict[self.plot_item]
            self.items_in_group = self.dict_to_plot.keys()
            self.plot_all = False

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

    @observe('pixel_or_pos')
    def _update_pp(self, change):
        if change['type'] != 'create':
            if change['value'] == 0:
                self.pixel_or_pos_for_plot = None
            elif change['value'] > 0 and len(self.x_pos) > 0 and len(self.y_pos) > 0:
                self.pixel_or_pos_for_plot = (self.x_pos[0], self.x_pos[-1],
                                              self.y_pos[0], self.y_pos[-1])
            self.show_image()

    @observe('plot_all')
    def _update_all_plot(self, change):
        if self.plot_all is True:
            self.set_initial_stat(bool_val=True)
        else:
            self.set_initial_stat(bool_val=False)
        #self.show_image()

    def set_initial_stat(self, bool_val=False):
        """
        Set up initial plotting status for all the 2D images.
        """
        self.stat_dict.clear()
        self.stat_dict = {k: bool_val for k in self.items_in_group}

    def update_plot(self):
        #self.fig.tight_layout(pad=4.0, w_pad=0.8, h_pad=0.8)
        #self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def show_image(self):
        self.fig.clf()
        stat_temp = self.get_activated_num()
        stat_temp = OrderedDict(sorted(six.iteritems(stat_temp), key=lambda x: x[0]))

        low_lim = 1e-4  # define the low limit for log image
        ic_norm = 10000  # multiply by this value for ic normalization
        plot_interp = 'Nearest'
        name_not_scalable = ['r_squared']  # do not apply scaler norm on those data

        if self.scaler_data is not None:
            if np.max(self.scaler_data) < 1.0:  # use current as ic, such as SRX
                ic_norm = 1.0
            if len(self.scaler_data[self.scaler_data == 0]) > 0:
                logger.warning('scaler data has zero values at {}'.format(np.where(self.scaler_data == 0)))
                self.scaler_data[self.scaler_data == 0] = np.mean(self.scaler_data)
                logger.warning('Use mean value {} instead for those points'.format(np.mean(self.scaler_data)))

        if self.color_opt == 'Orange':
            grey_use = cm.Oranges
        elif self.color_opt == 'Color':
            grey_use = None
        else:
            grey_use = cm.Greys_r

        ncol = int(np.ceil(np.sqrt(len(stat_temp))))
        nrow = int(np.ceil(len(stat_temp)/float(ncol)))

        a_pad_v = 0.8
        a_pad_h = 0.3

        grid = ImageGrid(self.fig, 111,
                         nrows_ncols=(nrow, ncol),
                         axes_pad=(a_pad_v, a_pad_h),
                         cbar_location='right',
                         cbar_mode='each',
                         cbar_size='7%',
                         cbar_pad='2%',
                         share_all=True)

        if self.scale_opt == 'Linear':
            for i, (k, v) in enumerate(six.iteritems(stat_temp)):

                if self.scaler_data is not None:
                    if k in name_not_scalable:
                        data_dict = self.dict_to_plot[k]
                    else:
                        data_dict = self.dict_to_plot[k]/self.scaler_data*ic_norm

                else:
                    data_dict = self.dict_to_plot[k]

                im = grid[i].imshow(data_dict,
                                    cmap=grey_use,
                                    interpolation=plot_interp,
                                    extent=self.pixel_or_pos_for_plot)
                grid_title = k #self.file_name+'_'+str(k)
                if self.pixel_or_pos_for_plot is not None:
                    title_x = self.pixel_or_pos_for_plot[0]
                    title_y = self.pixel_or_pos_for_plot[3] + (self.pixel_or_pos_for_plot[3] -
                                                               self.pixel_or_pos_for_plot[2])*0.04

                else:
                    title_x = 0
                    title_y = - data_dict.shape[0]*0.05
                grid[i].text(title_x, title_y, grid_title)
                grid.cbar_axes[i].colorbar(im)
        else:
            for i, (k, v) in enumerate(six.iteritems(stat_temp)):

                if self.scaler_data is not None:
                    if k in name_not_scalable:
                        data_dict = self.dict_to_plot[k]
                    else:
                        data_dict = self.dict_to_plot[k]/self.scaler_data*ic_norm

                else:
                    data_dict = self.dict_to_plot[k]

                maxz = np.max(data_dict)

                im = grid[i].imshow(data_dict,
                                    norm=LogNorm(vmin=low_lim*maxz,
                                                 vmax=maxz),
                                    cmap=grey_use,
                                    interpolation=plot_interp,
                                    extent=self.pixel_or_pos_for_plot)

                grid_title = k #self.file_name+'_'+str(k)
                if self.pixel_or_pos_for_plot is not None:
                    print(self.pixel_or_pos_for_plot)
                    title_x = self.pixel_or_pos_for_plot[0]
                    title_y = self.pixel_or_pos_for_plot[3] + (self.pixel_or_pos_for_plot[3] -
                                                               self.pixel_or_pos_for_plot[2])*0.05

                else:
                    title_x = 0
                    title_y = - data_dict.shape[0]*0.05
                grid[i].text(title_x, title_y, grid_title)
                grid.cbar_axes[i].colorbar(im)
        self.update_plot()

    def get_activated_num(self):
        return {k: v for (k, v) in six.iteritems(self.stat_dict) if v is True}
