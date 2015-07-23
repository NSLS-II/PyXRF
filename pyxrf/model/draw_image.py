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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm

from atom.api import Atom, Str, observe, Typed, Int, List, Dict

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
    color_opt = Str('Color')

    group_names = List()
    group_name = Str()
    items_in_group = List()

    scaler_group_name = Str()
    scaler_items = List()
    scaler_name = Str()
    scaler_data = Typed(object)

    def __init__(self):
        self.fig = plt.figure()

    @observe('data_dict')
    def init_plot_status(self, change):
        logger.info('2D image display: {}'.format(self.data_dict.keys()))
        self.set_initial_stat()
        self.group_names = [' '] + self.data_dict.keys()

        scaler_groups = [v for v in self.data_dict.keys() if 'scaler' in v]
        self.scaler_group_name = scaler_groups[0]
        self.scaler_items = [' '] + self.data_dict[self.scaler_group_name].keys()
        self.scaler_data = None

    @observe('group_name')
    def _change_img_group(self, change):
        self.items_in_group = []
        self.items_in_group = self.data_dict[self.group_name].keys()

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

    def set_initial_stat(self):
        """
        Set up initial plotting status for all the 2D images.
        """
        for k, v in six.iteritems(self.data_dict):
            temp = {m: False for m in six.iterkeys(v)}
            self.stat_dict.update({k: temp})

    def update_plot(self):
        self.fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
        self.fig.canvas.draw()

    def show_image(self):
        self.fig.clf()
        stat_temp = self.get_activated_num()

        fontsize = 10

        low_lim = 1e-4  # define the low limit for log image

        if self.color_opt == 'Color':
            grey_use = None
        else:
            grey_use = cm.Greys_r

        if len(stat_temp) == 1:
            ax = self.fig.add_subplot(111)
            for k, v in sorted(stat_temp):
                if self.scale_opt == 'Linear':
                    if self.scaler_data is not None:
                        data_dict = self.data_dict[k][v]/self.scaler_data
                    else:
                        data_dict = self.data_dict[k][v]
                    im = ax.imshow(data_dict,
                                   cmap=grey_use)
                else:
                    maxz = np.max(self.data_dict[k][v])
                    im = ax.imshow(self.data_dict[k][v],
                                   norm=LogNorm(vmin=low_lim*maxz,
                                                vmax=maxz),
                                   cmap=grey_use)
                ax.set_title('{}'.format(k+'_'+v), fontsize=fontsize)

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                self.fig.colorbar(im, cax=cax)

        elif len(stat_temp) == 2:
            for i, (k, v) in enumerate(sorted(stat_temp)):
                ax = self.fig.add_subplot(eval('12'+str(i+1)))
                if self.scale_opt == 'Linear':
                    if self.scaler_data is not None:
                        data_dict = self.data_dict[k][v]/self.scaler_data
                    else:
                        data_dict = self.data_dict[k][v]
                    im = ax.imshow(data_dict,
                                   cmap=grey_use)
                else:
                    maxz = np.max(self.data_dict[k][v])
                    im = ax.imshow(self.data_dict[k][v],
                                   norm=LogNorm(vmin=low_lim*maxz,
                                                vmax=maxz),
                                   cmap=grey_use)
                ax.set_title('{}'.format(k+'_'+v), fontsize=fontsize)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                self.fig.colorbar(im, cax=cax)

        elif len(stat_temp) <= 4 and len(stat_temp) > 2:
            for i, (k, v) in enumerate(sorted(stat_temp)):
                ax = self.fig.add_subplot(eval('22'+str(i+1)))
                if self.scale_opt == 'Linear':
                    if self.scaler_data is not None:
                        data_dict = self.data_dict[k][v]/self.scaler_data
                    else:
                        data_dict = self.data_dict[k][v]
                    im = ax.imshow(data_dict,
                                   cmap=grey_use)
                else:
                    maxz = np.max(self.data_dict[k][v])
                    im = ax.imshow(self.data_dict[k][v],
                                   norm=LogNorm(vmin=low_lim*maxz,
                                                vmax=maxz),
                                   cmap=grey_use)
                ax.set_title('{}'.format(k+'_'+v), fontsize=fontsize)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                self.fig.colorbar(im, cax=cax)

        elif len(stat_temp) <= 6 and len(stat_temp) > 4:
            for i, (k, v) in enumerate(sorted(stat_temp)):
                ax = self.fig.add_subplot(eval('23'+str(i+1)))
                if self.scale_opt == 'Linear':
                    if self.scaler_data is not None:
                        data_dict = self.data_dict[k][v]/self.scaler_data
                    else:
                        data_dict = self.data_dict[k][v]
                    im = ax.imshow(data_dict,
                                   cmap=grey_use)
                else:
                    maxz = np.max(self.data_dict[k][v])
                    im = ax.imshow(self.data_dict[k][v],
                                   norm=LogNorm(vmin=low_lim*maxz,
                                                vmax=maxz),
                                   cmap=grey_use)
                ax.set_title('{}'.format(k+'_'+v), fontsize=fontsize)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                self.fig.colorbar(im, cax=cax)

        elif len(stat_temp) > 6 and len(stat_temp) <= 9:
            for i, (k, v) in enumerate(sorted(stat_temp)):
                ax = self.fig.add_subplot(eval('33'+str(i+1)))
                if self.scale_opt == 'Linear':
                    if self.scaler_data is not None:
                        data_dict = self.data_dict[k][v]/self.scaler_data
                    else:
                        data_dict = self.data_dict[k][v]
                    im = ax.imshow(data_dict,
                                   cmap=grey_use)
                else:
                    maxz = np.max(self.data_dict[k][v])
                    im = ax.imshow(self.data_dict[k][v],
                                   norm=LogNorm(vmin=low_lim*maxz,
                                                vmax=maxz),
                                   cmap=grey_use)
                ax.set_title('{}'.format(k+'_'+v), fontsize=fontsize)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                self.fig.colorbar(im, cax=cax)

        elif len(stat_temp) > 9 and len(stat_temp) <= 12:
            nrow = 3
            ncol = 4
            self.fig = _img_helper(stat_temp, self.data_dict,
                                   nrow, ncol,
                                   self.scale_opt, fontsize,
                                   scaler_data=self.scaler_data,
                                   color_opt=grey_use,
                                   low_lim=1e-4)

        elif len(stat_temp) > 12 and len(stat_temp) <= 16:
            nrow = 4
            ncol = 4
            self.fig = _img_helper(stat_temp, self.data_dict,
                                   nrow, ncol,
                                   self.scale_opt, fontsize,
                                   scaler_data=self.scaler_data,
                                   color_opt=grey_use,
                                   low_lim=1e-4)
        self.update_plot()

    def get_activated_num(self):
        data_temp = []
        for k, v in six.iteritems(self.stat_dict):
            for m in six.iterkeys(v):
                if v[m]:
                    data_temp.append((k, m))
        return data_temp


def _img_helper(stat_temp, data_dict, nrow, ncol,
                scale_opt, fontsize,
                scaler_data=None, color_opt=None,
                low_lim=1e-4):
    """
    Draw nrow by ncol 2D images.

    Parameters
    ----------
    stat_temp : dict
        the status of each plot
    data_dict : dict
        all 2D images
    nrow : int
        number of row in 2D plots
    ncol : int
        number of column in 2D plots
    scale_opt : str
        linear or other
    fontsize : int
        size of font in plot
    scaler_data : array, optional
        data used for normalization
    color_opt : str, optional
        color or grey
    low_lim : float, optional
        define low limit in log plot

    Returns
    fig :
        plt.fig
    """
    fig, ax_all = plt.subplots(nrows=nrow, ncols=ncol)

    for i, (k, v) in enumerate(sorted(stat_temp)):
        if len(stat_temp) == 1:
            ax = ax_all
        else:
            m = i / ncol
            n = i % ncol
            ax = ax_all[m][n]

        data = data_dict[k][v]
        if scale_opt == 'Linear':
            if scaler_data is not None:
                data = data_dict[k][v]/scaler_data
            im = ax.imshow(data,
                           cmap=color_opt)
        else:
            maxz = np.max(data)
            im = ax.imshow(data,
                           norm=LogNorm(vmin=low_lim*maxz,
                                        vmax=maxz),
                           cmap=color_opt)
        ax.set_title('{}'.format(k+'_'+v), fontsize=fontsize)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

    for m in range(nrow):
        for n in range(ncol):
            if m*ncol+n >= len(stat_temp):
                ax_all[m, n].set_visible(False)

    return fig
