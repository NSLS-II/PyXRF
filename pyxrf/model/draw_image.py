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


class DrawImage(Atom):
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
    #img_num = Int()
    file_name = Str()
    stat_dict = Dict()
    data_dict = Dict()
    file_opt = Int(0)
    plot_opt = Int(0)
    single_file = Dict()

    def __init__(self):
        self.fig = plt.figure()
        #plt.tight_layout()

    def plot_status(self):
        for k, v in six.iteritems(self.data_dict.values()[0]['roi_sum']):
            self.stat_dict.update({k: False})

    @observe('file_opt')
    def update_file(self, change):
        if self.file_opt == 0:
            return
        self.file_name = sorted(self.data_dict.keys())[self.file_opt-1]
        self.plot_status()
        self.single_file = self.data_dict[self.file_name]
        self.show_image()

    @observe('plot_opt')
    def update_calculation_type(self, change):
        """
        Plot roi sum or fitted.
        """
        if self.plot_opt == 0:
            return
        if self.plot_opt == 1:
            self.img_data = self.single_file['roi_sum']
            self.show_image()

    #@observe('stat_dict')
    def show_image(self):
        if self.plot_opt == 1:
            self.img_data = self.single_file['roi_sum']

        self.fig.clf()
        stat_temp = self.get_activated_num()
        if len(stat_temp) == 1:
            ax = self.fig.add_subplot(111)
            for k, v in six.iteritems(stat_temp):
                cax = ax.imshow(self.img_data[k])
                ax.set_title('{}'.format(k))
                self.fig.colorbar(cax)
            #self.fig.suptitle(self.file_name, fontsize=14)
        elif len(stat_temp) == 2:
            for i, (k, v) in enumerate(six.iteritems(stat_temp)):
                ax = self.fig.add_subplot(eval('12'+str(i+1)))
                cax = ax.imshow(self.img_data[k])
                self.fig.colorbar(cax)
                ax.set_title('{}'.format(self.file_name))
            #self.fig.suptitle(self.file_name, fontsize=14)
        elif len(stat_temp) <= 4 and len(stat_temp) > 2:
            for i, (k, v) in enumerate(six.iteritems(stat_temp)):
                ax = self.fig.add_subplot(eval('22'+str(i+1)))
                cax = ax.imshow(self.img_data[k])
                self.fig.colorbar(cax)
                ax.set_title('{}'.format(k))
            #self.fig.suptitle(self.file_name, fontsize=14)
        elif len(stat_temp) <= 6 and len(stat_temp) > 4:
            for i, (k, v) in enumerate(six.iteritems(stat_temp)):
                ax = self.fig.add_subplot(eval('32'+str(i+1)))
                cax = ax.imshow(self.img_data[k])
                self.fig.colorbar(cax)
                ax.set_title('{}'.format(k), fontsize=10)
            #self.fig.suptitle(self.file_name, fontsize=14)
        elif len(stat_temp) > 6: # and len(stat_temp) > 4:
            for i, (k, v) in enumerate(six.iteritems(stat_temp)):
                ax = self.fig.add_subplot(eval('33'+str(i+1)))
                cax = ax.imshow(self.img_data[k])
                self.fig.colorbar(cax)
                ax.set_title('{}'.format(k), fontsize=10)
            #self.fig.suptitle(self.file_name, fontsize=14)
            #ax2 = self.fig.add_subplot(222)
            #ax2.imshow(self.img_data[k[1]])
            #ax2.set_title('{}: {}'.format(self.file_name, k[1]))
        try:
            self.fig.tight_layout(pad=0.1)#, w_pad=0.1, h_pad=0.1)
        except ValueError:
            pass
        self.fig.canvas.draw()

    def get_activated_num(self):
        data_temp = {}
        for k, v in six.iteritems(self.stat_dict):
            if v:
                data_temp.update({k: v})
        return data_temp


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
        self.group_names = ['None'] + self.data_dict.keys()

        scaler_groups = [v for v in self.data_dict.keys() if 'scaler' in v]
        self.scaler_group_name = scaler_groups[0]
        self.scaler_items = ['None'] + self.data_dict[self.scaler_group_name].keys()
        self.scaler_data = None

    @observe('group_name')
    def _change_img_group(self, change):
        self.items_in_group = []
        self.items_in_group = self.data_dict[self.group_name].keys()

    @observe('scaler_name')
    def _get_scaler_data(self, change):
        if self.scaler_name == 'None':
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

        low_lim = 1e-4 # define the low limit for log image

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
            #self.fig.suptitle(self.file_name, fontsize=14)
        elif len(stat_temp) == 2:
            for i, (k, v) in enumerate(sorted(stat_temp)):
                ax = self.fig.add_subplot(eval('21'+str(i+1)))
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
                ax = self.fig.add_subplot(eval('32'+str(i+1)))
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

        elif len(stat_temp) > 6:
            for i, (k, v) in enumerate(sorted(stat_temp)):
                if i >= 9:
                    break
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

        self.update_plot()

    def get_activated_num(self):
        data_temp = []
        for k, v in six.iteritems(self.stat_dict):
            for m in six.iterkeys(v):
                if v[m]:
                    data_temp.append((k, m))
        return data_temp
