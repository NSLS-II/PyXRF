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
import copy
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from atom.api import Atom, Str, observe, Typed, Int, List, Dict

class DrawImage(Atom):

    img_data = Typed(object)
    fig = Typed(Figure)
    img_num = Int()
    file_name = Str()
    stat_dict = Dict()
    data_dict = Dict()
    file_opt = Int(0)
    plot_opt = Int(0)
    single_file = Dict()

    def __init__(self):
        self.fig = plt.figure()
        #plt.tight_layout()

    @observe('file_opt')
    def update_file(self, change):
        #if change['type'] == 'update':
        if self.file_opt == 0:
            return
        self.file_name = sorted(self.data_dict.keys())[self.file_opt-1]
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
            #print('img data: {}'.format(self.img_data.keys()))

    def plot_status(self):
        for k, v in six.iteritems(self.data_dict.values()[0]['roi_sum']):
            self.stat_dict.update({k: False})

    #@observe('stat_dict')
    def show_image(self):
        print(dir(self.fig))
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

        self.fig.tight_layout(pad=0.1)#, w_pad=0.1, h_pad=0.1)
        self.fig.canvas.draw()

    def get_activated_num(self):
        data_temp = {}
        for k, v in six.iteritems(self.stat_dict):
            if v:
                data_temp.update({k: v})
        return data_temp