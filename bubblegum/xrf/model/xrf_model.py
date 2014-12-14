from pprint import pprint
from atom.api import Atom, Str, observe, Typed, Int
import numpy as np
import os
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from skxray.fitting.xrf_model import (ModelSpectrum, set_range, k_line,
                                      l_line, m_line, get_linear_model, PreFitAnalysis)
from skxray.fitting.background import snip_method
from skxray.constants.api import XrfElement as Element


class FileIOModel(Atom):
    tool_name = Str('PyXRF: X-ray Fluorescence Analysis Tool')
    folder_name = Str('')
    file_name = Str('')
    data = Typed(object)
    file_path = Str()
    load_status = Str()
    #tool_name = 'PyXRF'

    @observe('folder_name', 'file_name')
    def update(self, changed):
        pprint(changed)
        if changed['type'] == 'create':
            return
        print('{} was changed from {} to {}'.format(changed['name'],
                                                    changed['oldvalue'],
                                                    changed['value']))
        #if changed['name'] == 'file_name':
        #    self.load_data()

    @observe('data')
    def data_changed(self, data):
        print('The data was changed. First five lines of new data:\n{}'
              ''.format(self.data[:5]))

    def set_path(self):
        self.file_path = os.path.join(self.folder_name, self.file_name)
        if os.path.exists(self.file_path):
            self.load_status = 'File {0} is loaded successfully.'.format(self.file_name)
            self.data = np.loadtxt(self.file_path)
        else:
            self.load_status = 'File {0} does not exist.'.format(self.file_name)


class LinePlotModel(Atom):
    data = Typed(object)
    _fig = Typed(Figure)
    _ax = Typed(Axes)
    _canvas = Typed(object)
    element_id = Typed(object)
    incident_energy = Typed(object)
    elist = Typed(object)
    plot_opt = Int(0)

    def __init__(self, data=None):
            super(LinePlotModel, self).__init__()
            # mpl setup
            #self._fig = plt.figure()
            self._fig = Figure()
            self.elist = []
            #self._ax = self._fig.add_subplot(111)

    @observe('data')
    def _new_data(self, change):
        print('data changed')
        data_arr = np.asarray(self.data)
        #self._ax.imshow(data_arr)
        print 'data is: ', data_arr
        pprint(change)
        #x_v = np.arange(len(data_arr))*0.01
        #self._ax.plot(x_v, data_arr)

    def set_data(self, data):
        self.data = data

    @observe('plot_opt')
    def _new_opt(self, change):
        if change['type'] == 'update':
            self.plot_raw()

    def plot_raw(self):
        plot_type = ['LinLog', 'Linear']
        self._ax = self._fig.add_subplot(111)
        self._ax.hold(False)
        data_arr = np.asarray(self.data)
        #self._ax.imshow(data_arr)
        print 'data is: ', data_arr
        x_v = np.arange(len(data_arr))*0.01

        if plot_type[self.plot_opt] == 'Linear':
            self._ax.plot(x_v, data_arr)
        else:
            self._ax.semilogy(x_v, data_arr)

        minv = np.min(data_arr)
        if len(self.elist) != 0:
            self._ax.hold(True)
            for i in range(len(self.elist)):
                if plot_type[self.plot_opt] == 'Linear':
                    self._ax.plot(x_v, data_arr)
                    self._ax.plot([self.elist[i][0], self.elist[i][0]],
                                  [minv, self.elist[i][1]*np.max(data_arr)],
                                  'r-', linewidth=2.0)
                else:
                    self._ax.semilogy([self.elist[i][0], self.elist[i][0]],
                                      [minv, self.elist[i][1]*np.max(data_arr)],
                                      'r-', linewidth=2.0)

        self._fig.canvas.draw()

    @observe('element_id')
    def set_element(self, change):
        if change['value'] == 0:
            return

        self.elist = []
        pprint(change)

        total_list = k_line + l_line + m_line
        pprint('element name: {}'.format(self.element_id))
        ename = total_list[self.element_id-1]
        self.incident_energy = 11.0

        if len(ename) <= 2:
            e = Element(ename)
            if e.cs(self.incident_energy)['ka1'] != 0:
                for i in range(4):
                    self.elist.append((e.emission_line.all[i][1],
                                       e.cs(self.incident_energy).all[i][1]/e.cs(self.incident_energy).all[0][1]))

        elif '_L' in ename:
            e = Element(ename[:-2])
            print e.cs(self.incident_energy)['la1']
            if e.cs(self.incident_energy)['la1'] != 0:
                for i in range(4, 17):
                    self.elist.append((e.emission_line.all[i][1],
                                       e.cs(self.incident_energy).all[i][1]/e.cs(self.incident_energy).all[4][1]))

        else:
            e = Element(ename[:-2])
            if e.cs(self.incident_energy)['ma1'] != 0:
                for i in range(17, 21):
                    self.elist.append((e.emission_line.all[i][1],
                                       e.cs(self.incident_energy).all[i][1]/e.cs(self.incident_energy).all[17][1]))

        self.plot_raw()
