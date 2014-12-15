from pprint import pprint
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
#from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from atom.api import Atom, Str, observe, Typed, Int

from skxray.fitting.xrf_model import (k_line, l_line, m_line)
from skxray.constants.api import XrfElement as Element


class LinePlotModel(Atom):
    data = Typed(object)
    _fig = Typed(Figure)
    _ax = Typed(Axes)
    _canvas = Typed(object)
    element_id = Typed(object)
    incident_energy = Typed(object)
    elist = Typed(object)
    plot_opt = Int(0)
    total_y = Typed(object)
    total_y_l = Typed(object)
    prefit_bg = Typed(object)
    prefit_x = Typed(object)

    def __init__(self, data=None):
        super(LinePlotModel, self).__init__()
        # mpl setup
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111)
        #self._fig = Figure()
        self.elist = []
        self.total_y = []
        self.total_y_l = []
        self.prefit_bg = []

    def set_data(self, data):
        self.data = data

    @observe('plot_opt')
    def _new_opt(self, change):
        if change['type'] == 'update':
            self.plot_data()

    def plot_data(self):
        plot_type = ['LinLog', 'Linear']
        #self._ax = self._fig.add_subplot(111)
        self._ax.hold(False)
        data_arr = np.asarray(self.data)
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
                    self._ax.plot([self.elist[i][0], self.elist[i][0]],
                                  [minv, self.elist[i][1]*np.max(data_arr)],
                                  'r-', linewidth=2.0)
                else:
                    self._ax.semilogy([self.elist[i][0], self.elist[i][0]],
                                      [minv, self.elist[i][1]*np.max(data_arr)],
                                      'r-', linewidth=2.0)

        if len(self.total_y) != 0:
            self._ax.hold(True)
            if plot_type[self.plot_opt] == 'Linear':
                if len(self.prefit_bg) != 0:
                    self._ax.plot(self.prefit_x, self.prefit_bg, 'grey')
                    self._ax.plot(self.prefit_x, np.sum(self.total_y, axis=1) +
                                  np.sum(self.total_y_l, axis=1)+self.prefit_bg, 'b-', label='prefit')
                else:
                    self._ax.plot(self.prefit_x, np.sum(self.total_y, axis=1) +
                                  np.sum(self.total_y_l, axis=1), 'b-', label='prefit')

                self._ax.plot(self.prefit_x, self.total_y, 'g-')
                self._ax.plot(self.prefit_x, self.total_y_l, 'purple')
            else:
                if len(self.prefit_bg) != 0:
                    self._ax.semilogy(self.prefit_x, self.prefit_bg, 'grey')
                    self._ax.semilogy(self.prefit_x, np.sum(self.total_y, axis=1) +
                                      np.sum(self.total_y_l, axis=1)+self.prefit_bg, 'b-', label='prefit')
                else:
                    self._ax.semilogy(self.prefit_x, np.sum(self.total_y, axis=1) +
                                      np.sum(self.total_y_l, axis=1), 'b-', label='prefit')

                self._ax.semilogy(self.prefit_x, self.total_y, 'g-')
                self._ax.semilogy(self.prefit_x, self.total_y_l, 'purple')
            self._ax.set_xlim([self.prefit_x[0], self.prefit_x[-1]])

        self._ax.set_ylim([minv, np.max(data_arr)*2.0])
        self._ax.set_xlabel('Energy [keV]')
        self._ax.set_ylabel('Counts')

        self._fig.canvas.draw()

    @observe('element_id')
    def set_element(self, change):
        if change['value'] == 0:
            self.elist = []
            self.plot_data()
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

        self.plot_data()

    def set_prefit_data(self, prefit_x, total_y, total_y_l):
        self.prefit_x = prefit_x
        # k lines
        self.total_y = total_y
        # l lines
        self.total_y_l = total_y_l

    def set_prefit_bg(self, prefit_bg):
        """
        set background from prefit plot
        """
        self.prefit_bg = prefit_bg
