__author__ = 'Li Li'

import numpy as np
import time

from atom.api import Atom, Str, observe, Typed, Int, List, Dict

from skxray.fitting.xrf_model import (k_line, l_line, m_line)
from skxray.constants.api import XrfElement as Element

#from skxray.fitting.physics_peak import (gauss_tail, gauss_peak)
#from skxray.fitting.models import (ElasticModel, ComptonModel, GaussModel)
from skxray.fitting.xrf_model import (ModelSpectrum, update_parameter_dict,
                                      get_sum_area, set_parameter_bound,
                                      ElementController, set_range, get_linear_model,
                                      PreFitAnalysis, k_line, l_line)
from skxray.fitting.background import snip_method


class Fit1D(Atom):

    param_dict = Dict()
    data = Typed(np.ndarray)
    fit_x = Typed(np.ndarray)
    fit_y = Typed(np.ndarray)
    comps = Typed(object)

    def __init__(self):
        pass

    def fit_data(self):

        c_val = 1e-2
        self.data = np.asarray(self.data)
        x = np.arange(self.data.size)
        x0, y0 = set_range(self.param_dict, x, self.data)

        # get background
        bg = snip_method(y0,
                         self.param_dict['e_offset']['value'],
                         self.param_dict['e_linear']['value'],
                         self.param_dict['e_quadratic']['value'])

        MS1 = ModelSpectrum(self.param_dict)
        MS1.model_spectrum()
        p1 = MS1.mod.make_params()

        #y_init = MS1.mod.eval(x=x0, params=p1)

        print('Start fitting!')
        t0 = time.time()
        result1 = MS1.model_fit(x0, y0-bg, w=1/np.sqrt(y0), maxfev=100,
                                xtol=c_val, ftol=c_val, gtol=c_val)
        t1 = time.time()
        print('time used: {}'.format(t1-t0))

        fitname = list(result1.values.keys())
        namelist = [str(v.prefix) for v in result1.model.components]

        #print('name list: {}'.format(namelist))

        #print('fit: {}'.format(result1.values))

        #print('dir1: {}'.format(MS1.mod.components))

        #comp = result1.model.components['compton']
        #print('dir2: {}'.format(dir(result1.model)))

        self.comps = result1.eval_components(x=x0)
        #print('component: {}'.format(self.comps))

        xnew = result1.values['e_offset'] + result1.values['e_linear'] * x0 +\
               result1.values['e_quadratic'] * x0**2
        self.fit_x = xnew
        #self.fit_y = result1.model.components[0].eval(x=xnew, params=result1.params)
        self.fit_y = result1.best_fit + bg

        #print(result1.fit_report())
