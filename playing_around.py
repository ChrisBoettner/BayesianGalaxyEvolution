#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:41:12 2022

@author: chris
"""

from model.interface import load_model, run_model, save_model
from model.plotting.plotting import *

#o = run_model('mbh', 'quasar')

#o = run_model('mbh', 'quasar', fitting_method='mcmc', num_walker=10, min_chain_length=0, parallel = True,
#              prior_name='successive')

#m = run_model('mstar','changing', fitting_method='mcmc',num_walker=10, autocorr_chain_multiple=1,
#              redshift=[4,5], tolerance = 0.01, parallel=True, min_chain_length=20000, parameter_calc=False,
#              autocorr_discard=True)
#save_model('mstar', 'changing', num_walker=50)

#mstar = load_model('mstar','changing')
#muv   = load_model('Muv','changing')
#lbol   = load_model('Lbol','quasar',prior_name='successive')
#mbh    = load_model('mbh','quasar',prior_name='successive')

#Plot_best_fit_ndfs(o)


# abundance matching
import numpy as np
import matplotlib.pyplot as plt
from model.analysis.calculations import calculate_best_fit_ndf

import cProfile

#lbol   = load_model('Lbol','quasar',prior_name='successive')
#mbh    = load_model('mbh','quasar',prior_name='successive')
# def run():  
#     lbol = run_model('Lbol', 'eddington', prior_name='successive')
#     return(lbol)
# cProfile.run('run()')

print('fits look better if you fix ERDF to values at z=0, so do that')
print('then you can run the calibration, but you should also finish up the rest of the model'
      '(functions + doc strings)')

print('value of C parameter are close to the 38.1 you get from eddington calculation, '
      'kinda makes sense since the m^gamma model is similar to bh model (esp at high masses)'
      'so you effectively put in the black hole mass as in the original calculation')

lbol = run_model('Lbol', 'eddington', prior_name='successive')

#Plot_best_fit_ndfs(lbol)

