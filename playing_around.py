#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:41:12 2022

@author: chris
"""

from model.interface import load_model, run_model, save_model
from model.plotting.plotting import *

#o = run_model('Lbol', 'quasar')

#print('NOOOO. THINK ABOUT FITTING IN LOG SPACE FOR ALL/SOME QUANTITIES, ALSO WEIGHTS')
#print('BUT MORE IMPORTANT FOR Lbol: UPPER LIMIT FOR A SHOULD BE LOWEST MEASURED VALUE AT EVERY z INDIVIDUALLY, NOT ALL FOR z=0')

o = run_model('Lbol', 'quasar', fitting_method='mcmc', num_walker=10, min_chain_length=0, parallel = True,
              redshift = [0,1,2], prior_name='successive')

#m = run_model('mstar','changing', fitting_method='mcmc',num_walker=10, autocorr_chain_multiple=1,
#              redshift=[4,5], tolerance = 0.01, parallel=True, min_chain_length=20000, parameter_calc=False,
#              autocorr_discard=True)
#save_model('mstar', 'changing', num_walker=50)

#mstar = load_model('mstar','changing')
#muv   = load_model('Muv','changing')
#lbol  = load_model('Lbol','quasar',prior_name='successive')

#Plot_best_fit_ndfs(muv)
