#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 19:27:34 2022

@author: chris
"""

from model.plotting.plotting import *
from model.api import run_model, load_model

quantity_name = 'mstar'
#feedback_name = 'none'
#m = load_model(quantity_name, 'stellar_blackhole')

m = run_model(quantity_name, 'changing', fitting_method = 'mcmc', chain_length = 50,
              parameter_calc=True)

# m = run_model(quantity_name, 'changing', fitting_method = 'mcmc',
#               chain_length = 5000, num_walkers = 150, data_subset='Bouwens2021',
#               autocorr_discard = True, parameter_calc = False, progress = True,
#               redshifts = 2)

# m = [run_model(quantity_name, 'none'),
#     run_model(quantity_name, 'stellar'),
#     run_model(quantity_name, 'stellar_blackhole')]


#o = Plot_schechter_sample(m)
