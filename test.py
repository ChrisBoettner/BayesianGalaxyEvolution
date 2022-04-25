#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 19:27:34 2022

@author: chris
"""

from model.plotting.plotting import *
from model.interface import run_model, load_model, save_model

quantity_name = 'mstar'
#feedback_name = 'none'
#m = load_model(quantity_name, 'stellar_blackhole')

m = run_model(quantity_name, 'stellar_blackhole', fitting_method='mcmc', chain_length=50000,
              num_walker=50,
              parameter_calc=False)

# m = run_model(quantity_name, 'stellar_blackhole',
#               chain_length = 5000, num_walkers = 200,
#               redshifts = [0,1], autocorr_discard=False)

# m = [run_model(quantity_name, 'none'),
#     run_model(quantity_name, 'stellar'),
#     run_model(quantity_name, 'stellar_blackhole')]

#from model.analysis.schechter import tabulate_schechter_parameter

#o = tabulate_schechter_parameter(m,m.redshift)
