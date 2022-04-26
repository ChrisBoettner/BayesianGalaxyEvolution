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

m = save_model(quantity_name, 'changing',
              num_walker=50, min_chain_length=0)

# m = run_model(quantity_name, 'stellar_blackhole', fitting_method='mcmc',
#               num_walker=50)

# m = [run_model(quantity_name, 'none'),
#     run_model(quantity_name, 'stellar'),
#     run_model(quantity_name, 'stellar_blackhole')]

#from model.analysis.schechter import tabulate_schechter_parameter

#o = tabulate_schechter_parameter(m,m.redshift)

Plot_best_fit_ndfs(m)
