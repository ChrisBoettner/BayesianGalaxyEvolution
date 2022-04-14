#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 19:27:34 2022

@author: chris
"""

from model.api import run_model, load_model

quantity_name = 'Muv'
#feedback_name = 'none'

m = run_model(quantity_name, 'changing', fitting_method = 'least_squares')

#m = [run_model(quantity_name, 'none'),
#     run_model(quantity_name, 'stellar'),
#     run_model(quantity_name, 'stellar_blackhole')]

from model.plotting.plotting import *

o = Plot_ndfs(m)