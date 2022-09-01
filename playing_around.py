#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:41:12 2022

@author: chris
"""

from model.interface import load_model, run_model, save_model
from model.plotting.plotting import *

mstar  = load_model('mstar','changing')
# muv    = load_model('Muv','changing')
mbh    = load_model('mbh','quasar', prior_name='successive')
# lbol   = load_model('Lbol', 'eddington', prior_name='successive')

#mstar = run_model('mstar', 'changing', redshift=0, fitting_method='mcmc', 
#                  num_walker=10, min_chain_length=5000)
#mbh = run_model('mbh', 'quasar', redshift=0, fitting_method='mcmc', 
#                num_walker=10, min_chain_length=5000)