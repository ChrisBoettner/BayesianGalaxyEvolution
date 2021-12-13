#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 10:43:45 2021

@author: boettner
"""

from matplotlib import rc_file
rc_file('plots/settings.rc')  # <-- the file containing your settings

import numpy as np
import matplotlib.pyplot as plt

from smf_modelling   import model_container
from data_processing import load_data

import astropy.units as u
from astropy.cosmology import Planck18, z_at_value

# coose option
fitting_method = 'mcmc'    # 'least_squares' or 'mcmc'   
prior_model    = 'full'    # 'uniform', 'marginal' or 'full'
mode           = 'saving'  # 'saving', 'loading' or 'temp'

# load data
groups, smfs, hmfs = load_data()

# create model smfs
model_container(smfs, hmfs, 'both', fitting_method, 'full', mode)
prior_models = ['uniform', 'marginal', 'full']
feedbacks = ['both']*4+['sn']*6
for p in prior_models:
    model_container(smfs, hmfs, feedbacks, fitting_method, p, mode)