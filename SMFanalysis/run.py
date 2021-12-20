#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 12:49:53 2021

@author: boettner
"""

import sys

from smf_modelling   import model_container
from data_processing import load_data

# coose option
fitting_method = 'mcmc'     # 'least_squares' or 'mcmc'   
prior_model    = 'full'     # 'uniform', 'marginal' or 'full'
mode           = 'saving'   # 'saving', 'loading' or 'temp'

# load data
groups, smfs, hmfs = load_data()

# define model
feedback_model = {1: 'none',
                  2: 'sn',
                  3: 'both',            
                  4: ['both']*4+['sn']*6
                 }

prior_mode = {1: 'uniform',
              2: 'marginal',
              3: 'full'
             }


# choose model from external input
f_choose = int(sys.argv[1])
p_choose = int(sys.argv[2])

model_container(smfs, hmfs, feedback_model[f_choose], fitting_method,
                prior_mode[p_choose], mode)