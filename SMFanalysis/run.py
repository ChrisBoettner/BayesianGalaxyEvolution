#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 12:49:53 2021

@author: boettner
"""

import sys
import timeit

from smf_modelling   import model_container
from data_processing import load_data

# coose option
fitting_method = 'mcmc'     # 'least_squares' or 'mcmc'
mode           = 'saving'   # 'saving', 'loading' or 'temp'

# load data
groups, smfs, hmfs = load_data()

# define model
feedback_model = {1: 'none',
                  2: 'sn',
                  3: 'both',            
                  4: ['both']*5+['sn']*6
                 }

prior_mode = {1: 'uniform',
              2: 'marginal',
              3: 'full'
             }


# choose model from external input
f_choose = int(sys.argv[1])
p_choose = int(sys.argv[2])

print(feedback_model[f_choose])
print(prior_mode[p_choose])

start = timeit.default_timer()
model_container(smfs, hmfs, feedback_model[f_choose], fitting_method,
                prior_mode[p_choose], mode)
                
end  = timeit.default_timer()
print('DONE')
print(end-start)
