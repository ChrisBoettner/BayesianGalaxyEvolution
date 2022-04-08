#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 13:48:39 2022.

@author: chris
"""

import sys
import timeit

from model.data.load import load_data, load_hmf_functions
from model.calibration.calibration import CalibrationResult

fitting_method = 'mcmc' 
saving_mode    = 'saving'

# choose option
quantity_name  = sys.argv[1]
prior_name     = sys.argv[2]
feedback_name  = sys.argv[3]

# load data
groups, log_ndfs = load_data(quantity_name)
log_hmfs         = load_hmf_functions()
redshifts        = list(log_ndfs.keys())

# choose model from external input

print(quantity_name, prior_name, feedback_name)

start = timeit.default_timer()

CalibrationResult(redshifts, log_ndfs, log_hmfs, 
                  quantity_name, feedback_name, prior_name,
                  fitting_method, saving_mode,
                  chain_length = 200000, num_walkers = 25,
                  autocorr_discard = True, progress=False,
                  parameter_calc=False)
                
end  = timeit.default_timer()
print('DONE')
print(end-start)