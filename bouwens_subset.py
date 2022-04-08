# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import timeit

from model.data.load import load_data, load_hmf_functions
from model.calibration.calibration import CalibrationResult

fitting_method = 'mcmc' 
saving_mode    = 'saving'

# choose option
quantity_name  = 'Muv'
prior_name     = 'successive'
feedback_name  = 'changing'

# load data
data_subset = 'Bouwens2021'

groups, log_ndfs = load_data(quantity_name, data_subset)
log_hmfs         = load_hmf_functions()
redshifts        = list(log_ndfs.keys())

start = timeit.default_timer()

CalibrationResult(redshifts, log_ndfs, log_hmfs, 
                  quantity_name, feedback_name, prior_name,
                  fitting_method, saving_mode,
                  chain_length = 10000, num_walkers = 200,
                  autocorr_discard = True, progress=True,
                  name_addon = data_subset,
                  parameter_calc=False)
                
end  = timeit.default_timer()
print('DONE')
print(end-start)