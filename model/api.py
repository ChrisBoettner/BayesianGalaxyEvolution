#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 11:51:46 2022

@author: chris
"""
import timeit

from model.data.load import load_data, load_hmf_functions
from model.calibration.calibration import CalibrationResult, save_parameter

def load_model(quantity_name, feedback_name, data_subset = None,
               prior_name = None, **kwargs):
    '''
    Load saved (MCMC) model. Built for simplicity so that feedback_name is 
    associated with specific prior, but can be changed if needed.
    Loads parameter from file.
    Can choose to load and run on subset of data, just put in list of data set
    names (of form AuthorYear).   
    '''

    groups, log_ndfs = load_data(quantity_name, data_subset)
    log_hmfs         = load_hmf_functions()
    redshifts        = list(log_ndfs.keys())
    
    if prior_name is None:
        if feedback_name == 'changing':
            prior_name   =  'successive'
        else:
            prior_name   =  'uniform'
    
    model = CalibrationResult(redshifts, log_ndfs, log_hmfs, 
                              quantity_name, feedback_name, prior_name,
                              groups         = groups,
                              name_addon     = data_subset,
                              fitting_method = 'mcmc',
                              saving_mode    = 'loading',
                              **kwargs)
    return(model)
    
def save_model(quantity_name, feedback_name, data_subset = None,
               prior_name = None, **kwargs):
    '''
    Run and save (MCMC) model. Built for simplicity so that feedback_name is 
    associated with specific prior, but can be changed if needed.
    Also saves parameter to same folder.
    Can choose to load and run on subset of data, just put in list of data set
    names (of form AuthorYear).
    '''
    
    groups, log_ndfs = load_data(quantity_name, data_subset)
    log_hmfs         = load_hmf_functions()
    redshifts        = list(log_ndfs.keys())
    
    if prior_name is None:
        if feedback_name == 'changing':
            prior_name   =  'successive'
        else:
            prior_name   =  'uniform'
    
    print(quantity_name, prior_name, feedback_name)
    start = timeit.default_timer()
    model = CalibrationResult(redshifts, log_ndfs, log_hmfs, 
                              quantity_name, feedback_name, prior_name,
                              groups         = groups,
                              name_addon     = data_subset,
                              fitting_method = 'mcmc',
                              saving_mode    = 'saving',
                              parameter_calc = True,
                              **kwargs)
    save_parameter(model, data_subset)
    end  = timeit.default_timer()
    print('DONE')
    print(str((end-start)/3600) + 'hours')
    return(model)
    
def run_model(quantity_name, feedback_name, fitting_method = 'least_squares',
              chain_length = 10000, num_walkers = 10, autocorr_discard = False,
              data_subset = None, prior_name = None, **kwargs):
    '''
    Run a model calibration without saving. Default is least_squares fit (without
    mcmc), but can be changed. If fitting_method is mcmc, uses low chain_length and
    num_walker so that it finishes quickly (autocorr_discard disabled by
    default, also adjustable).
    Can choose to load and run on subset of data, just put in list of data set
    names (of form AuthorYear).
    '''
    
    groups, log_ndfs = load_data(quantity_name, data_subset)
    log_hmfs         = load_hmf_functions()
    redshifts        = list(log_ndfs.keys())
    
    if prior_name is None:
        if feedback_name == 'changing':
            prior_name   =  'successive'
        else:
            prior_name   =  'uniform'
            
    model = CalibrationResult(redshifts, log_ndfs, log_hmfs, 
                              quantity_name, feedback_name, prior_name,
                              groups         = groups,
                              fitting_method    = fitting_method,
                              name_addon        = data_subset,
                              chain_length      = chain_length,
                              num_walkers       = num_walkers,
                              autocorr_discard  = False,
                              saving_mode       = 'temp',
                              **kwargs)
    return(model)