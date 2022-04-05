#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 11:39:07 2022

@author: chris
"""

import numpy as np
from scipy.interpolate import interp1d

import emcee

import os

def load_hmf_functions():
    '''
    Load HMF data and transform it to callable function by interpolating
    the data.
    '''
    hmfs = np.load('data/HMF.npz')
    hmf_functions = []
    #import pdb; pdb.set_trace()
    for i in range(20):
        h          = np.power(10,hmfs[str(i)])
        lower_fill = h[0,1]; upper_fill = h[-1,1]
        hmf_functions.append(interp1d(*h.T, bounds_error = False, 
                                      fill_value = (lower_fill, upper_fill)))
    return(hmf_functions)

def load_mcmc_data(quantity_name):
    '''
    Load mcmc data for SMF and UVLF fit (changing feedback).
    '''
    # relate physical quantites to save locations of files
    load_dict = {'lum'   : 'UVLF',
                 'mstar' : 'SMF'}
    
    # use correct file path depending on system
    save_path = '/data/p305250/' + load_dict[quantity_name] + '/mcmc_runs/changing/'
    if os.path.isdir(save_path): # if path exists use this one (cluster structure)
        pass 
    else: # else use path for home computer
        save_path = '/home/chris/Desktop/mcmc_runs/' + load_dict[quantity_name] + '/changing/'  

    # pre-calculated best fit parameter
    parameter = np.load(save_path + 'changing' + '_parameter_' + 'full' + '.npy', allow_pickle=True)
    
    distribution = []
    for z in range(len(parameter)):
        file = save_path + 'changing' + str(z) +'full.h5'
        #import pdb; pdb.set_trace()
        savefile = emcee.backends.HDFBackend(file)
    
        sampler = savefile
        
        # get autocorrelationtime and discard burn-in of mcmc walk 
        tau       = np.array(sampler.get_autocorr_time())
        posterior = sampler.get_chain(discard=5*np.amax(tau).astype(int), flat=True)
        distribution.append(posterior)
    
    return(parameter, distribution)

def load_best_fit_parameter(quantity_name, prior_model = 'full'):
    '''
    Load pre-calculated best fit parameter for model (changing feedback).
    '''
    # relate physical quantites to save locations of files
    load_dict = {'lum'   : 'UVLF',
                 'mstar' : 'SMF'}
    
    # use correct file path depending on system
    save_path = '/data/p305250/' + load_dict[quantity_name] + '/mcmc_runs/changing/'
    if os.path.isdir(save_path): # if path exists use this one (cluster structure)
        pass 
    else: # else use path for home computer
        save_path = '/home/chris/Desktop/mcmc_runs/' + load_dict[quantity_name] + '/changing/'  

    # pre-calculated best fit parameter
    parameter = np.load(save_path + 'changing' + '_parameter_' + prior_model + '.npy', 
                        allow_pickle=True)

    # correct for the change in variables done in the fit originally
    if quantity_name == 'lum':
        for p in parameter:
            p[0] = p[0]*1e+18
    
    return(parameter)
    