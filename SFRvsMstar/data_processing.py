#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 11:39:07 2022

@author: chris
"""

import numpy as np
import emcee

import os

def load_mcmc_data(quantity_name):
    '''
    Load mcmc data for SMF and UVLF fit.
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

    # pre-calculated geometric medians of distributions
    parameter = np.load('data/' + load_dict[quantity_name] + '_changing_medians.npy', allow_pickle=True) 
    
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