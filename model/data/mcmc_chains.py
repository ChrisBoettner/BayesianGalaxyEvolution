#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 19:20:50 2023

@author: chris
"""
import numpy as np
import pandas as pd

import emcee
from pathlib import Path
import copy

from model.helper import make_array

def process_mcmc_chains(ModelResult, redshift=None,
                        autocorr_discard=10, num=int(1e+4)):
    '''
    Produce processed mcmc chains by discarding burn-in and selecting a
    random selection of num samples. If no redshift is given, iterate through
    all redshift in ModelResult.
    '''
    
    if redshift:
        redshift = make_array(redshift)
    else:
        redshift = ModelResult.redshift
    
    for z in redshift:
        # load raw chains
        raw_chain_path = ModelResult.directory
        file = raw_chain_path + ModelResult.filename.at_z(z) + '.h5'
        if not Path(file).is_file():
            raise FileNotFoundError('mcmc data file does not exist.')
        savefile = emcee.backends.HDFBackend(file)
        
        # discard burn-in
        tau = savefile.get_autocorr_time()
        chain = savefile.get_chain(
                         discard=autocorr_discard*np.amax(tau).astype(int),
                         flat=True)
        
        # select num random samples from chain
        random_draw = np.random.choice(len(chain), size=num, replace=False)
        chain = chain[random_draw]
        
        # create DataFrame using processed chain
        header = copy.copy(ModelResult.quantity_options['param_y_labels'])
        if not ModelResult.fixed_m_c:
            header.insert(0, r'$\log M_\mathrm{c}^'
                             + ModelResult.quantity_options[
                             'quantity_subscript'] + r'$')
        header = np.array(header)
        
        # if number of parameter changes with z, adjust header
        if chain.shape[1]<len(header):
            used_parameter = ModelResult.physics_model.at_z(z).parameter_used
            header = header[used_parameter] 
        
        chain = pd.DataFrame(chain, index=None, columns=header)
        
        # save to file
        name = ModelResult.filename.at_z(z) + '.csv' 
        path = raw_chain_path + '/processed_chains/'
                
        
        Path(path).mkdir(parents=True, exist_ok=True)
        chain.to_csv(path + '/' + name, index=None)
    return