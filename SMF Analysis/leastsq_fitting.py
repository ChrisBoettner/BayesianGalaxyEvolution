#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 15:55:20 2021

@author: boettner
"""
import numpy as np
from scipy.optimize import minimize

## MAIN FUNCTION
def lsq_fit(smf, hmf, smf_model, z = 0):
    '''
    Calculate parameter that match observed SMFs to modelled SMFs using least
    squares regression and a pre-defined cost function (which is not the usual
    one! See cost_function for details). 
    '''
                                         
    # fit smf model to data based on pre-defined cost function
    fitting       = minimize(cost_function, smf_model.feedback_model.initial_guess,
                                  args = (smf, smf_model))
    par           = fitting.x
    cost          = cost_function(par, smf, smf_model)
    
    # create data for modelled smf (for plotting)
    modelled_smf = hmf.copy()
    modelled_smf[:,1] = smf_model.function(hmf[:,0], par)
    return(par, modelled_smf, cost)

## LEAST SQUARE HELP FUNCTIONS
def cost_function(params, smf, smf_model):
    '''
    Cost function for fitting. Includes physically sensible bounds for parameter.
    IMPORTANT :   We minimize the log of the phi_obs and phi_mod, instead of the
                  values themselves. Otherwise the low-mass end would have much higher
                  constribution due to the larger number density.
    '''      
    m_obs   = smf[:,0]
    phi_obs = smf[:,1]
    
    phi_mod = smf_model.function(m_obs, params)
    
    if not within_bounds(params, [0,0,0], [1,10,10]):
        return(1e+10) # return inf (or huge value) if outside of bounds
    
    cost = np.linalg.norm(np.log10(phi_obs) - np.log10(phi_mod))**2   
    return(cost/len(smf)) # otherwise return normalised cost
    

## HELP FUNCTIONS
def within_bounds(values, lower_bounds, upper_bounds):
    '''
    Checks if list of values is within lower and upper bounds (if value is equal
    to bound, also return true). All three arrays must have same length.
    '''
    is_within = []
    for i in range(len(values)):
        is_within.append((values[i] >= lower_bounds[i]) & (values[i] <= upper_bounds[i]))
        
    # return True if all elements are within bounds, otherwise False
    if all(is_within)==True:
        return(True)
    return(False)

        


