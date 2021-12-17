#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 15:55:20 2021

@author: boettner
"""
import numpy as np
from scipy.optimize import least_squares

## MAIN FUNCTION
def lsq_fit(smf_model):
    '''
    Calculate parameter that match observed SMFs to modelled SMFs using least
    squares regression and a pre-defined cost function (which is not the usual
    one! See cost_function for details). 
    '''                    
    # fit smf model to data based on pre-defined cost function
    fitting       = least_squares(cost_function, smf_model.feedback_model.initial_guess,
                                  args = (smf_model,))
    par           = fitting.x
    
    par_distribution = None # for compatibility with mcmc fit
    return(par, par_distribution)

## LEAST SQUARE HELP FUNCTIONS
def cost_function(params, smf_model):
    '''
    Cost function for fitting. Includes physically sensible bounds for parameter.
    IMPORTANT :   We minimize the log of the phi_obs and phi_mod, instead of the
                  values themselves. Otherwise the low-mass end would have much higher
                  constribution due to the larger number density.
    '''      
    m_obs   = smf_model.observations[:,0]
    phi_obs = smf_model.observations[:,1]
    
    # check if parameter are within bounds
    if not within_bounds(params, *smf_model.feedback_model.bounds):
        return(np.inf)
    
    phi_mod = smf_model.function(m_obs, params)
    
    
    res = np.log10(phi_obs) - np.log10(phi_mod)
    return(res) # otherwise return cost
    

## HELP FUNCTIONS
def within_bounds(values, lower_bounds, upper_bounds):
    '''
    Checks if list of values is within lower and upper bounds (strictly
    within bounds). All three arrays must have same length.
    '''
    is_within = []
    for i in range(len(values)):
        is_within.append((values[i] > lower_bounds[i]) & (values[i] < upper_bounds[i]))
        
    # return True if all elements are within bounds, otherwise False
    if all(is_within)==True:
        return(True)
    return(False)


