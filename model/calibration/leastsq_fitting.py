#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:48:41 2022

@author: chris
"""

import numpy as np
from scipy.optimize import least_squares, dual_annealing, brute, minimize

from model.helper import within_bounds

################ MAIN FUNCTIONS ###############################################
def lsq_fit(model, method = 'least_squares'):
    '''
    Calculate parameter that match observed LFs to modelled LFs using least
    squares regression and a pre-defined cost function (which is not the usual
    one! See cost_function for details). 
    '''  
    bounds = list(zip(*model.feedback_model.bounds))
    # fit lf model to data based on pre-defined cost function
    if method == 'least_squares':
        optimization_res = least_squares(cost_function, model.feedback_model.initial_guess,
                                         bounds = model.feedback_model.bounds,
                                         args = (model,'res'))
    
    if method == 'minimize':
        optimization_res = minimize(cost_function, x0 = model.feedback_model.initial_guess,
                                    bounds = bounds,
                                    args = (model,))
    
    elif method == 'annealing':
        optimization_res = dual_annealing(cost_function, 
                                          bounds = bounds,
                                          maxiter = 100,
                                          args = (model,))
    elif method == 'brute':
        optimization_res = brute(cost_function, 
                                  ranges = bounds,
                                  Ns = 100,
                                  args = (model,))      
    
    par = optimization_res.x
    if not optimization_res.success:
        print('Warning: MAP optimization did not succeed')
    
    par_distribution = None # for compatibility with mcmc fit
    return(par, par_distribution)

def cost_function(params, model, out = 'cost', uncertainties = True):
    '''
    Cost function for fitting. Includes physically sensible bounds for parameter.
    
    If uncertainties is True, include errorbars in fit
    
    IMPORTANT :   We minimize the log of the phi_obs and phi_mod, instead of the
                  values themselves. Otherwise the low-mass end would have much higher
                  constribution due to the larger number density.
    '''          
    log_quantity_obs     = model.log_observations[:,0]
    log_phi_obs          = model.log_observations[:,1]  
    
    if uncertainties: # use symmetrized uncertainties
        log_phi_obs_uncertainties = symmetrize_uncertainty(log_phi_obs ,model.log_observations[:,2:])
    else:             # use same uncertainty value for every point
        log_phi_obs_uncertainties = 1
    
    # check if parameter are within bounds
    if not within_bounds(params, *model.feedback_model.bounds):
        return(1e+30) # if outside of bound, return huge value to for cost func
    
    # calculate model ndf
    log_phi_mod = model.log_ndf(log_quantity_obs, params)
    if not np.all(np.isfinite(log_phi_mod)):
        return(1e+30)
    
    # calculate residuals
    res  = (log_phi_obs - log_phi_mod)/log_phi_obs_uncertainties
    
    if out == 'res':
        return(res) # return residuals
    cost = 0.5*np.sum(res**2)
    if out == 'cost':
        return(cost) # otherwise return cost
    
    
################ MAIN FUNCTIONS ###############################################
def symmetrize_uncertainty(log_phi_obs, log_uncertainties):
    '''
    Symmetrize the uncertainties by taking their average. Input shape must be
    (n, 2).
    If any uncertainties are not finite (inf or nan), assign 10* largest errors 
    of the remaining set to them, to be save.
    '''   
    lower_bound = (log_phi_obs - log_uncertainties[:,0])
    upper_bound = (log_phi_obs + log_uncertainties[:,1])
    log_unc     = (upper_bound-lower_bound)/2
    
    # replace nan values with large error estimate
    log_unc[np.logical_not(np.isfinite(log_unc))] = np.nanmax(log_unc)*10
    return(log_unc)
    
