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

def cost_function(params, model, out = 'cost', space = 'linear',
                  uncertainties = True):
    '''
    Cost function for fitting. Includes physically sensible bounds for parameter.
    
    Choose if you want to fit linear space (res = phi_obs - phi_mod) or
    log space (res = log_phi_obs - log_phi_mod).
    
    If uncertainties is True, include errorbars in fit.
    '''          
    log_quantity_obs     = model.log_observations[:,0]
    log_phi_obs          = model.log_observations[:,1]  
    
    # check if parameter are within bounds
    if not within_bounds(params, *model.feedback_model.bounds):
        return(1e+30) # if outside of bound, return huge value to for cost func
    
    # calculate model ndf
    log_phi_mod = model.log_ndf(log_quantity_obs, params)
    if not np.all(np.isfinite(log_phi_mod)):
        return(1e+30)
    
    # calculate residuals 
    if space == 'linear':
        res = 10**log_phi_obs - 10**log_phi_mod
    if space == 'log':
        res = log_phi_obs - log_phi_mod 
    
    # calculate weights
    if uncertainties:
        weights = calculate_weights(model, space = space)
    else:
        weights = 1
    
    weighted_res = res * weights
    
    if out == 'res':
        return(weighted_res) # return residuals
    cost = np.sum(weighted_res**2)
    if out == 'cost':
        return(cost) # otherwise return cost
    
    
################ UNCERTAINTIES AND WEIGHTS ####################################
def calculate_weights(model, space):
    '''
    Calculate weights for residuals based on measurement uncertainties.
    '''
    log_phi_obs               = model.log_observations[:,1]  
    log_phi_obs_uncertainties = model.log_observations[:,2:]  
    
    # calculate uncertainties
    uncertainties = symmetrize_uncertainty(log_phi_obs, log_phi_obs_uncertainties,
                                           space)
    
    weights = 1/uncertainties
    return(weights)    

def symmetrize_uncertainty(log_phi_obs, log_uncertainties, space):
    '''
    Symmetrize the uncertainties by taking their average. Input shape must be
    (n, 2).
    Choose if you want to symmetrize in linear space, or log space.
    
    If any uncertainties are not finite (inf or nan), assign 10* largest errors 
    of the remaining set to them, to be save.
    '''
    if space == 'linear':
        lower_bound = 10**(log_phi_obs - log_uncertainties[:,0])
        upper_bound = 10**(log_phi_obs + log_uncertainties[:,1]) 
    if space == 'log':
        lower_bound = (log_phi_obs - log_uncertainties[:,0])
        upper_bound = (log_phi_obs + log_uncertainties[:,1])

    uncertainty = (upper_bound-lower_bound)/2
    
    # replace nan values with large error estimate
    uncertainty[np.logical_not(np.isfinite(uncertainty))] = np.nanmax(uncertainty)*10
    return(uncertainty)