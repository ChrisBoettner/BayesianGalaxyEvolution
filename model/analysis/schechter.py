#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 00:20:31 2022

@author: chris
"""
import numpy as np
from scipy.optimize import curve_fit

from model.helper import make_list, mag_to_lum, lum_to_mag

################ MAIN FUNCTIONS ############################################### 

def get_schecher_parameter_distribution(ModelResult, redshift, num = 100):
    '''
    Calculate distribution of Schechter parameter for model at a given redshift.
    Do this by calculating num model ndfs and fitting Schechter functions to 
    each one. Returns dictonary of form {redshift:distribution}.
    '''
    redshift = make_list(redshift)
    
    # define initial guess for fit
    if ModelResult.quantity_name == 'mstar':
        p0 = [-4, 10, -1.5] # log_phi_star, characteristic mass, slope
    elif ModelResult.quantity_name == 'Muv':
        p0 = [-4, -20, -1.5] # log_phi_star, characteristic magnitude, slope
    else:
        raise ValueError('quantity_name not known.')    
    
    # calculate distributions
    schechter_parameter_distribution = {}
    for z in redshift:
        parameter_at_z = []
        ndf_sample     = ModelResult.get_ndf_sample(z, num = num)
        
        for ndf in ndf_sample:
            # fit correct Schechter function depending on input
            if ModelResult.quantity_name == 'mstar':
                params = fit_schechter_function(ndf, p0 = p0)
            elif ModelResult.quantity_name == 'Muv':
                params = fit_mag_schechter_function(ndf, p0 = p0)
            else:
                raise ValueError('quantity_name not known.')    
                
            parameter_at_z.append(params)
        
        schechter_parameter_distribution[z] = np.array(parameter_at_z)
    return(schechter_parameter_distribution)


################ BASE FUNCTIONS ############################################### 
def fit_schechter_function(data, p0, uncertainties = None):
    '''
    Fit Schechter function to data. May include uncertainties.

    '''
    fit_parameter, _ = curve_fit(log_schechter_function, data[:,0], data[:,1],
                                 sigma = uncertainties, p0 = p0,
                                 maxfev = int(1e+5))
    return(fit_parameter)

def fit_mag_schechter_function(data, p0, uncertainties = None):
    '''
    Fit Magnitude Schechter function to data. May include uncertainties.

    '''
    schechter_parameter, _ = curve_fit(log_schechter_function_mag, data[:,0],
                                       data[:,1], sigma = uncertainties, p0 = p0,
                                       maxfev = int(1e+5))
    return(schechter_parameter)

def log_schechter_function(log_quantity, log_phi_star, log_quant_star, alpha):
    '''
    Calculate the value of Schechter function log10(d(n)/dlog10(obs)) for an
    observable (in base10 log), using Schechter parameters.
    '''
    x           = log_quantity-log_quant_star
    
    # normalisation
    norm        = np.log10(np.log(10)) + log_phi_star
    # power law
    power_law   = (alpha+1)*x
    # exponential
    if np.any(np.abs(x)>80): # deal with overflow
        exponential = np.sign(x)*np.inf
    else:
        exponential = -np.power(10,x)/np.log(10)
    return(norm + power_law + exponential)

def log_schechter_function_mag(magnitude, log_phi_star, mag_star, alpha):
    '''
    Calculate Schechter function if input is magnitude and not log_quantity.
    '''
    x           = 0.4 * (mag_star - magnitude)
    
    # normalisation
    norm        = np.log10(0.4) + np.log10(np.log(10)) + log_phi_star
    # power law
    power_law   = (alpha+1)*x
    # exponential
    if np.any(np.abs(x)>80): # deal with overflow
        exponential = np.sign(x)*np.inf
    else:
        exponential = - np.power(10, x)/np.log(10)
    return(norm + power_law + exponential)