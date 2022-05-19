#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 12:24:46 2022

@author: chris
"""
import numpy as np

def log_double_power_law(log_quantity, log_phi_star, log_quant_star, gamma_1,
                  gamma_2):
    '''
    Calculate the value of double power law log10(d(n)/dlog10(obs)) for an
    observable (in base10 log), using double power law parameters.
    '''
    log_solar_luminosity = 33.585

    x = log_quantity - (log_quant_star + log_solar_luminosity) # characteristic 
                                                               # value in 
                                                               # solar luminosities

    # normalisation
    norm = log_phi_star
    # power law
    power_law_1 = (gamma_1) * x
    power_law_2 = (gamma_2) * x
    
    # deal with overflows
    denominator = np.empty_like(x)
    
    overflow_check = 50
    overflow_mask  = (power_law_1<overflow_check)*(power_law_1<overflow_check)
    
    denominator[overflow_mask] = 10**power_law_1[overflow_mask] +\
                                 10**power_law_2[overflow_mask]
                                 
    denominator[np.logical_not(overflow_mask)] = np.inf
    
    # deal with negative values
    double_power_law = np.empty_like(denominator)
    invalid_mask     = denominator<= 0
    double_power_law[invalid_mask] = 0
    
    # calculate values
    valid_mask = np.logical_not(invalid_mask)
    double_power_law[valid_mask] = (norm -
                                    np.log10(denominator[valid_mask]))
    
    return(double_power_law)