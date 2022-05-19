#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 00:20:31 2022

@author: chris
"""
import numpy as np

def log_schechter_function(log_quantity, log_phi_star, log_quant_star, alpha):
    '''
    Calculate the value of Schechter function log10(d(n)/d(obs)) for an
    observable (in base10 log), using Schechter parameters.
    '''
    x = log_quantity - log_quant_star

    # normalisation
    norm = np.log10(np.log(10)) + log_phi_star
    # power law
    power_law = (alpha + 1) * x
    # exponential
    if np.any(np.abs(x) > 80):  # deal with overflow
        exponential = np.sign(x) * np.inf
    else:
        exponential = -np.power(10, x) / np.log(10)
    return(norm + power_law + exponential)


def log_schechter_function_mag(magnitude, log_phi_star, mag_star, alpha):
    '''
    Calculate Schechter function if input is magnitude and not log_quantity.
    '''
    x = 0.4 * (mag_star - magnitude)

    # normalisation
    norm = np.log10(0.4) + np.log10(np.log(10)) + log_phi_star
    # power law
    power_law = (alpha + 1) * x
    # exponential
    if np.any(np.abs(x) > 80):  # deal with overflow
        exponential = np.sign(x) * np.inf
    else:
        exponential = - np.power(10, x) / np.log(10)
    return(norm + power_law + exponential)
