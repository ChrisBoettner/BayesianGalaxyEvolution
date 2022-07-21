#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:11:04 2022

@author: chris
"""
import numpy as np

from model.helper import calculate_percentiles, make_list

################ MAIN FUNCTIONS ###############################################
def calculate_q1_q2_relation(q1_model, q2_model, z, 
                             log_q1=np.linspace(8,13,100), num = 2000,
                             sigma=1):
    '''
    Calculates the distribution of an observable quantity q2 for a given
    input observable q1 by first calculating a distribution of halo masses
    from q1, and then using this distribution of halo masses to calculate a 
    distribution of q2, using a respective model for each relation. The exact 
    range for the lower and upper percentile limit can be chosen using sigma
    argument, seemodel.helper.calculate_percentiles for more infos; multiple
    values can be chosen. Returns dictonary of form (sigma:array) if sigma is 
    an array, where the array contains input quantity 1 value, median 
    quantity 2 value and lower and upper percentile for every q1 value. 
    If sigma is scalar, array is returned directly. 
    '''  
    if not np.isscalar(z):
        raise ValueError('Redshift must be scalar quantity.')
    
    sigma     = make_list(sigma)
    log_q1    = make_list(log_q1)
    
    # calculate halo mass distribution for input quantity q1
    log_mh_dist = q1_model.calculate_halo_mass_distribution(
                  log_q1, z, num=num)
    # calculate quantity q2 distribution for every halo mass
    # (you get num halo masses for a q1 and then num q2 values 
    # for each halo mass, so for every q1 you get num^2 
    # q2 values, the array is reshaped accordingly)
    log_q2_dist = q2_model.calculate_quantity_distribution(
                  log_mh_dist, z, num=num).reshape(num**2,len(log_q1))
    
    log_q1_q2_rel = {}
    for s in sigma:
        # calculate percentiles (median, lower, upper) and add input
        # q1 to list: (q1 value, median q2 value, lower bound, upper bound)
        log_q2_percentiles = calculate_percentiles(log_q2_dist,
                                                   sigma_equiv=s)
        log_q1_q2_rel[s]   = np.array([log_q1, *log_q2_percentiles]).T
        
    # if simga scalar, return array instead of dict
    if len(sigma)==1:
        log_q1_q2_rel = log_q1_q2_rel[s]
    return(log_q1_q2_rel)

def calculate_qhmr(ModelResult, z, 
                   log_m_halos=np.linspace(8, 14, 100), num=int(5e+3),
                   sigma=1):
    '''
    Calculates the quantity distribution for an array of input halo masses at 
    given redshift. You can input different sigma equivalents (see 
    model.helper.calculate_percentiles for more infos). Returns array
    of form (sigma:array) if sigma is an array, where the array contains 
    input halo mass, median quantity and lower and upper  percentile for
    every halo mass. If sigma is scalar, array is returned directly.
    '''
    if not np.isscalar(z):
        raise ValueError('Redshift must be scalar quantity.')
    
    sigma       = make_list(sigma)
    log_m_halos = make_list(log_m_halos)
    
    # calculate halo mass distribution for every quantity value
    log_q_dist = ModelResult.calculate_quantity_distribution(
                 log_m_halos, z, num)
    # calculate percentiles (median, lower, upper) and add input
    # halo mass to list: (halo mass, median quantity, lower bound,  
    #                     upper bound)
    
    qhmr = {}
    for s in sigma:
        # calculate percentiles (median, lower, upper) and add input
        # q1 to list: (q1 value, median q2 value, lower bound, upper bound)
        log_q_percentiles    = calculate_percentiles(log_q_dist,
                                                     sigma_equiv=s)
        qhmr[s]              = np.array([log_m_halos, *log_q_percentiles]).T
    # if simga scalar, return array instead of dict
    if len(sigma)==1:
        qhmr = qhmr[s]
    return(qhmr)


def calculate_best_fit_ndf(ModelResult, redshifts, quantity_range=None):
    '''
    Calculate best fit number density function by passing calculated best fit
    parameter to calculate_ndf method. Returns array of form {redshift:ndf}
    '''

    redshifts = make_list(redshifts)
    best_fit_ndfs = {}
    for z in redshifts:
        quantity, phi = ModelResult.calculate_ndf(
            z, ModelResult.parameter.at_z(z),
            quantity_range=quantity_range)
        best_fit_ndfs[z] = np.array([quantity, phi]).T
    return(best_fit_ndfs)

