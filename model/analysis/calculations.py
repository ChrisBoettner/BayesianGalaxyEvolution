#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:11:04 2022

@author: chris
"""
import numpy as np

from model.helper import calculate_percentiles, make_list

################ MAIN FUNCTIONS ###############################################
def calculate_q1_q2_relation(q1_model, q2_model, redshifts, 
                             log_q1=np.linspace(8,13,100), num = 500,
                             sigma=1):
    '''
    Calculates the distribution of an observable quantity q2 for a given
    input observable q1 by first calculating a distribution of halo masses
    from q1, and then using this distribution of halo masses to calculate a 
    distribution of q2, using a respective model for each relation.
    Returns dictonary of form (redshift:array), where the array contains 
    input quantity 1 value, median quantity 2 value and lower and upper 
    percentile for every q1 value (at every redshift). The exact range for the
    lower and upper percentile limit can be chosen using sigma argument, see
    model.helper.calculate_percentiles for more infos.
    
    '''   
    redshifts = make_list(redshifts)
    log_q1    = make_list(log_q1)
    
    log_q1_q2_rel = {}
    for z in redshifts:
        # calculate halo mass distribution for input quantity q1
        log_mh_dist = q1_model.calculate_halo_mass_distribution(
                      log_q1, z, num)
        # calculate quantity q2 distribution for every halo mass
        # (you get num halo masses for a q1 and then num q2 values 
        # for each halo mass, so for every q1 you get num^2 
        # q2 values, the array is reshaped accordingly)
        log_q2_dist = q2_model.calculate_quantity_distribution(
                      log_mh_dist, z, num).reshape(num**2,len(log_q1))
        
        # calculate percentiles (median, lower, upper) and add input
        # q1 to list: (q1 value, median q2 value, lower bound, upper bound)
        log_q2_percentiles = calculate_percentiles(log_q2_dist,
                                                   sigma_equiv=sigma)
        log_q1_q2_rel[z]   = np.array([log_q1, *log_q2_percentiles]).T
    return(log_q1_q2_rel)

def calculate_qhmr(ModelResult, redshifts, 
                   log_m_halos=np.linspace(8, 14, 100), num=int(5e+3),
                   sigma=1):
    '''
    Calculates the quantity distribution for an array of input halo masses
    for different redshifts. Returns dictonary of form (redshift:array), where
    the array contains input halo mass, median quantity and lower and upper 
    percentile for every halo mass (at every redshift). The exact range for the
    lower and upper percentile limit can be chosen using sigma argument, see
    model.helper.calculate_percentiles for more infos.
    '''
    redshifts = make_list(redshifts)
    log_m_halos = make_list(log_m_halos)

    qhmr = {}
    for z in redshifts:
        # calculate halo mass distribution for every quantity value
        log_q_dist = ModelResult.calculate_quantity_distribution(
                     log_m_halos, z, num)
        # calculate percentiles (median, lower, upper) and add input
        # halo mass to list: (halo mass, median quantity, lower bound,  
        #                     upper bound)
        log_q_percentiles    = calculate_percentiles(log_q_dist,
                                                     sigma_equiv=sigma)
        qhmr[z]              = np.array([log_m_halos, *log_q_percentiles]).T
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

