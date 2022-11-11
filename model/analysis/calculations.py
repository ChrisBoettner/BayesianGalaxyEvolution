#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:11:04 2022

@author: chris
"""
import numpy as np

from model.helper import calculate_percentiles, make_list

################ MAIN FUNCTIONS ###############################################
def calculate_expected_black_hole_mass_from_ERDF(ModelResult, lum, z,
                                                 num = 500, sigma=1):
    '''
    Calculates the distribution of expected black hole masses from the 
    conditional ERDF for a given luminosity for a sample of parameter.
    The exact range for the lower and upper percentile limit
    can be chosen using sigma argument, see model.helper.calculate_percentiles 
    for more infos; multiple values can be chosen. Returns dictonary of form 
    (sigma:array) if sigma is  an array, where the array contains x_space value,
    median erdf value and lower and upper percentile for every q2 value.
    '''
    if not np.isscalar(z):
        raise ValueError('Redshift must be scalar quantity.')
    if ModelResult.quantity_name != 'Lbol':
        raise NotImplementedError('Only works for Lbol model.')

    sigma     = make_list(sigma)

    # draw parameter sample and calculate ERDF for every parameter set
    parameter_sample = ModelResult.draw_parameter_sample(z, num=num)
    mbh_dist = [] # distribution of ERDF values
    for p in parameter_sample:
        expected_mbh = ModelResult.\
            calculate_expected_log_black_hole_mass_from_ERDF(
                               lum, z, p)
        mbh_dist.append(expected_mbh)    

    mbh_dist = np.array(mbh_dist) 
    
    # calculate percentiles of ERDF at given sigmas
    erdf = {}
    for s in sigma:
        # calculate percentiles (median, lower, upper) and add x_space
        # to list: (x value, median value, lower bound, upper bound)
        percentiles = calculate_percentiles(mbh_dist,
                                            sigma_equiv=s)
        erdf[s]     = np.array([lum, *percentiles]).T   
    return(erdf)

def calculate_conditional_ERDF_distribution(
                                    ModelResult, lum, z, 
                                    eddington_space=np.linspace(-6, 31, 1000), 
                                    num = 500, sigma=1,
                                    black_hole_mass_distribution=False):
    '''
    Calculates the distribution of values of the ERDF for a given luminosity
    over an Eddington ratio space at redshift z using parameter sample from
    the Lbol model. The exact range for the lower and upper percentile limit
    can be chosen using sigma argument, see model.helper.calculate_percentiles 
    for more infos; multiple values can be chosen. Returns dictonary of form 
    (sigma:array) if sigma is  an array, where the array contains x_space value,
    median erdf value and lower and upper percentile for every q2 value. The x
    space is chosen using the black_hole_mass_distribution argument. If False
    use eddington ratio space, if True it's black hole masses.
    '''
    if not np.isscalar(z):
        raise ValueError('Redshift must be scalar quantity.')
    if not np.isscalar(lum):
        raise NotImplementedError('Not implemented yet for multiple '
                                  'luminosities.')
    if ModelResult.quantity_name != 'Lbol':
        raise NotImplementedError('Only works for Lbol model.')

    sigma     = make_list(sigma)

    # draw parameter sample and calculate ERDF for every parameter set
    parameter_sample = ModelResult.draw_parameter_sample(z, num=num)
    prob_distribution = [] # distribution of ERDF values
    for p in parameter_sample:
        edd_dist = ModelResult.calculate_conditional_ERDF(
                                 lum, z, p, eddington_space, 
                                 black_hole_mass_distribution)
        prob_distribution.append(edd_dist[lum][:,1])    
    
    # save x space (Eddington ratios or black hole masses) and distribution
    # of ERDF values
    x_space           = edd_dist[lum][:,0]
    prob_distribution = np.array(prob_distribution) 
    
    # calculate percentiles of ERDF at given sigmas
    erdf = {}
    for s in sigma:
        # calculate percentiles (median, lower, upper) and add x_space
        # to list: (x value, median value, lower bound, upper bound)
        percentiles = calculate_percentiles(prob_distribution,
                                                   sigma_equiv=s)
        erdf[s]     = np.array([x_space, *percentiles]).T   
    return(erdf)


def calculate_q1_q2_relation(q1_model, q2_model, z, log_q1, num = 500,
                             sigma=1):
    '''
    Calculates the distribution of an observable quantity q2 for a given
    input observable q1 by first calculating a distribution of halo masses
    from q1, and then using this distribution of halo masses to calculate a 
    distribution of q2, using a respective model for each relation. The exact 
    range for the lower and upper percentile limit can be chosen using sigma
    argument, see model.helper.calculate_percentiles for more infos; multiple
    values can be chosen. Returns dictonary of form (sigma:array) if sigma is 
    an array, where the array contains input quantity 1 value, median 
    quantity 2 value and lower and upper percentile for every q2 value. 
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
    return(log_q1_q2_rel)

def calculate_ndf_percentiles(ModelResult, z, num = 5000,
                              sigma=1, quantity_range = None):
    '''
    Calculates the distribution of number densities over a quantity range by
    drawing ndf samples from distribution and calculating their percentiles. 
    The exact range for the lower and upper percentile limit can be chosen 
    using sigma argument, see model.helper.calculate_percentiles for more 
    infos; multiple values can be chosen. Returns dictonary of form 
    (sigma:array) if sigma is an array, where the array contains input 
    quantity 1 value, median quantity 2 value and lower and upper percentile 
    for every q2 value.
    '''  
    if not np.isscalar(z):
        raise ValueError('Redshift must be scalar quantity.')

    sigma           = make_list(sigma)
    
    # calculate halo mass distribution for input quantity q1
    ndf_sample   = ModelResult.get_ndf_sample(z, num=num,
                                              quantity_range=quantity_range)
    log_quantity = ndf_sample[0][:,0]
    
    # get list of all number density for every quantity value
    abundances = np.array([ndf_sample[i][:,1] for i in range(num)])
    ndf_percentiles = {}
    for s in sigma:
        # calculate percentiles (median, lower, upper) and add input
        # quantity to list: (q1 value, ndens value, lower bound, upper bound)
        log_number_densities = calculate_percentiles(abundances,
                                                     sigma_equiv=s)
        ndf_percentiles[s]   = np.array([log_quantity,
                                         *log_number_densities]).T    
    return(ndf_percentiles)

def calculate_qhmr(ModelResult, z, 
                   log_m_halos=np.linspace(8, 14, 100), num=int(5e+3),
                   sigma=1, ratio=False):
    '''
    Calculates the quantity distribution for an array of input halo masses at 
    given redshift. You can input different sigma equivalents (see 
    model.helper.calculate_percentiles for more infos). Returns array
    of form (sigma:array) if sigma is an array, where the array contains 
    input halo mass, median quantity and lower and upper  percentile for
    every halo mass. If sigma is scalar, array is returned directly.
    If ratio is True, return q/m_h, else return q.
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
        if ratio:
            values = log_q_dist - log_m_halos
        else:
            values = log_q_dist 
        # calculate percentiles (median, lower, upper) and add input
        # q1 to list: (q1 value, median q2 value, lower bound, upper bound)
        log_q_percentiles    = calculate_percentiles(values,
                                                     sigma_equiv=s)
        qhmr[s]              = np.array([log_m_halos, *log_q_percentiles]).T
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

