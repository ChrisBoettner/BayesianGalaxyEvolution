#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:11:04 2022

@author: chris
"""
import numpy as np

from model.helper import calculate_percentiles, make_list, make_array,\
                         get_uv_lum_sfr_factor, get_return_fraction, z_to_t
from scipy.integrate import cumulative_trapezoid

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
    log_q1    = make_array(log_q1)
    
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
                              sigma=1, quantity_range = None, hmf_z=None,
                              **kwargs):
    '''
    Calculates the distribution of number densities over a quantity range by
    drawing ndf samples from distribution and calculating their percentiles. 
    The exact range for the lower and upper percentile limit can be chosen 
    using sigma argument, see model.helper.calculate_percentiles for more 
    infos; multiple values can be chosen. Returns dictonary of form 
    (sigma:array) if sigma is an array, where the array contains input 
    quantity 1 value, median quantity 2 value and lower and upper percentile 
    for every q2 value. hmf_z chooses z of HMF independently of main z.
    Extra kwargs can be passed to get_ndf_sample.
    '''  
    if not np.isscalar(z):
        raise ValueError('Redshift must be scalar quantity.')

    sigma           = make_list(sigma)
    
    # calculate halo mass distribution for input quantity q1
    ndf_sample   = ModelResult.get_ndf_sample(z, num=num,
                                              quantity_range=quantity_range,
                                              hmf_z=hmf_z, **kwargs)
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
    every halo mass. If ratio is True, return q/m_h, else return q.
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

def calculate_quantity_density(ModelResult, redshift, num_samples=500,
                               num_integral_points=500, sigma=1,
                               return_samples=False):
    '''
    Calculates the quantity density by integrating over ndf at the given 
    redshift values by drawing a parameter sample at z and integrating over
    resulting ndfs. You can input different sigma equivalents (see 
    model.helper.calculate_percentiles for more infos). Returns dictonary
    of form (sigma:array) if sigma is an array, where the array contains 
    input redshift, median density and lower and upper percentile for
    every redshift. The number of samples can be adjusted using num_samples, 
    while the number of points calculated for the integral can be adjusted 
    using num_integral_points.
    If return_samples is True, return dictonaries (z:sample) instead.
    '''  
    redshift        = make_array(redshift)
    sigma           = make_list(sigma)
    
    # calculate densities at every redshift for parameter samples
    quantity_density = [] 
    for z in redshift:
        parameter_sample = ModelResult.draw_parameter_sample(z,
                                                             num=num_samples)
        quantity_density_at_z = []
        for p in parameter_sample:
            quantity_density_at_z.append(ModelResult.
                                         calculate_quantity_density(z, p,
                                                    num=num_integral_points))
        quantity_density.append(np.array(quantity_density_at_z))
    quantity_density=np.array(quantity_density).T
    
    # return complete samples
    if return_samples:
        quantity_dict = {}
        for i, z in enumerate(redshift):
            quantity_dict[z] = np.log10(quantity_density[:,i])
        return(quantity_dict)
    # or return percentiles
    else:
        density_percentiles = {}
        for s in sigma:
            log_densities = np.log10(calculate_percentiles(quantity_density,
                                                           sigma_equiv=s))
            density_percentiles[s]   = np.array([redshift, *log_densities]).T    
    return(density_percentiles)

def calculate_stellar_mass_density(mstar, muv, sigma=1, start_redshift=10,
                                   end_redshift=0, num_samples=500,
                                   num_integral_points=500,
                                   return_samples=False):
    '''
    Calculates the stellar mass density by integrating the SMF. Done in two
    ways, one calculates the stellar mass density directly from the modelled
    SMFs, the other approach estimates a star formation rate density from the
    UV luminosity function which is then integrated. Returns two dictonary
    of form (sigma:array) if sigma is an array, where the array contains 
    input redshift, median density and lower and upper percentile for
    every redshift, first array is for direct method, second array is for
    UVLF method. Start and stop redshift for integration can be chosen using 
    start_redshift and end_redshift arguments. The number of samples can
    be adjusted using num_samples, while the number of points calculated
    for the integral can be adjusted using num_integral_points.
    If return_samples is True, return dictonaries (z:sample) instead.
    '''  
    if not (mstar.quantity_name=='mstar' and muv.quantity_name=='Muv'):
        raise ValueError('First entry must be mstar model and second '
                         'muv model.')
    sigma           = make_list(sigma)
    
    # get conversion factor
    k_uv = get_uv_lum_sfr_factor()
    R    = get_return_fraction()
    
    redshift = np.arange(end_redshift, start_redshift+1)[::-1]
    t        = z_to_t(redshift, mode='age') * 1e+9 # age in Gyr
    
    ## CALCULATE STELLAR MASS DENSITY FROM SMF
    # get sample of stellar mass density at every z
    rho_mstar = []
    for z in redshift:
        par_sample = mstar.draw_parameter_sample(z, num=num_samples)
        rho_mstar_at_z = []
        for p in par_sample:
            rho_mstar_at_z.append(mstar.calculate_quantity_density(z, p,
                                                    num=num_integral_points))
        rho_mstar.append(rho_mstar_at_z)
    rho_mstar = np.array(rho_mstar)
    
    ## CALCULATE STELLAR MASS DENSITY FROM UVLF
    # get sample of UV luminosity density at every z
    rho_muv = []
    for z in muv.redshift:
        par_sample = muv.draw_parameter_sample(z, num=num_samples)
        rho_muv_at_z = []
        for p in par_sample:
            rho_muv_at_z.append(muv.calculate_quantity_density(z, p, 
                                                    num=num_integral_points))
        rho_muv.append(rho_muv_at_z)
    # convert to star formation rate density
    sfr_density = np.array(rho_muv) * k_uv
    # integrate SFR density to get stellar mass density
    inferred_rho_mstar = (1-R) * cumulative_trapezoid(sfr_density, t, axis=0)
    # add integration constant at start_redshift so that values coincide there
    inferred_rho_mstar = inferred_rho_mstar + rho_mstar[0]
    inferred_rho_mstar = np.insert(inferred_rho_mstar, 0, rho_mstar[0],
                                   axis=0)
    
    # return complete samples
    if return_samples:
        rho_dict, inferred_rho_dict = {}, {}
        for i, z in enumerate(redshift):
            rho_dict[z]          = np.log10(rho_mstar[i,:])
            inferred_rho_dict[z] = np.log10(inferred_rho_mstar[i,:])
        return(rho_dict, inferred_rho_dict)
    # or return percentiles
    else:
        density_percentiles          = {}
        inf_density_percentiles      = {}
        for s in sigma:
            log_densities      = np.log10(calculate_percentiles(rho_mstar.T,
                                                                sigma_equiv=s))
            log_inf_densities  = np.log10(calculate_percentiles(
                                            inferred_rho_mstar.T, 
                                            sigma_equiv=s))
            
            density_percentiles[s]     = np.array([redshift, 
                                                   *log_densities]).T[::-1]  
            inf_density_percentiles[s] = np.array([redshift, 
                                                   *log_inf_densities]).T[::-1] 
        return(density_percentiles, inf_density_percentiles)

def calculate_best_fit_ndf(ModelResult, redshift, quantity_range=None):
    '''
    Calculate best fit number density function by passing calculated best fit
    parameter to calculate_ndf method. Returns array of form {redshift:ndf}
    '''

    redshift = make_list(redshift)
    best_fit_ndfs = {}
    for z in redshift:
        quantity, phi = ModelResult.calculate_ndf(
            z, ModelResult.parameter.at_z(z),
            quantity_range=quantity_range)
        best_fit_ndfs[z] = np.array([quantity, phi]).T
    return(best_fit_ndfs)