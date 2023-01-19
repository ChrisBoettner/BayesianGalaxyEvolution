#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 10:13:43 2023

@author: chris
"""

import numpy as np
import pandas as pd

from model.data.load import load_surveys
from model.helper import calculate_percentiles, make_list, make_array,\
                         mag_to_lum
from model.analysis.calculations import calculate_quantity_density

from astropy.cosmology import Planck18
from astropy.units import arcmin, Mpc

def tabulate_expected_number(model, instrument, to_latex=True, caption=''):
    '''
    Tabulate expected number of galaxies in UV for different surveys and at
    different redshits. Expected number is given as upper limit by the 
    95th percentile.
    instrument currently 'JWST' or 'Euclid'. 
    If to_latex=False, return pandas DataFrame directly.
    '''
    data = load_surveys()
    
    try:
        survey_data = data[instrument]
    except:
        raise ValueError('instrument name not known.')
    
    if instrument == 'JWST':
        redshift = [7, 9, 11, 13, 15]
    elif instrument == 'Euclid':
        redshift = [5, 6, 7, 8, 11]
    else:
        raise ValueError('instrument name not known.')
    
    survey_names = survey_data.keys()

    # pre-load DataFrame
    formatted_DataFrame = pd.DataFrame(index   = range(len(survey_names)),
                                       columns = range(len(redshift)+3))    
    
    # calculate upper number limits at each redshift for each survey and
    # write to DataFrame
    for i, survey_name in enumerate(survey_names):
        #calculate upper limits on numbers at each redshift
        survey_properties = survey_data[survey_name]

        numbers = calculate_expected_number(
                                    model, 
                                    redshift, 
                                    area_arcmin2=survey_properties[0],
                                    limiting_magnitude=survey_properties[1], 
                                    sigma=2, 
                                    num_samples=int(1e+4),                                      
                                    percentiles_mode='percentiles'
                                    )[2]
        number_limits = np.rint(numbers[:,3]).astype(int) # number limits as 
                                                          # integers
        formatted_DataFrame.loc[i,0]  = survey_name
        formatted_DataFrame.loc[i,1]  = survey_properties[0]
        formatted_DataFrame.loc[i,2]  = survey_properties[1]
        formatted_DataFrame.loc[i,3:] = number_limits
        
    # create header
    header = [rf'$z={z}$' for z in redshift]
    header.insert(0, r'$5\sigma$ Depth')
    header.insert(0, r'Area (arcmin$^2$)')
    header.insert(0, r'Survey')
    formatted_DataFrame.columns=header
    
    # create Latex table
    if to_latex:
        column_format = 'lrr|' + 'r' * len(redshift)
        latex_table = formatted_DataFrame.to_latex(index=False,
                                                   escape=False,
                                                   column_format=column_format,
                                                   caption=caption)
        return(latex_table)
    else:
        return(formatted_DataFrame)
    

def calculate_expected_number(ModelResult, redshift, area_arcmin2, 
                              limiting_magnitude, num_samples=500, 
                              num_integral_points=500, sigma=1,
                              upper_magnitude_limit=-50,
                              return_samples=False,
                              percentiles_mode='percentiles'):

    '''
    Calculate the expected number of galaxies for a given redshift, 
    survey area (in arcmin^2) and limiting_magnitude, by calculating survey
    volume (assumine delta_z = 1) and expected number density integrated up to
    limiting magnitude. Returns dictonary of form (sigma:array), where the 
    array contains input redshift, median number and lower and upper percentile
    for every redshift. The number of samples can be adjusted using num_samples, 
    while the number of points calculated for the integral can be adjusted 
    using num_integral_points. The upper integration limit can be adjusted using
    upper_magnitude_limit.
    If return_samples is True, return dictonaries (z:sample) instead.
    If percentiles_mode is 'uncertainties', the array contains the upper and
    lower uncertainty rather than percentiles (difference between percentile 
    and median).
    '''
    if ModelResult.quantity_name != 'Muv':
        raise ValueError('Only works for Muv model.')
    
    redshift = make_array(redshift)
    sigma    = make_list(sigma)
    
    expected_number = []
    for z in redshift:
        ## calculate volume
        # unit conversion from arcsec to cMpc^2
        length_scale = Planck18.arcsec_per_kpc_comoving(
                                            z).to(arcmin/Mpc).value
        area = area_arcmin2/length_scale**2
        # calculate depth with delta z = 1 in cMpc
        depth = np.diff(Planck18.comoving_distance([z-0.5, 
                                                    z+0.5]).value)[0]
        # calculate volume in cMpc^3
        volume = area*depth
        
        ## get number density
        # calculate luminosity space
        limiting_magnitude_at_z = (limiting_magnitude 
                                   - Planck18.distmod(z).value)
        limiting_luminosity     = np.log10(mag_to_lum(
                                            limiting_magnitude_at_z)[0])
        upper_luminosity        = np.log10(mag_to_lum(
                                            upper_magnitude_limit)[0])
        log_q_space = np.linspace(limiting_luminosity, upper_luminosity, 
                                  num_integral_points)
        # calculate number density
        log_density = calculate_quantity_density(
                                       ModelResult, z, 
                                       log_q_space=log_q_space,
                                       num_samples=num_samples, 
                                       num_integral_points=num_integral_points, 
                                       return_samples=True, 
                                       number_density=True)
        ## calculate expected number
        expected_number.append(np.power(10, log_density[z]) * volume)
    expected_number=np.array(expected_number).T
        
    # return complete samples
    if return_samples:
        number_dict = {redshift[i]:expected_number[:,i] 
                       for i in range(len(redshift))}
        return(number_dict)
    # or return percentiles
    else:
        number_percentiles = {}
        for s in sigma:
            log_densities = calculate_percentiles(expected_number,
                                                  sigma_equiv=s,
                                                  mode=percentiles_mode)
            number_percentiles[s]   = np.array([redshift, *log_densities]).T
        return(number_percentiles)