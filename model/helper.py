#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:16:21 2022

@author: chris
"""
import numpy as np
import pandas as pd
import os
from functools import wraps
from time import time

from scipy.optimize import root_scalar, curve_fit
from scipy.stats import gaussian_kde

import astropy.units as u
from astropy.cosmology import Planck18, z_at_value

################ PHYSICS ######################################################

def get_uv_lum_sfr_factor():
    '''
    Conversion factor between UV luminosity and star formation rate for
    Chabrier IMF. 
    '''
    return(1.4*1e-28)

def get_return_fraction():
    ''' Get return fraction for Chabrier IMF. '''
    return(0.41)

def lum_to_mag(L_nu):
    '''
    Convert luminosity (ergs s^-1 Hz^-1) to Absolute Magnitude.
    
    Reference:
    https://astronomy.stackexchange.com/questions/39806/relation-between-absolute-
    magnitude-of-uv-and-star-formation-rate
    '''
    L_nu = make_array(L_nu)
    d = 3.086e+19                # 10pc in cm
    flux = L_nu / (4 * np.pi * d**2)
    M_uv = -2.5 * np.log10(flux) - 48.6  # definition in AB magnitude system
    return(M_uv)

def mag_to_lum(M_uv):
    '''
    Convert Absolute Magnitude to luminosity (ergs s^-1 Hz^-1).
    '''
    M_uv = make_array(M_uv)
    d = 3.086e+19  # 10pc in cm
    log_L = (M_uv + 48.6) / (-2.5) + np.log10(4 * np.pi * d**2)
    return(np.power(10, log_L))

def log_L_uv_to_log_sfr(log_L_uv):
    '''
    Convert log luminosity (ergs s^-1 Hz^-1) to log star formation rate.
    '''
    log_L_uv = make_array(log_L_uv)
    k_uv     = get_uv_lum_sfr_factor()
    return(log_L_uv+np.log10(k_uv))
def log_sfr_to_log_L_uv(log_sfr):
    '''
    Convert log star formation rate to log luminosity (ergs s^-1 Hz^-1).
    '''
    log_sfr = make_array(log_sfr)
    k_uv    = get_uv_lum_sfr_factor()
    return(log_sfr-np.log10(k_uv))

def log_L_bol_to_log_L_band(log_Lbol, 
                            correction_factors=(4.073, -0.026, 12.60, 0.278)):
    '''
    Convert AGN bolometric luminosity to band luminosity using correction_factors and
    relation from Shen2020. Default correction factors are for hard X-ray (2-10 keV) 
    band.
    '''
    log_Lbol = make_array(log_Lbol)
    
    log_solar_luminosity = 33.585 # in erg/s
    ref_L = 10**(log_Lbol - (10 + log_solar_luminosity))
    
    power_laws = (correction_factors[0] * ref_L ** correction_factors[1] +
                  correction_factors[2] * ref_L ** correction_factors[3])
    
    log_Lband = log_Lbol - np.log10(power_laws)
    return(log_Lband)

def z_to_t(redshift, mode='lookback_time'):
    '''
    Convert redshift to time (in Gyr) in Planck18 cosmology. You can choose 
    between lookback time and age.
    '''
    redshift = make_list(redshift)
    if mode == 'lookback_time':
        t = np.array([Planck18.lookback_time(z).value for z in redshift])
    elif mode == 'age':
        t = np.array([Planck18.age(z).value for z in redshift])     
    else:
        raise NotImplementedError('mode not known.')
    return(t)
def t_to_z(t, mode='lookback_time'):
    '''
    Convert time (in Gyr) to redshift in Planck18 cosmology. You can choose 
    between lookback time and age.
    '''
    t = make_list(t)
    if mode == 'lookback_time':
        redshift = np.array([z_at_value(Planck18.lookback_time,
                                        k * u.Gyr).value for k in t])
    if mode == 'lookback_time':
        redshift = np.array([z_at_value(Planck18.age,
                                        k * u.Gyr).value for k in t])
    else:
        raise NotImplementedError('mode not known.')
    return(redshift)

################ MATH #########################################################


def within_bounds(values, lower_bounds, upper_bounds):
    '''
    Checks if list of values is (strictly) within lower and upper bounds. 
    All three arrays must have same length.
    '''
    values, lower_bounds, upper_bounds = make_list(values),\
        make_list(lower_bounds),\
        make_list(upper_bounds)

    is_within = []
    for i in range(len(values)):
        is_within.append(
            (values[i] >= lower_bounds[i]) & (
                values[i] <= upper_bounds[i]))

    # return True if all elements are within bounds, otherwise False
    return(all(is_within))


def invert_function(func, fprime, fprime2, x0_func, y, args, **kwargs):
    '''
    For a function y=f(x), calculate x values for an input set of y values.
    '''
    x = []
    for val in y:
        def root_func(m, *args):
            return(func(m, *args) - val)

        x0_in = x0_func(val, *args)  # guess initial value

        root = root_scalar(root_func, fprime=fprime, fprime2=fprime2,
                           args=args, method='halley', x0=x0_in,
                           **kwargs).root
        # if Halley's method doesn't work, try Newton
        if np.isnan(root):
            root = root_scalar(root_func, fprime=fprime, fprime2=fprime2,
                               args=args, method='newton', x0=x0_in,
                               **kwargs).root
        x.append(root)
    x = np.array(x)
    return(x)


def calculate_percentiles(data, axis=0, sigma_equiv=1, mode='percentiles'):
    '''
    Returns median, lower and upper percentile of data. The exact percentiles
    are chosen using sigma_equiv, which is the corresponding probability for 
    a gaussian distribution. 
    sigma_equiv=1 -> 68%
    sigma_equiv=2 -> 95%
    sigma_equiv=3 -> 99.7%
    If mode=='percentiles' return percentiles directly, if 
    mode=='uncertainties', return difference between upper/lower percentile
    and mean.
    
    IMPORTANT: inf values are converted to NaN, and NaN are ignored in 
               percentile calculation.
    '''
    sigmas = {1: (50, 16   , 84   ),
              2: (50,  2.5 , 97.5 ),
              3: (50,  0.15, 99.85),
              4: (50, 0.0003, 99.99997),
              5: (50, 0.0000003, 99.9999997)}
    try:
        sigma = sigmas[sigma_equiv]
    except KeyError:
        raise KeyError('sigma_equiv must be value in [1,2,3,4,5].')
    
    data = make_array(data)  # turn into numpy array if not already
    data[~np.isfinite(data)] = np.nan # ignore inf values
        
    # percentiles in order: median, lower, upper
    percentiles = np.nanpercentile(data, sigma, axis=axis)
    
    # return percentiles or errors depending on mode
    if mode == 'percentiles':
        pass
    elif mode == 'uncertainties':
        if axis == 0:
            percentiles[1] = percentiles[0] - percentiles[1]
            percentiles[2] = percentiles[2] - percentiles[0]
        else:
            raise NotImplementedError('uncertainties mode currently only '
                                      'supports percentile calculation along '
                                      'axis=0')
    else:
        raise ValueError('mode must be either \'percentiles\' or'
                         ' \'uncertainties\'')
    return(percentiles)


def calculate_limit(func, initial_value, rtol=1e-4,
                    max_evaluations=int(1e+5)):
    '''
    Simple function to calculate the limit of an input function as
    x -> infinity . Start at x=initial value and increase value lineary.
    Convergence is reached when relative change is <= rtol. An error is 
    raised when maximum number of iterations is exceeded.
    '''

    x = initial_value
    value_old = func(x)

    # calculate limit
    for i in range(max_evaluations):
        x = x + initial_value/10
        value_new = func(x)

        if value_old == 0:
            value_old = value_new
        elif np.abs(value_new-value_old)/value_old < rtol:
            return(value_new)
        else:
            value_old = value_new

    # raise error if convergence could not be achieved
    raise StopIteration(
        'Limit could not be calculated within max_evaluations.')
    return


def fit_function(function, data, p0, uncertainties=None,
                 bounds=(-np.inf, np.inf)):
    '''
    Fit function to data. May include uncertainties and bounds.

    '''
    data = make_array(data)
    # remove infinites and nans
    data = data[np.isfinite(data[:, 1])]
    if len(data) == 0:
        return(np.array([np.nan, np.nan, np.nan]))

    fit_parameter, _ = curve_fit(function, data[:, 0], data[:, 1],
                                 sigma=uncertainties, p0=p0,
                                 bounds=bounds,
                                 maxfev=int(1e+5))
    return(fit_parameter)

################ SORTING AND FILTERING ########################################


def sort_by_density(data, keep=None):
    '''
    Estimate density of data using Gaussian KDE and return data sorted by
    density and density values. Can choose which columns to keep before
    creating density map using keep argument.
    '''
    data = make_array(data)
    if keep:
        data = data[:,keep]
    density = gaussian_kde(data.T).evaluate(data.T)
    idx = density.argsort()
    sorted_data, density = data[idx], density[idx]
    return(sorted_data, density)


def is_sublist(list1, list2):
    '''
    Check if list2 contains all elements of list 1.
    '''
    return(set(list1) <= set(list2))

################ PROGRESS TRACKING ############################################


def custom_progressbar():
    ''' Define how progressbar should look like. '''
    from progressbar import ProgressBar, Counter, FormatLabel, Timer,\
        FileTransferSpeed

    widgets = [FormatLabel(''), ' ||| ', Counter('Iteration: %(value)d'),
               ' ||| ',  Timer('%(elapsed)s'), ' ||| ',
               FileTransferSpeed(unit='it'), ' ||| ', FormatLabel('')]

    progressbar = ProgressBar(widgets=widgets)
    return(progressbar)

################ ERROR HANDLING ###############################################


def catch_warnings(catch=True):
    ''' Catch warnings as if they were exceptions '''
    import warnings
    if catch:
        warnings.filterwarnings('error')
    else:
        warnings.filterwarnings('default')
    return()

################ TYPE CHECKING ################################################


def make_list(variable):
    '''
    Makes input variable into a list if it is not one already. Needed for
    functions that may take scalars or arrays.
    '''
    if isinstance(variable, (list, range, pd.core.series.Series, np.ndarray)):
        return(list(variable))
    else:
        return([variable])


def make_array(variable):
    '''
    Makes input variable into a array if it is not one already. Needed for
    functions that may take scalars or arrays.
    '''
    if isinstance(variable, np.ndarray):
        return(variable)
    elif isinstance(variable, (list, range, pd.core.series.Series)):
        return(np.array(variable))
    else:
        return(np.array([variable]))


def pick_from_list(variable, ind):
    '''
    Pick specific element from list, if variable is list. If variable is scalar,
    just return variable.
    '''
    if isinstance(variable, (list, pd.core.series.Series, np.ndarray)):
        return(variable[ind])
    else:
        return(variable)

################ COMPUTER PATHS ###############################################


def system_path():
    '''
    Choose path to save files depending on if code is run on cluster of my 
    laptop.
    '''
    path = os.getcwd() + '/model/data/MCMC_chains/'
    return(path)

################ TIMING #######################################################
def timing(decimals=2):
    '''Timing function usable as decorator'''
    # outer function to accept argument for number of decimal points for timer
    def decorator(function):
        # decorator function
        @wraps(function)
        def wrap(*args, **kwargs):
            # wrapper for timing
            ts = time()
            result = function(*args, **kwargs)
            te = time()
            t  = np.around(te-ts, decimals)
            print(f'Function \'{function.__name__}\' took {t} seconds.')
            return(result)
        return(wrap)
    return(decorator)
