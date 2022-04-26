#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:16:21 2022

@author: chris
"""
import numpy as np
import pandas as pd
import os

from scipy.optimize import root_scalar
from scipy.stats import gaussian_kde

import astropy.units as u
from astropy.cosmology import Planck18, z_at_value

################ PHYSICS ######################################################


def lum_to_mag(L_nu):
    '''
    Convert luminosity (ergs s^-1 Hz^-1) to Absolute Magnitude.
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


def z_to_t(z):
    '''
    Convert redshift to lookback time (in Gyr) in Planck18 cosmology.
    '''
    z = make_list(z)
    t = np.array([Planck18.lookback_time(k).value for k in z])
    return(t)


def t_to_z(t):
    '''
    Convert lookback time (in Gyr) to redshift in Planck18 cosmology.
    '''
    t = make_list(t)
    z = np.array(
        [z_at_value(Planck18.lookback_time, k * u.Gyr).value for k in t])
    return(z)

################ MATH #########################################################


def within_bounds(values, lower_bounds, upper_bounds):
    '''
    Checks if list of values is within lower and upper bounds (strictly
    within bounds). All three arrays must have same length.
    '''
    is_within = []
    for i in range(len(values)):
        is_within.append(
            (values[i] > lower_bounds[i]) & (
                values[i] < upper_bounds[i]))

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


def calculate_percentiles(data, axis=0):
    '''
    Returns median, 16th and 84th percentile of data.
    '''
    data = make_array(data)  # turn into numpy array if not already

    median = np.percentile(data, 50, axis=axis)
    lower  = np.percentile(data, 16, axis=axis)
    upper  = np.percentile(data, 84, axis=axis)
    return([median, lower, upper])

################ SORTING AND FILTERING ########################################


def sort_by_density(data):
    '''
    Estimate density of data using Gaussian KDE and return data sorted by
    density and density values.
    '''
    data = make_array(data)
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
        
    widgets = [Counter('Iteration: %(value)d'),' ||| ', Timer('%(elapsed)s'), 
               ' |||', FileTransferSpeed(unit='it'),' |||', FormatLabel('')]
    
    progressbar = ProgressBar(widgets=widgets)
    return(progressbar)
    
################ TYPE CHECKING ################################################


def make_list(variable):
    '''
    Makes input variable into a list if it is not one already. Needed for
    functions that may take scalars or arrays.
    '''
    if isinstance(variable, (list, pd.core.series.Series, np.ndarray)):
        return(variable)
    else:
        return([variable])


def make_array(variable):
    '''
    Makes input variable into a array if it is not one already. Needed for
    functions that may take scalars or arrays.
    '''
    if isinstance(variable, np.ndarray):
        return(variable)
    elif isinstance(variable, (list, pd.core.series.Series)):
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
    Choose path to save files depending on if code is run on cluster of my laptop.
    '''
    path = '/data/p305250/mcmc_runs/'
    if os.path.isdir(path):  # if path exists use this one (cluster structure)
        pass
    else:  # else use path for home computer
        path = '/home/chris/Desktop/mcmc_runs/'
    return(path)