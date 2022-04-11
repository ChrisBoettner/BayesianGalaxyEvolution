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

################ PHYSICS ######################################################
# convert between luminosity and abosulte magntidue (luminosity given in 
#  ergs s^-1 Hz^-1)
def lum_to_mag(L_nu):
    '''
    Convert luminosity (ergs s^-1 Hz^-1) to Absolute Magnitude.
    '''   
    d    = 3.086e+19                # 10pc in cm
    flux = L_nu/(4*np.pi*d**2)
    M_uv = -2.5*np.log10(flux)-48.6 # definition in AB magnitude system
    return(M_uv)

def mag_to_lum(M_uv):
    '''
    Convert Absolute Magnitude to luminosity (ergs s^-1 Hz^-1).
    '''
    d     = 3.086e+19 # 10pc in cm
    log_L = (M_uv + 48.6)/(-2.5) + np.log10(4*np.pi*d**2) 
    return(np.power(10,log_L))

################ MATH  ########################################################
def within_bounds(values, lower_bounds, upper_bounds):
    '''
    Checks if list of values is within lower and upper bounds (strictly
    within bounds). All three arrays must have same length.
    '''
    is_within = []
    for i in range(len(values)):
        is_within.append((values[i] > lower_bounds[i]) & (values[i] < upper_bounds[i]))
        
    # return True if all elements are within bounds, otherwise False
    if all(is_within)==True:
        return(True)
    return(False)

def invert_function(func, fprime, fprime2, x0_func, y, args):
    '''
    For a function y=f(x), calculate x values for an input set of y values.
    '''
    x = []      
    for val in y:
        def root_func(m,*args):
            return(func(m,*args)-val)
        
        x0_in = x0_func(val, *args) # guess initial value
        
        root = root_scalar(root_func, fprime = fprime, fprime2=fprime2, args = args,
                            method='halley', x0 = x0_in, rtol=1e-6).root
        # if Halley's method doesn't work, try Newton
        if np.isnan(root):
                root = root_scalar(root_func, fprime = fprime, fprime2=fprime2, args = args,
                                    method='newton', x0 = x0_in, rtol=1e-6).root
        x.append(root)
    x = np.array(x)
    return(x)

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

################ COMPUTER PATHS ###############################################
def system_path():
    '''
    Choose path to save files depending on if code is run on cluster of my laptop.
    '''
    path = '/data/p305250/mcmc_runs/'
    if os.path.isdir(path): # if path exists use this one (cluster structure)
        pass 
    else: # else use path for home computer
        path = '/home/chris/Desktop/mcmc_runs/'
    return(path)

def quantity_path(quantity_name):
    '''
    Choose saving path for different physical quantities.
    '''
    if quantity_name == 'mstar':
        path = 'SMF'
    if quantity_name == 'Muv':
        path = 'UVLF'
    return(path)