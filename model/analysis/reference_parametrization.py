#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 12:43:36 2022

@author: chris
"""
import numpy as np
import pandas as pd

from model.helper import make_list, calculate_percentiles, fit_function
from model.quantity_options import get_quantity_range
from model.calibration.leastsq_fitting import calculate_weights
from model.analysis.calculations import calculate_best_fit_ndf

################ MAIN FUNCTIONS ###############################################
def tabulate_reference_parameter(ModelResult, redshift, num=100):
    '''
    Make Latex table for likely reference function parameter ranges 
    (16th-84th percentile).    
    '''
    if ModelResult.distribution.is_None():
        raise AttributeError(
            'distributions have not been calculated.')
    redshift = make_list(redshift)
    #breakpoint()
    # calculate distribution    
    parameter_distribution = get_reference_parameter_distribution(ModelResult,
                                                                  redshift,
                                                                  num=num)
    # calculate percentiles (likely parameter ranges)
    lower_bounds, upper_bounds = {}, {}
    for z in redshift:
        _, lower_bounds[z], upper_bounds[z] = calculate_percentiles(
            parameter_distribution[z])
    
    # pre-load dataframe
    parameter_num = ModelResult.quantity_options['reference_param_num']
    formatted_DataFrame = pd.DataFrame(index   = redshift,
                                       columns = range(parameter_num))
    
    for p in range(parameter_num):
        for z in redshift:
            pres = 2    # rounding precision
            range_lower = lower_bounds[z][p]
            range_upper = upper_bounds[z][p]     
            
            # get value of exponent
            exponent    = np.format_float_scientific(range_upper, precision = pres)
            _, exponent = exponent.split('e')
            exponent = int(exponent)
            
            range_l     = np.round(range_lower, pres)
            range_u     = np.round(range_upper, pres)
            
            # if p == 1:  # should be the log_obs_*, looks nicer like this
            #     exponent     += -1
            
            # # round to representable values
            # range_u = np.around(range_upper/10**exponent, pres)
            # range_l = np.around(range_lower/10**exponent, pres)
            # if range_l == 0:
            #     range_l = np.around(range_lower/exponent, pres+1) 
            # if range_l == 0:
            #     range_l = np.abs(range_l) # bc otherwise there is sometime a -0.0 problem
                
            # turn into strings and format latex compatible
            
            l_str = str(range_l)
            u_str = str(range_u)   
            if len(l_str.split('.')[1]) < pres:
                l_str = l_str + '0'
            if len(u_str.split('.')[1]) < pres:
                u_str = u_str + '0' 
            if (range_u<0) and (range_l<=0):
                string = r'$-\left[' + l_str[1:] + ' \text{-} ' + u_str[1:]\
                         + '\right]$'  
            elif (range_u>0) and (range_l>=0):
                string = r'$\left[' + l_str + ' \text{-} ' + u_str + '\right]$' 
            else:
                breakpoint()
                raise ValueError('Whoops, I was to lazy to implement that case')
            #if exponent != 0:
            #    string = string[:-1] + r' \cdot 10^{' + str(exponent) + r'}$'
                     
            formatted_DataFrame.loc[z,p] = string
            
    # add column for redshifts        
    formatted_DataFrame.insert(0, 'z', redshift)
    # add header to DataFrame
    formatted_DataFrame.columns = ModelResult.quantity_options['reference_table_header']
    # add caption to table
    caption = 'Likely ' + ModelResult.quantity_options['reference_function_name'] +\
              ' parameter ranges for ' + ModelResult.quantity_options['ndf_name'] +\
              ' given by 16th and 84th percentile.'
    # turn into latex
    latex_table = formatted_DataFrame.to_latex(index = False,
                                               escape=False,
                                               column_format = 'r'*(parameter_num+1),
                                               caption = caption)
    
    return(latex_table)   


def calculate_best_fit_reference_parameter(ModelResult, redshift):
    '''
    Calculate best fit reference parameter for best fit model.
    
    IMPORTANT: only fit ndfs up to log_phi > cutoff to stay consistent with 
               calibration.
    '''
    if ModelResult.parameter.is_None():
        raise AttributeError(
            'best fit parameter have not been calculated.')
        
    redshift = make_list(redshift)
    
    reference, p0 = ModelResult.quantity_options['reference_function'],\
                    ModelResult.quantity_options['reference_p0']
    bounds        = ModelResult.quantity_options['reference_p0_bounds']
    
    reference_parameter = {}
    for z in redshift:
        quantity_range = get_quantity_range(z, ModelResult)
        ndf = calculate_best_fit_ndf(ModelResult, z, quantity_range)[z]

        cutoff = ModelResult.quantity_options['cutoff']
        ndf = ndf[ndf[:,1]>cutoff]
        
        reference_parameter[z] = fit_function(reference,
                                              ndf,
                                              p0 = p0,
                                              bounds = bounds)
    return(reference_parameter)
        

def calculate_reference_parameter_from_data(ModelResult, redshift):
    '''
    Calculate reference parameter from ndf observational data. Include 
    uncertainties if possible.
    '''
    
    redshift = make_list(redshift)
    function, p0 = ModelResult.quantity_options['reference_function'],\
                   ModelResult.quantity_options['reference_p0']
    bounds        = ModelResult.quantity_options['reference_p0_bounds']
    
    reference_parameter = {}
    for z in redshift:
        try: # include uncertainties if given
            weights  = calculate_weights(ModelResult, z=z)
        except:
            weights  = None

        reference_parameter[z] = fit_function(function,
                                              ModelResult.log_ndfs.at_z(z)[:,:2],
                                              p0 = p0,
                                              uncertainties = 1/weights,
                                              bounds = bounds)
    return(reference_parameter)

def get_reference_function_sample(ModelResult, redshift, num=100,
                                  quantity_range=None):
    '''
    Calculate sample of reference functions over typical quantity_range
    by fitting functions to ndf samples.
    '''
    
    redshift = make_list(redshift)
    function, log_quantity = ModelResult.quantity_options['reference_function'],\
                             ModelResult.quantity_options['quantity_range']

    reference_parameter = get_reference_parameter_distribution(ModelResult,
                                                               redshift,
                                                               num=num)

    sample = {}
    for z in redshift:
        sample_at_z = []
        for params in reference_parameter[z]:
            phi = function(log_quantity, *params)
            sample_at_z.append(np.array([log_quantity, phi]).T)
        sample[z] = sample_at_z
    return(sample)


def get_reference_parameter_distribution(ModelResult, redshift, num=100):
    '''
    Calculate distribution of reference function parameter for model at a given
    redshift. Do this by calculating num model ndfs and fitting the reference
    functions (Schechter/power law) to each one. 
    Returns dictonary of form {redshift:distribution}.t
    
    IMPORTANT: only fit ndfs up to log_phi > cutoff to stay consistent with 
               calibration.
    '''
    if ModelResult.distribution.is_None():
        raise AttributeError(
            'distributions have not been calculated.')
    
    redshift = make_list(redshift)
    function, p0 = ModelResult.quantity_options['reference_function'],\
                   ModelResult.quantity_options['reference_p0']
    bounds        = ModelResult.quantity_options['reference_p0_bounds']

    # calculate distributions
    reference_parameter_distribution = {}
    for z in redshift:
        parameter_at_z = []
        quantity_range = get_quantity_range(z, ModelResult)
        ndf_sample = ModelResult.get_ndf_sample(z, num=num,
                                                quantity_range=quantity_range)

        for ndf in ndf_sample:
            cutoff = ModelResult.quantity_options['cutoff']
            ndf = ndf[ndf[:,1]>cutoff]
            
            params = fit_function(function, ndf, p0=p0, bounds = bounds)
            parameter_at_z.append(params)

        reference_parameter_distribution[z] = np.array(parameter_at_z)
    return(reference_parameter_distribution)