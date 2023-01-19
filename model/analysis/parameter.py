#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 13:12:18 2023

@author: chris
"""
import numpy as np
import pandas as pd

from model.helper import make_list, calculate_percentiles
################ MAIN FUNCTIONS ###############################################
def tabulate_parameter(models, precision=2, caption='',
                       **kwargs):
    '''
    Make Latex table for likely parameter ranges. Calculates parameter 
    estimates for all input models at given redshifts. precision parameter
    controls the number of decimal points. Caption can be added using caption
    argument. Further kwargs are passed to 
    calculate_parameter_with_uncertainties.
    '''
    # pre-load dataframe
    parameter_dict = calculate_parameter_with_uncertainties(models, **kwargs)
    redshift = list(parameter_dict.keys())
    
    parameter_num = len(parameter_dict[redshift[0]])
    formatted_DataFrame = pd.DataFrame(index   = redshift,
                                       columns = range(parameter_num))
    
    for p in range(parameter_num):
        for z in redshift:
            param = parameter_dict[z][p]
            if np.isfinite(param).all():
                param = np.around(param, 2)
                # turn into strings and format latex compatible
                string = (r'$' + str(param[0]) 
                               + r'_{-'  + str(param[1]) 
                               + r'}^{+' + str(param[2]) + '}$')
            else:
               # if string contains nans or infs, leave cell empty
               string = ''
               
            formatted_DataFrame.loc[z,p] = string
            
    # add column for redshifts        
    formatted_DataFrame.insert(0, 'z', redshift)
    
    # create header to DataFrame
    header = sum([m.quantity_options['param_y_labels'] for m in models], [])
    # add m_c at appropriate locations
    idx_counter = 0
    for model in models:
        if not model.fixed_m_c:
            header.insert(idx_counter, r'$\log M_\mathrm{c}^'
                                       + model.quantity_options[
                                               'quantity_subscript'] + r'$')
            idx_counter +=1
        idx_counter += model.quantity_options['model_param_num']
    header.insert(0, r'$z$')
    formatted_DataFrame.columns = header
    
    # turn into latex
    column_format = 'r'*(parameter_num+1)
    latex_table = formatted_DataFrame.to_latex(index=False,
                                               escape=False,
                                               column_format=column_format,
                                               caption=caption)
    
    return(latex_table)   

def calculate_parameter_with_uncertainties(models, 
                                           marginalise = True,
                                           redshift=np.arange(11), 
                                           sigma_equiv=2):
    '''
    Calculates parameter estimates for all input models at given redshift. Also
    return their uncertainties. The uncertainties are calculated using
    calculate_percentiles using sigma_equiv (see calculate percentiles for
    details). Returns dictonary where keys span the redshift. Every value
    contains a numpy array where the first columns contains the medians,
    second column the lower uncertainties and third column the upper 
    uncertainties. Uncertainties are calculated as the difference between
    upper/lower percentile and median. If marginalise is True, marginalise over
    nuisance parameter.
    '''
    models   = make_list(models)
    redshift = make_list(redshift) 
    
    parameter_dict = {}
    for z in redshift:
        parameter_at_z = []
        for model in models:
            
            marg_flag = False
            param_num = model.quantity_options['model_param_num']
            if model.fixed_m_c == False:
                param_num +=1
            
            if z in model.redshift:
                distribution   = model.distribution.at_z(z)
                if marginalise and z>=model.quantity_options['marg_z']:
                     distribution = distribution[:,model.quantity_options[
                                                 'marg_keep']]
                     marg_flag = True
                
                # calculate percentiles
                model_par = calculate_percentiles(distribution,
                                                  sigma_equiv=sigma_equiv, 
                                                  mode='uncertainties').T
                
                # deal with changing parameter size, either due to 
                # simpler model at higher z or marginalisation
                if marg_flag or (len(model_par)<param_num):
                    if marg_flag:
                        param_used = model.quantity_options['marg_keep']
                    elif len(model_par)<param_num:
                        param_used = model.physics_model.at_z(z).parameter_used
                    model_par_temp = np.copy(model_par)
                    model_par = np.full([param_num,3], np.nan)
                    model_par[param_used] = model_par_temp
            else:
                # if z not in redshift, pad with nans
                model_par = np.full([param_num, 3], np.nan)
            parameter_at_z.append(model_par)
        parameter_at_z = np.concatenate(parameter_at_z)
        parameter_dict[z] = parameter_at_z
    return(parameter_dict)
    