#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 10:07:33 2022

@author: chris
"""
import numpy as np

from model.analysis.schechter import log_schechter_function, log_schechter_function_mag


def get_quantity_specifics(quantity_name):
    '''
    Function that returns a dictonary with all values and options that are
    different for the different physical quantites.
    In list are:
        model_p0        : Initial conditions for fitting the model.
        model_bounds    : Bounds for model parameter.
        schechter       : callable Schechter function adapted to quantity.
        schechter_p0    : Initial guess for fitting a Schechter function.
        quantity_range  : Representative list over which ndf can be calculated. 
        ndf_xlabel      : Default xlabel for ndf plots.
        ndf_ylabel      : Default ylabel for ndf plots.     
        log_A_label     : Default label for the parameter log_A.
        legend_colums   : Default number of columns for legend of groups.
        
    Parameters
    ----------
    quantity_name : str
        Quantity in question. Must be 'mstar' or 'Muv'.

    Returns
    -------
    options : dict
        Dictonary of options.    
    '''
    options = {}

    if quantity_name == 'mstar':
        options['model_p0']         = np.array([-2, 1, 0.5])
        options['model_bounds']     = np.array([[-5, 0, 0], [np.log10(2), 4, 0.99]])
        options['schechter']        = log_schechter_function
        options['schechter_p0']     = [-4, 10, -1.5]
        options['quantity_range']   = np.linspace(7.32, 12.46, 100)
        options['ndf_xlabel']       = r'log $M_*$ [$M_\odot$]'
        options['ndf_ylabel']       = r'log $\phi(M_*)$ [cMpc$^{-3}$ dex$^{-1}$]'
        options['log_A_label']      = r'$\log A$'
        options['legend_columns']   = 1
    elif quantity_name == 'Muv':
        options['model_p0']         = np.array([18, 1, 0.5])
        options['model_bounds']     = np.array([[13, 0, 0], [20, 4, 0.99]])
        options['schechter']        = log_schechter_function_mag
        options['schechter_p0']     = [-4, -20, -1.5]
        options['quantity_range']   = np.linspace(-23.42, -12.46, 100)  
        options['ndf_xlabel']       = r'$\mathcal{M}_{UV}$'
        options['ndf_ylabel']       = r'log $\phi(\mathcal{M}_{UV})$ [cMpc$^{-3}$ mag$^{-1}$]'
        options['log_A_label']      = r'$\log A$' + '\n' +\
                                      r'[ergs s$^{-1}$ Hz$^{-1}$ $M_\odot^{-1}$]'
        options['legend_columns']   = 3
    else:
        raise ValueError('quantity_name not known.')
    return(options)