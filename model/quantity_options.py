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
    In dictonary are:
        quantity_name           : Name of observable quantity.
        ndf_name                : Name of associated number density function.
        log_m_c                 : Value for critical mass 
                                  (obtained from fitting at z=0).
        feedback_change_z       : Redshift at which feedback changes from
                                  stellar+blackhole to stellar.
        
        model_param_num         : (Maximum) number of model parameter.
        model_p0                : Initial conditions for fitting the model.
        model_bounds            : Bounds for model parameter.
        
        schechter               : Callable Schechter function adapted to quantity.
        schechter_p0            : Initial guess for fitting a Schechter function.
        
        quantity_range          : Representative list over which ndf can be 
                                  calculated. 
        subplot_grid            : Layout of subplots.
        ndf_xlabel              : Default xlabel for ndf plots.
        ndf_ylabel              : Default ylabel for ndf plots.     
        ndf_y_axis_limit        : y-axis limit for ndf plots.
        param_y_labels          : Labels for the model parameter.
        legend_colums           : Default number of columns for legend of groups.
        schechter_table_header  : Header for Latex table of Schechter parameters. 
        
    Parameters
    ----------
    quantity_name : str
        Quantity in question. Must be 'mstar', 'Muv' or 'Lbol'.

    Returns
    -------
    options : dict
        Dictonary of options.    
    '''
    options = {}

    if quantity_name == 'mstar':
        # GENERAL
        options['quantity_name']            = 'mstar'
        options['ndf_name']                 = 'SMF'
        options['log_m_c']                  = 12.3
        options['feedback_change_z']        = 5
        # MODEL
        options['model_param_num']          = 3
        options['model_p0']                 = np.array([-2, 1, 0.5])
        options['model_bounds']             = np.array([[-5, 0, 0],
                                                        [np.log10(2), 4, 0.96]])
        #SCHECHTER
        options['schechter']                = log_schechter_function
        options['schechter_p0']             = [-4, 10, -1.5]
        # PLOTS AND TABLES
        options['quantity_range']           = np.linspace(7.32, 12.16, 100)
        options['subplot_grid']             = (4,3)
        options['ndf_xlabel']               = r'log $M_*$ [$M_\odot$]'
        options['ndf_ylabel']               = r'log $\phi(M_*)$ [cMpc$^{-3}$ dex$^{-1}$]'
        options['ndf_y_axis_limit']         = [-6, 3]
        options['param_y_labels']           = [r'$\log A$',
                                               r'$\gamma$',
                                               r'$\delta$']
        options['legend_columns']           = 1
        options['schechter_table_header']   = [r'$z$',
                                               r'$\log \phi_*$ [cMpc$^{-1}$ dex$^{-1}$]',
                                               r'$\log M_*^\mathrm{c}$ [$M_\odot$]',
                                               r'$\alpha$']
        
    elif quantity_name == 'Muv':
        # GENERAL
        options['quantity_name']            = 'Muv'
        options['ndf_name']                 = 'UVLF'
        options['log_m_c']                  = 11.5
        options['feedback_change_z']        = 8
        # MODEL
        options['model_param_num']          = 3
        options['model_p0']                 = np.array([18, 1, 0.5])
        options['model_bounds']             = np.array([[13, 0, 0], 
                                                        [20, 4, 0.99]])
        # SCHECHTER
        options['schechter']                = log_schechter_function_mag
        options['schechter_p0']             = [-4, -20, -1.5]
        # PLOTS AND TABLES
        options['quantity_range']           = np.linspace(-23.42, -12.46, 100)  
        options['subplot_grid']             = (4,3)
        options['ndf_xlabel']               = r'$\mathcal{M}_\mathrm{UV}$'
        options['ndf_ylabel']               = r'log $\phi(\mathcal{M}_{UV})$ [cMpc$^{-3}$ mag$^{-1}$]'
        options['ndf_y_axis_limit']         = [-6, 3]
        options['param_y_labels']           = [r'$\log A$' + '\n'\
                                               r'[mag$^{-1}$ $M_\odot^{-1}$]',
                                               r'$\gamma$',
                                               r'$\delta$']
        options['legend_columns']           = 3
        options['schechter_table_header']   = [r'$z$',
                                               r'$\log \phi_*$ [cMpc$^{-1}$ mag$^{-1}$]',
                                               r'$\mathcal{M}_\mathrm{UV}^\mathrm{c}$',
                                               r'$\alpha$']
    elif quantity_name == 'Lbol':
        # GENERAL
        options['quantity_name']            = 'Lbol'
        options['ndf_name']                 = 'QLF'
        options['log_m_c']                  = 11.5
        options['feedback_change_z']        = np.nan
        # MODE
        options['model_param_num']          = 2
        options['model_p0']                 = np.array([35, 3])
        # lower limit for A parameter in model chosen to be minimum observed
        # luminosity
        options['model_bounds']             = np.array([[30, 0], 
                                                        [39.1552, 10]])
        # SCHECHTER        
        options['schechter']                = np.nan
        options['schechter_p0']             = np.nan
        # PLOTS AND TABLES
        options['quantity_range']           = np.linspace(39.65, 50.71, 100)  
        options['subplot_grid']             = (4,2)
        options['ndf_xlabel']               = r'$L_\mathrm{bol}$'
        options['ndf_ylabel']               = r'log $\phi(L_\mathrm{bol})$ [cMpc$^{-3}$ dex$^{-1}$]'
        options['ndf_y_axis_limit']         = [-15, 3]
        options['param_y_labels']           = [r'$\log A$' + '\n'\
                                               r'[ergs s$^{-1}$ Hz$^{-1}$ $M_\odot^{-1}$]',
                                               r'$\epsilon$']
        options['legend_columns']           = 1
        options['schechter_table_header']   = [r'$z$',
                                               r'$\log \phi_*$ [cMpc$^{-1}$ dex$^{-1}$]',
                                               r'$L_\mathrm{bol}^\mathrm{c}$',
                                               r'$\alpha$']
        
    else:
        raise ValueError('quantity_name not known.')
    return(options)
