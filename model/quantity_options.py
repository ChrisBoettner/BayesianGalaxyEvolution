#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 10:07:33 2022

@author: chris
"""
import numpy as np

from model.analysis.schechter import log_schechter_function, log_schechter_function_mag
from model.analysis.power_law import log_double_power_law


def get_quantity_specifics(quantity_name):
    '''
    Function that returns a dictonary with all values and options that are
    different for the different physical quantites.
    In dictonary are:
        quantity_name           : Name of observable quantity.
        quantity_name_tex       : Latex formatted name.
        ndf_name                : Name of associated number density function.
        cutoff                  : Lowest allowed value for phi in ndf.
        log_m_c                 : Value for critical mass 
                                  (obtained from fitting at z=0).
        feedback_change_z       : Redshift at which feedback model changes.

        model_param_num         : (Maximum) number of model parameter.
        model_p0                : Initial conditions for fitting the model.
        model_bounds            : Bounds for model parameter.
        fitting_space           : Choose if quantity is fit in 'linear' or 
                                  'log' space.
        relative_weights        : Choose if relative or absolute uncertainties
                                  are used as weights (True/False).     
        extrapolation_z         : Range of redshifts used for extrapolation to
                                  higher redshift.

        reference_function_name : Name of reference function used.
        reference_function      : Callable reference function adapted to 
                                  quantity (Schechter function/double power law).
        reference_p0            : Initial guess for fitting reference function.
        reference_p0_bounds     : Bounds for reference function parameter.
        reference_param_num     : Number of parameters in reference function.

        quantity_range          : Representative list over which ndf can be 
                                  calculated. 
        subplot_grid            : Layout of subplots.
        ndf_xlabel              : Default xlabel for ndf plots.
        ndf_ylabel              : Default ylabel for ndf plots.     
        ndf_y_axis_limit        : y-axis limit for ndf plots.
        param_y_labels          : Labels for the model parameter.
        ndf_legend_pos          : Location of (first) legend in ndf plots.
        marker_alpha            : Transparency for data points. 
        legend_colums           : Default number of columns for legend of groups.
        reference_table_header  : Header for Latex table of Schechter parameters. 

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
        options['quantity_name'] = 'mstar'
        options['quantity_name_tex'] = r'$M_\star$'
        options['ndf_name'] = 'SMF'
        options['physics_models'] = ['none', 'stellar',
                                     'stellar_blackhole', 'changing']
        options['cutoff'] = -9
        options['log_m_c'] = {0: 12.1,1: 12.25,2: 12.41}
        options['feedback_change_z'] = 3
        # MODEL
        options['model_param_num'] = 3
        options['model_p0'] = np.array([-2, 1, 0.5])
        options['model_bounds'] = np.array([[-5, 0, 0],
                                            [np.log10(0.1), 4, 0.96]])
        options['fitting_space'] = 'log'
        options['relative_weights'] = True
        options['systematic_uncertainties'] = True
        options['extrapolation_z'] = np.array([5,6,7,8,9,10])
        # REFERENCE FUNCTION
        options['reference_function_name'] = 'Schechter'
        options['reference_function'] = log_schechter_function
        options['reference_p0'] = [-4, 10, -1.5]
        options['reference_p0_bounds'] = (- np.inf, np.inf)
        options['reference_param_num'] = 3
        # PLOTS AND TABLES
        options['quantity_range'] = np.linspace(7.02, 12.48, 100)
        options['subplot_grid'] = (4, 3)
        options['ndf_xlabel'] = r'log $M_\star$ [$M_\odot$]'
        options['ndf_ylabel'] = r'log $\phi(M_\star)$ [cMpc$^{-3}$ dex$^{-1}$]'
        options['ndf_y_axis_limit'] = [-6, 1]
        options['param_y_labels'] = [r'$\log A$',
                                     r'$\gamma$',
                                     r'$\delta$']
        options['ndf_legend_pos'] = 8
        options['marker_alpha'] = 0.4
        options['legend_columns'] = 2
        options['reference_table_header'] = [r'$z$',
                                             r'$\log \phi_*$ [cMpc$^{-1}$ dex$^{-1}$]',
                                             r'$\log M_\star^\mathrm{c}$ [$M_\odot$]',
                                             r'$\alpha$']

    elif quantity_name == 'Muv':
        # GENERAL
        options['quantity_name'] = 'Muv'
        options['quantity_name_tex'] = r'$\mathcal{M}_\mathrm{UV}$'
        options['ndf_name'] = 'UVLF'
        options['physics_models'] = ['none', 'stellar',
                                     'stellar_blackhole', 'changing']
        options['cutoff'] = -9
        options['log_m_c'] = {0: 11.73,1: 11.51, 2: 11.4, 3: 11.47, 4: 11.15}
        options['feedback_change_z'] = 5
        # MODEL
        options['model_param_num'] = 3
        options['model_p0'] = np.array([18, 1, 0.5])
        options['model_bounds'] = np.array([[13, 0.2, 0],
                                            [24.5, 4, 0.99]])
        options['fitting_space'] = 'log'
        options['relative_weights'] = True
        options['systematic_uncertainties'] = True
        options['extrapolation_z'] = np.array([5,6,7,8,9,10])
        # REFERENCE FUNCTION
        options['reference_function_name'] = 'Schechter'
        options['reference_function'] = log_schechter_function_mag
        options['reference_p0'] = [-4, -20, -1.5]
        options['reference_p0_bounds'] = (- np.inf, np.inf)
        options['reference_param_num'] = 3
        # PLOTS AND TABLES
        options['quantity_range'] = np.linspace(-22.21, -14.46, 100)
        options['subplot_grid'] = (4, 3)
        options['ndf_xlabel'] = r'$\mathcal{M}_\mathrm{UV}$'
        options['ndf_ylabel'] = r'log $\phi(\mathcal{M}_{UV})$ [cMpc$^{-3}$ dex$^{-1}$]'
        options['ndf_y_axis_limit'] = [-6, 3]
        options['param_y_labels'] = [r'$\log A$',
                                     r'$\gamma$',
                                     r'$\delta$']
        options['ndf_legend_pos'] = 8
        options['marker_alpha'] = 0.4
        options['legend_columns'] = 3
        options['reference_table_header'] = [r'$z$',
                                             r'$\log \phi_*$ [cMpc$^{-1}$ dex$^{-1}$]',
                                             r'$\mathcal{M}_\mathrm{UV}^\mathrm{c}$',
                                             r'$\alpha$']
    elif quantity_name == 'mbh':
        # GENERAL
        options['quantity_name'] = 'mbh'
        options['quantity_name_tex'] = r'$M_\bullet$'
        options['ndf_name'] = 'BHMF'
        options['physics_models'] = ['none', 'quasar']
        options['cutoff'] = -13
        options['log_m_c'] = {0: 12.1, 1: 12.25, 2: 12.41}
        options['feedback_change_z'] = 3
        # MODE
        options['model_param_num'] = 2
        options['model_p0'] = np.array([5, 1])
        # upper limit for A parameter in model chosen to be minimum observed
        # black hole mass
        options['model_bounds'] = np.array([[  0,  0],
                                            [ 15, 20]])
        options['fitting_space'] = 'log'
        options['relative_weights'] = True
        options['systematic_uncertainties'] = False
        options['extrapolation_z'] = np.array([1,2,3,4,5])
        # REFERENCE FUNCTION
        options['reference_function_name'] = 'Double power law'
        options['reference_function'] = log_double_power_law
        options['reference_p0'] = [-4.5, 12, 0.2, 6]
        options['reference_p0_bounds'] = [[-np.inf, 0, 0, 0],
                                          [np.inf, np.inf, np.inf, np.inf]]
        options['reference_param_num'] = 4
        # PLOTS AND TABLES
        options['quantity_range'] = np.linspace(6.87, 10.73, 100)
        options['subplot_grid'] = (3, 2)
        options['ndf_xlabel'] = r'log $M_\bullet$ [$M_\odot$]'
        options['ndf_ylabel'] = r'log $\phi(M_\bullet)$ [cMpc$^{-3}$ dex$^{-1}$]'
        options['ndf_y_axis_limit'] = [-15, 3]
        options['param_y_labels'] = [r'$\log B$',
                                     r'$\eta$']
        options['ndf_legend_pos'] = 3
        options['marker_alpha'] = 0.4
        options['legend_columns'] = 1
        options['reference_table_header'] = [r'$z$',
                                             r'$\log \phi_*$ [cMpc$^{-1}$ dex$^{-1}$]',
                                             r'$\log M_\star^\mathrm{c}$ [$M_\odot$]',
                                             r'$\gamma_1$',
                                             r'$\gamma_2$']    
    
    elif quantity_name == 'Lbol':
        # GENERAL
        options['quantity_name'] = 'Lbol'
        options['quantity_name_tex'] = r'$L_\mathrm{bol}$'
        options['ndf_name'] = 'QLF'
        options['physics_models'] = ['none', 'eddington',
                                     'eddington_free_ERDF',
                                     'eddington_changing']
        options['cutoff'] = -12
        options['log_m_c'] = {0: 12.1,1: 12.25,2: 12.41}
        options['feedback_change_z'] = 3
        # MODE
        options['model_param_num'] = 4
        options['model_p0'] = np.array([39,   5, -2, 1.9])
        options['model_bounds'] = np.array([[38,  0,  -4,   1],
                                            [51,  8,  -1,  10]])
        options['fitting_space'] = 'log'
        options['relative_weights'] = True
        options['systematic_uncertainties'] = False
        options['extrapolation_z'] = np.array([2,3,4,5,6,7])
        # REFERENCE FUNCTION
        options['reference_function_name'] = 'Double power law'
        options['reference_function'] = log_double_power_law
        options['reference_p0'] = [-4.5, 12, 0.2, 6]
        options['reference_p0_bounds'] = [[-np.inf, 0, 0, 0],
                                          [np.inf, np.inf, np.inf, np.inf]]
        options['reference_param_num'] = 4
        # PLOTS AND TABLES
        options['quantity_range'] = np.linspace(41.49, 50.91, 100)
        options['subplot_grid'] = (4, 2)
        options['ndf_xlabel'] = r'log $L_\mathrm{bol}$ [erg s$^{-1}$]'
        options['ndf_ylabel'] = r'log $\phi(L_\mathrm{bol})$ [cMpc$^{-3}$ dex$^{-1}$]'
        options['ndf_y_axis_limit'] = [-15, 3]
        options['param_y_labels'] = [r'$\log C$',
                                     r'$\theta$',
                                     r'$ \log \lambda_\mathrm{c}$',
                                     r'$\rho$']
        options['ndf_legend_pos'] = 3
        options['marker_alpha'] = 0.1
        options['legend_columns'] = 1
        options['reference_table_header'] = [r'$z$',
                                             r'$\log \phi_*$ [cMpc$^{-1}$ dex$^{-1}$]',
                                             r'$L_\mathrm{bol}^\mathrm{c}$ [$L_\odot$]',
                                             r'$\gamma_1$',
                                             r'$\gamma_2$']

    else:
        raise NameError('quantity_name not known.')
    return(options)


def update_bounds(model, parameter, z=None):
    '''
    Calculate and update bounds for parameter where bounds depend on each other.
    (For lbol model.)
    ''' 
    if z is None:
        z = model._z
    
    bounds = np.copy(model.physics_model.at_z(z).bounds).astype(float)
    # for lbol model, the parameter rho (slope of ERDF) must be larger
    # than the 1+(slope of the HMF/eta) to converge.
    if model.physics_model.at_z(z).name == 'eddington_free_ERDF':
        eta = parameter[1]
        rho_bound = 1 + model.hmf_slope/eta
        bounds[0, 3] = rho_bound
            
    # if physics model is eddington, the ERDF is fixed, so we adjust the 
    # bounds of the eta instead of the ERDF slope (rho)          
    elif model.physics_model.at_z(model._z).name == 'eddington':
        rho         = model.physics_model.at_z(model._z).parameter[1]
        eta_bound   = np.abs(model.hmf_slope)/(rho-1)
        bounds[0,1] = eta_bound     
    return(bounds)
