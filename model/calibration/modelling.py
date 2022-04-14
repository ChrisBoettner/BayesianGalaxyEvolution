#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:52:35 2022

@author: chris
"""
import numpy as np

from model.calibration          import mcmc_fitting, leastsq_fitting
from model.calibration.feedback import feedback_model
from model.helper               import mag_to_lum, system_path, quantity_path, \
                                       within_bounds

################ MAIN FUNCTIONS ###############################################
def fit_model(redshifts, log_ndfs, log_hmfs, quantity_name, feedback_name,
              prior_name, fitting_method, saving_mode, name_addon = None,
              **kwargs):
    '''
    Fit the modelled number density functions (ndf) (modelled from HMF + feedback) 
    for specified redshifts (as array) to observations (LFs/SMFs).
    
    feedback_name can either be a single string (then this model is used for all
    redshifts) or a list of strings corresponding to the model used for each redshift.
    
    Choose between 3 different prior model:
        uniform    : assume uniform prior within bounds for each redshift
        successive : use full distribution for parameter from previous 
                     redshift (assuming dependence of parameter)
        marginal : use marginalized distribution for parameter from previous 
                   redshift (assuming independence of parameter) (DEPRECATED)      
                    
    feedback_name can be: 'none', 'stellar', 'stellar_blackhole', 'changing'
                          (which uses 'stellar_blackhole' up to z=4 and 'stellar'
                           afterwards), or a custom list of feedbacks.
    
    saving_mode can be: 'saving', 'loading', 'temp'
    
    If alternative file name is wanted, use name_addon.
    
    Additional parameter can be given to mcmc fit via emcee using **kwargs.
    
    IMPORTANT : Critical mass presetto 10^(12.45) solar masses.
    '''
    
    parameter, distribution, models = [], [], []
    posterior_samp = None
    bounds         = None

    for z in redshifts:
        print('z=' + str(z))
        
        log_ndf = np.copy(log_ndfs[z])
        log_hmf = log_hmfs[z]
        
        log_m_c = 12.45 # critical mass for feedback
        
        # create model object
        # (choose feedback model based on feedback_name input)
        if feedback_name in ['none', 'stellar', 'stellar_blackhole']:
            model           = Model(log_ndf, log_hmf, quantity_name,
                                    feedback_name, log_m_c, z=z)
            model.directory = system_path() + quantity_path(quantity_name) + \
                              '/' + model.feedback_model.name + '/'
        elif feedback_name == 'changing': # standard changing feedback
            if z<=4:
                fb_name = 'stellar_blackhole'
            elif z>4:
                fb_name = 'stellar'
            model           = Model(log_ndf, log_hmf, quantity_name, fb_name,
                                    log_m_c, z=z)
            model.directory = system_path() + quantity_path(quantity_name) +\
                              '/changing/'    
        elif len(feedback_name) == len(redshifts): # custom changing feedback
            model           = Model(log_ndf, log_hmf, quantity_name,
                                    feedback_name[z], log_m_c, z = z) 
            model.directory = system_path() + quantity_path(quantity_name) + \
                              '/changing/'    
        else:
            raise ValueError('feedback_name must either be known string or a \
                              list of strings with the same length as redshifts.')
                              
        model.filename  = prior_name + '_z' + str(model.z)
        
        # if manual modification of saving path is wanted
        if name_addon:
            model.filename = model.filename + ''.join(name_addon)
        
        # create new prior from distribution of previous iteration
        if prior_name == 'uniform':
            prior, bounds = mcmc_fitting.uniform_prior(model, posterior_samp, bounds) 
        elif prior_name == 'marginal':
            raise DeprecationWarning('Marginal prior not really sensible anymore, I think.')
            prior, bounds = mcmc_fitting.dist_from_hist_1d(model, posterior_samp, bounds) 
        elif prior_name == 'successive':
            prior, bounds = mcmc_fitting.dist_from_hist_nd(model, posterior_samp, bounds) 
        else:
            raise ValueError('Prior model not known.')
        
        # fit parameter
        if fitting_method == 'least_squares':
            params, posterior_samp   = leastsq_fitting.lsq_fit(model)
        elif fitting_method == 'mcmc':
            params, posterior_samp   = mcmc_fitting.mcmc_fit(model, prior, saving_mode,
                                                             **kwargs)
        else:
            raise ValueError('fitting_method not known.')

        parameter.append(params)  
        distribution.append(posterior_samp)     
        models.append(model)
            
    return(parameter, distribution, models)

################ MODEL CLASS ##################################################
class Model():
    def __init__(self, log_ndf, log_hmf, quantity_name, feedback_name, log_m_c, z):
        self.quantity_name      = quantity_name
        self.log_observations   = log_ndf
        self.log_hmf_function   = log_hmf
        self.z                  = z
        
        # feedback model
        if self.quantity_name == 'mstar':
            # initial guess for log_A, alpha, beta
            initial_guess = np.array([-2, 1, 0.5])   
            # bounds for log_A, alpha, beta                      
            bounds        = np.array([[-5, 0, 0], [np.log10(2), 4, 0.8]])  
        elif self.quantity_name == 'Muv':
            initial_guess = np.array([18, 1, 0.5]) 
            bounds        = np.array([[13, 0, 0], [20, 4, 0.8]])
        else:
            raise ValueError('quantity_name not known.')
              
        self.feedback_model = feedback_model(feedback_name, log_m_c, initial_guess,
                                             bounds)
        self.directory      = None
        self.filename       = None
        
    def log_ndf(self, log_quantity, params):
        '''
        Calculate (log of) modelled number density function by multiplying HMF 
        function with feedback model derivative.
        Input is log_mstar for SMF and mag for UVLF.
        
        IMPORTANT: If calculated halo mass m_h is bigger than the largest one 
        given in HMFs by Pratika, set to highest available value instead.
        '''
        if self.quantity_name == 'mstar':
            pass
        elif self.quantity_name == 'Muv':
            # convert magnitude to luminosity
            log_quantity = np.log10(mag_to_lum(log_quantity))
        else:
            raise ValueError('quantity_name not known.')
        
        # check that parameters are sensible, otherwise invert function will
        # fail to determine halo masses
        if not within_bounds(params, *self.feedback_model.bounds):
            return(1e+30) # return inf (or huge value) if outside of bounds
        
        # calculate halo masses from stellar masses using model
        log_m_h = self.feedback_model.calculate_log_halo_mass(log_quantity, *params)            
        
        # calculate modelled number density function
        log_hmf       = self.log_hmf_function(log_m_h)
        log_fb_factor = np.log10(self.feedback_model.calculate_dlogquantity_dlogmh(log_m_h,*params)) 
        
        log_ndf = log_hmf - log_fb_factor
        return(log_ndf)