#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:52:35 2022

@author: chris
"""
import numpy as np

from model.calibration import mcmc_fitting, leastsq_fitting
from model.helper import mag_to_lum, system_path, quantity_path, invert_function, within_bounds

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
        m_c     = np.power(10,12.45) 
        
        # create model object
        # (choose feedback model based on feedback_name input)
        if feedback_name in ['none', 'stellar', 'stellar_blackhole']:
            model           = Model(log_ndf, log_hmf, quantity_name, feedback_name, m_c, z=z)
            model.directory = system_path() + quantity_path(quantity_name) + '/' + model.feedback_model.name + '/'
     
        elif feedback_name == 'changing': # standard changing feedback
            if z<=4:
                fb_name = 'stellar_blackhole'
            elif z>4:
                fb_name = 'stellar'
            model           = Model(log_ndf, log_hmf, quantity_name, fb_name, m_c, z=z)
            model.directory = system_path() + quantity_path(quantity_name) + '/changing/'
            
        elif len(feedback_name) == len(redshifts): # custom changing feedback
            model           = Model(log_ndf, log_hmf, quantity_name, feedback_name[z], m_c,
                                    z=z) 
            model.directory = system_path() + quantity_path(quantity_name) + '/changing/'
            
        else:
            raise ValueError('feedback_name must either be a string or a \
                              list of strings with the same length as redshifts.')
                              
        model.filename  = prior_name + '_z' + str(model.z)
        
        # if manual modification of saving path is wanted
        if name_addon:
            model.filename = model.filename + name_addon
        
        # create new prior from distribution of previous iteration
        if prior_name == 'uniform':
            prior, bounds = mcmc_fitting.uniform_prior(model, posterior_samp, bounds) 
        elif prior_name == 'marginal':
            raise DeprecationWarning('Marginal prior not really sensible anymore, I think.')
            prior, bounds = mcmc_fitting.dist_from_hist_1d(model, posterior_samp, bounds) 
        elif prior_name == 'successive':
            prior, bounds = mcmc_fitting.dist_from_hist_nd(model, posterior_samp, bounds) 
        
        # fit parameter
        if fitting_method == 'least_squares':
            params, posterior_samp   = leastsq_fitting.lsq_fit(model)
        elif fitting_method == 'mcmc':
            params, posterior_samp   = mcmc_fitting.mcmc_fit(model, prior, saving_mode,
                                                             **kwargs)

        parameter.append(params)  
        distribution.append(posterior_samp)     
        models.append(model)
            
    return(parameter, distribution, models)

################ MODEL CLASS ##################################################
class Model():
    def __init__(self, log_ndf, log_hmf, quantity_name, feedback_name, m_c, z):
        self.quantity_name      = quantity_name
        self.log_observations   = log_ndf
        self.log_hmf_function   = log_hmf
        self.z                  = z
        
        # feedback model
        if self.quantity_name == 'mstar':
            initial_guess = np.array([-2, 1, 0.5])                         # initial guess for log_A, alpha, beta
            bounds        = np.array([[-5, 0, 0], [np.log10(2), 4, 0.8]]) # bounds for log_A, alpha, beta
        if self.quantity_name == 'Muv':
            initial_guess = np.array([18, 1, 0.5]) 
            bounds        = np.array([[13, 0, 0], [20, 4, 0.8]])
        self.feedback_model = feedback_model(feedback_name, m_c, initial_guess,
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
        
        if self.quantity_name == 'Muv':
            # convert magnitude to luminosity
            log_quantity = np.log10(mag_to_lum(log_quantity))
        
        # check that parameters are sensible, otherwise invert function will
        # fail to determine halo masses
        if not within_bounds(params, *self.feedback_model.bounds):
            return(1e+30) # return inf (or huge value) if outside of bounds
        
        # calculate halo masses from stellar masses using model
        log_m_h = self.feedback_model.calculate_log_halo_mass(log_quantity, *params)            
        
        # calculate modelled number density function
        log_hmf       = self.log_hmf_function(log_m_h)
        log_fb_factor = np.log10(self.feedback_model.calculate_dlogobservable_dlogmh(log_m_h,*params)) 
        
        log_ndf = log_hmf - log_fb_factor
        return(log_ndf)


################ FEEDBACK MODEL ###############################################    
def feedback_model(feedback_name, m_c, initial_guess, bounds):
    '''
    Return feedback model that relates SMF and HMF, including model function, 
    model name, initial guess and physical parameter bounds for fitting,
    that related SMF and HMF. 
    The model function parameters are left free and can be obtained via fitting.
    Three models implemented:
        none    : no feedback adjustment
        sn      : supernova feedback
        both    : supernova and black hole feedback
    '''
    if feedback_name == 'none':  
        model = NoFeedback(feedback_name, m_c, initial_guess[:1], bounds[:,:1])
    if feedback_name == 'stellar': 
        model = StellarFeedback(feedback_name, m_c, initial_guess[:2], bounds[:,:2])
    if feedback_name == 'stellar_blackhole':  
        model = StellarBlackholeFeedback(feedback_name, m_c, initial_guess, bounds)
    return(model)

# the feedback models with all necessary parameter and functional equations
# see overleaf notes where these equations come from
class NoFeedback():
    def __init__(self, feedback_name, m_c, initial_guess, bounds):
        self.name          = feedback_name
        self.m_c           = m_c
        self.initial_guess = initial_guess
        self.bounds        = bounds
    def calculate_log_observable(self, log_m_h, log_A):
        return(np.log10(np.power(10,log_A)/2*np.power(10, log_m_h)))
    def calculate_log_halo_mass(self, log_observable, log_A):
        return(np.log10(2*np.power(10, log_observable)/np.power(10,log_A)))
    def calculate_dlogobservable_dlogmh(self, log_m_h, log_A):
        return(1)        

class StellarFeedback():
    def __init__(self, feedback_name, m_c, initial_guess, bounds):
        self.name          = feedback_name
        self.m_c           = m_c      
        self.initial_guess = initial_guess
        self.bounds        = bounds
    def calculate_log_observable(self, log_m_h, log_A, alpha):
        if np.isnan(log_m_h).any() or np.any((log_m_h-np.log10(self.m_c))>18):
            return(np.nan)
        ratio = np.power(10, log_m_h - np.log10(self.m_c))
        sn = ratio**(-alpha)
        obs = np.power(10,log_A) * np.power(10, log_m_h)/(1 + sn)
        return(np.log10(obs))    
    def calculate_log_halo_mass(self, log_observable, log_A, alpha):
        log_m_h = invert_function(func    = self.calculate_log_observable,
                                  fprime  = self.calculate_dlogobservable_dlogmh,
                                  fprime2 = self.calculate_d2logobservable_dlogmh2,
                                  x0_func = self._initial_guess, 
                                  y       = log_observable,
                                  args    = (log_A, alpha))
        return(log_m_h)
    def calculate_dlogobservable_dlogmh(self, log_m_h, log_A, alpha):
        if np.isnan(log_m_h).any() or np.any((log_m_h-np.log10(self.m_c))>18):
            return(np.nan)
        ratio = np.power(10, log_m_h - np.log10(self.m_c))
        sn = ratio**(-alpha)
        return(1 - (-alpha*sn)/(1 + sn))
    def calculate_d2logobservable_dlogmh2(self, log_m_h, log_A, alpha): 
        if np.isnan(log_m_h).any() or np.any((log_m_h-np.log10(self.m_c))>18):
            return(np.nan)
        ratio = np.power(10, log_m_h - np.log10(self.m_c))
        sn = ratio**(-alpha)
        denom = (1 + sn)
        num_one = alpha**2*sn
        num_two =(-alpha*sn)**2
        return(-np.log(10)*(num_one/denom+num_two/denom**2))
    def _initial_guess(self, log_observable, log_A, alpha):
        # guess initial value for inverting function by using high and low mass
        # end approximation
        m_t          = np.power(10,log_A)/2*self.m_c  # turnover mass where dominating feedback changes
        trans_regime = 20            # rough estimate for transient regime where both are important
        if np.power(10, log_observable) < m_t/trans_regime:
            x0 = np.power((self.m_c)**alpha/np.power(10,log_A)*np.power(10, log_observable),1/(1+alpha))
        elif np.power(10, log_observable) > m_t*trans_regime:
            x0 = np.power(10,log_A)*self.m_c  
        else:
            x0 = np.power(10, log_observable)*2/np.power(10,log_A)
        return(np.log10(x0))

class StellarBlackholeFeedback():
    def __init__(self, feedback_name, m_c, initial_guess, bounds):
        self.name          = feedback_name
        self.m_c           = m_c
        self.initial_guess = initial_guess
        self.bounds        = bounds
    def calculate_log_observable(self, log_m_h, log_A, alpha, beta):
        if np.isnan(log_m_h).any() or np.any((log_m_h-np.log10(self.m_c))>18):
            return(np.nan)
        ratio = np.power(10, log_m_h - np.log10(self.m_c))
        sn = ratio**(-alpha)
        bh = ratio**beta
        obs = np.power(10,log_A) * np.power(10, log_m_h)/(sn + bh)
        return(np.log10(obs))    
    def calculate_log_halo_mass(self, log_observable, log_A, alpha, beta):
        log_m_h = invert_function(func    = self.calculate_log_observable,
                                  fprime  = self.calculate_dlogobservable_dlogmh,
                                  fprime2 = self.calculate_d2logobservable_dlogmh2,
                                  x0_func = self._initial_guess, 
                                  y       = log_observable,
                                  args    = (log_A, alpha, beta))
        return(log_m_h)
    def calculate_dlogobservable_dlogmh(self, log_m_h, log_A, alpha, beta):
        if np.isnan(log_m_h).any() or np.any((log_m_h-np.log10(self.m_c))>18):
            return(np.nan)
        ratio = np.power(10, log_m_h - np.log10(self.m_c))
        sn = ratio**(-alpha)
        bh = ratio**beta
        return(1 - (-alpha*sn + beta * bh)/(sn + bh))
    def calculate_d2logobservable_dlogmh2(self, log_m_h, log_A, alpha, beta): 
        if np.isnan(log_m_h).any() or np.any((log_m_h-np.log10(self.m_c))>18):
            return(np.nan)
        ratio = np.power(10, log_m_h - np.log10(self.m_c))
        sn = ratio**(-alpha)
        bh = ratio**beta
        denom = (sn + bh)
        num_one = (alpha**2*sn + beta**2 * bh)
        num_two =(-alpha*sn + beta * bh)**2
        return(-np.log(10)*(num_one/denom+num_two/denom**2))
    def _initial_guess(self, log_observable, log_A, alpha, beta):
        # guess initial value for inverting function by using high and low mass
        # end approximation
        m_t          = np.power(10,log_A)/2*self.m_c  # turnover mass where dominating feedback changes
        trans_regime = 20            # rough estimate for transient regime where both are important
        if np.power(10, log_observable) < m_t/trans_regime:
            x0 = np.power((self.m_c)**alpha/np.power(10,log_A)*np.power(10, log_observable),1/(1+alpha))
        elif np.power(10, log_observable) > m_t*trans_regime:
            x0 = np.power((self.m_c)**(-beta)/np.power(10,log_A)*np.power(10, log_observable),1/(1-beta))   
        else:
            x0 = np.power(10, log_observable)*2/np.power(10,log_A)
        return(np.log10(x0))

