#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 12:28:34 2021

@author: boettner
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from scipy.special import hyp2f1
import leastsq_fitting
import mcmc_fitting

#ALTERNATIVE IN THE SENSE THAT dm_*/dm_h gets fixed, nott m_*(m_h)

## MAIN FUNCTION
## MAIN FUNCTION
def fit_SMF_model(smfs, hmfs, feedback_name, fitting_method = 'least_squares',
                  mode = 'loading', m_crit=1e+11):
    '''
    Fit the modelled SMF (modelled from HMF + feedback) to the observed SMF (for
    all redshifts).
    Three feedback models: 'none', 'sn', 'both'
    Critical mass is pre-set to 1e+11
    
    Abundances (phi values) below 1e-6 are cut off because they cant be measured
    reliably.
    
    IMPORTANT   : When fitting the sn feedback model, only values up th the critical
                  mass are included.
    
    Returns:
    params       :   set of fitted parameter (A, alpha, beta)
    modelled_smf :   modelled SMF obtained from scaling HMF
    cost          :  value of cost function between modelled SMF and observational data
    '''
    
    # choose fitting method
    if fitting_method == 'least_squares':
        fit = leastsq_fitting.lsq_fit
    elif fitting_method == 'mcmc':
        def fit(smf, hmf, smf_model, z = 0): # choose saving/loading mode
            return(mcmc_fitting.mcmc_fit(smf, hmf, smf_model, mode, z))
    
    parameter = []; modelled_smf = []; cost = []
    for i in range(len(smfs)):
        smf = np.copy(smfs[i])[smfs[i][:,1]>1e-6] # cut unreliable values
        hmf = np.copy(hmfs[i+1])                  # list starts at z=0 not 1, like smf
        
        # convert units from 1 solar mass to 10^10 solar masses for numerical stability
        base_unit = 1e+10
        smf[:,0]    = smf[:,0]/base_unit
        hmf[:,0]    = hmf[:,0]/base_unit
        if i==0:    
            m_crit  = m_crit/base_unit
        # if just sn feedback is fitted, ignore values above high-mass knee, because 
        # they screw up the fit
        if feedback_name == 'sn':
            smf = smf[smf[:,0]<m_crit]  
        
        smf_model = smf_model_class(hmf, feedback_name, m_crit) 
        # fit and fit parameter
        params, mod_smf, c = fit(smf, hmf, smf_model, z=i+1) 
        parameter.append(params)
        mod_smf[:,0] = mod_smf[:,0]*base_unit # return to 1 solar mass unit   
        modelled_smf.append(mod_smf)
        cost.append(c)      
    return(parameter, modelled_smf, cost)

## CREATE THE SMF MODEL
class smf_model_class():
    def __init__(self, hmf, feedback_name, m_crit):
        self.hmf_function   = interp1d(*hmf.T) # turn hmf data into evaluable function (using linear interpolation)
        self.feedback_model = feedback_model(feedback_name, m_crit) # choose feedback model function
        
    def function(self, m_star, params):
        '''
        Create SMF model function by multiplying HMF function with feedback model 
        derivative.
        IMPORTANT: If calculated halo mass m_h is bigger than the largest one 
        given in HMFs by Pratika, set to highest available value instead. (Should
        not really be a problem, since this only happens at z=2, where the value 
        is only minimally bigger)
        '''
        
        # check that parameters are sensible, otherwise invert function will
        # fail to determine halo masses
        if np.any(params<0) or (params[0]==0):
            return(1e+10) # basically makes cost func infinite
        
        # calculate halo masses from stellar masses using model
        m_h = self.feedback_model.calculate_m_h(m_star, *params)   
        
        # if halo masses in HMFs is exceeded, set to this value
        m_h_max = 89125 # maximum mass in HMFs in 10^10 solar masses
        m_h[m_h>m_h_max] = m_h_max
        
        return(self.hmf_function(m_h) / self.feedback_model.calculate_dlogmstar_dlogmh(m_h,*params))

# DEFINE THE FEEDBACK MODELS
def feedback_model(feedback_name, m_crit):
    '''
    Return feedback model that relates SMF and HMF, including model function, 
    model name and initial guess for fitting, that related SMF and HMF. 
    The model function parameters are left free and can be obtained via fitting.
    Three models implemented:
        none    : no feedback adjustment
        sn      : supernova feedback
        both    : supernova and black hole feedback
    '''
    if feedback_name == 'none':  
        model = no_feedback(feedback_name, m_crit)
    if feedback_name == 'sn':  
        model = supernova_feedback(feedback_name, m_crit)
    if feedback_name == 'both':  
        model = supernova_blackhole_feedback(feedback_name, m_crit)
    return(model)

# the feedback models with all necessary parameter and functional equations
# see overleaf notes where these equations come from
class no_feedback():
    def __init__(self, feedback_name, m_crit):
        self.name          = feedback_name
        self.m_c           = m_crit
        self.initial_guess = [0.01]
    def calculate_m_star(self, m_h, A):
        return(A*m_h)
    def calculate_m_h(self, m_star, A):
        return(m_star/A)
    def calculate_dlogmstar_dlogmh(self, m_h, A):
        return(1)        
    
class supernova_feedback():
    def __init__(self, feedback_name, m_crit):
        self.name          = feedback_name
        self.m_c           = m_crit
        self.initial_guess = [0.01, 1] 
    def calculate_m_star(self, m_h, A, alpha):
        return( A/(alpha+1) * (m_h/self.m_c)**alpha * m_h)
    def calculate_m_h(self, m_star, A, alpha):
        return((m_star* (alpha+1)/A* self.m_c**alpha)**(1/(alpha+1)))
    def calculate_dlogmstar_dlogmh(self, m_h, A, alpha):
        return(alpha+1)

class supernova_blackhole_feedback():
    def __init__(self, feedback_name, m_crit):
        self.name          = feedback_name
        self.m_c           = m_crit
        self.initial_guess = [0.01, 1, 0.3]
    def calculate_m_star(self, m_h, A, alpha, beta):
        eta = (alpha+1)/(alpha+beta)
        alphabet = alpha+beta
        x = (m_h/self.m_c)
        return(A/(alpha+1) * self.m_c * x**(1+alpha) * hyp2f1(1, eta, eta+1, -x**alphabet))
    def calculate_m_h(self, m_star, A, alpha, beta):
        m_star_func = lambda m_halo : self.calculate_m_star(m_halo, A, alpha, beta)
        gradient    = lambda m_halo : self._calculate_dmstar_dmh(m_halo, A, alpha, beta)
        m_h = invert_function(m_star_func, gradient, m_star) 
        return(m_h)
    def calculate_dlogmstar_dlogmh(self, m_h, A, alpha, beta):
        eta = (alpha+1)/(alpha+beta)
        alphabet = alpha+beta
        x = (m_h/self.m_c)
        return((1+alpha)/(1+x**alphabet)*1/hyp2f1(1, eta, eta+1, -x**alphabet))
    def _calculate_dmstar_dmh(self, m_h,A,alpha,beta): # just used to calc inverse
        return(A * 1/((m_h/self.m_c)**(-alpha)+(m_h/self.m_c)**(beta)))


        
## HELP FUNCTIONS
def invert_function(func, fprime, y):
    '''
    For a function y=f(x), calculate x values for an input set of y values.

    '''
    x = []
    
    for val in y:
        root_func = lambda m: func(m) - val
        x.append(root_scalar(root_func, fprime = fprime, x0 = val*100, rtol=1e-8).root)
    return(np.array(x))

            
        