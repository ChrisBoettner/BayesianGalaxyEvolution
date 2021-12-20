#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 15:55:20 2021

@author: boettner
"""
import numpy as np
from scipy.optimize import curve_fit

## MAIN FUNCTION
def calculate_SHMR(smfs, hmfs, feedback):
    '''
    Calculate the stellar-to-halo mass relation between theoretical HMF and 
    observed SMF (for all redshifts).
    Three feedback models: 'none', 'sn', 'both'
    
    Uses data to calculate scaling factor and power-law slopes. Critical mass is 
    pre-set.
    
    Abundances (phi values) below 1e-6 are cut off because they cant be measured
    reliably.
    
    Returns:
    params      :   set of fitted parameter (A, alpha, beta)
    smf_m       :   modelled SMF obtained from scaling HMF
    ssr         :   sum of squared residuals between scaled HMF and observational data
    '''
    
    params = []; smf_mod = []; ssr = []
    
    for i in range(len(smfs)):
        smf = smfs[i][smfs[i][:,1]>1e-6] # cut unreliable values
        hmf = hmfs[i+1]                  # list starts at z=0 not 1, like smf
        
        ## FIT PARAMETERS
        par = fit(smf, hmf, fb=feedback) 
        params.append(par)
        
        ## CALCULATE MODELLED SMF
        f_fb,_,_   = feedback_function(feedback) # get feedback model function      
        # scale halo masses to stellar masses based on feedback model and fitted parameters
        smf_m      = hmf.copy()
        smf_m[:,0] = f_fb(smf_m[:,0], *par)
        smf_mod.append(smf_m)
        
        ## CALCULATE PHI RESIDUALS
        # find indices for closest matching mass (x) values
        # for observed SMF and modelled SMF calculated using SHMR
        m_matching_ind = find_closest(smf[:,0], smf_m[:,0])
        # calculate sum of squared differences between phi (y) values for
        # observed SMF and modelled SMF for same masses (x values)
        res = (smf[:,1]-smf_m[:,1][m_matching_ind])
        ssr.append(np.sum(res**2))
        
    return(params, smf_mod, ssr)

## SUBFUNCTIONS
def fit(smf, hmf, **kwargs):
    '''
    Calculate the fitting parameters for the function that relates stellar masses
    to halo masses.
    FEEDBACK MODEL  : PARAMETERS
    none            : A
    sn              : A, alpha
    both            : A, alpha, beta
    '''    
    
    # use abundance matching to match observed stellar mass abundances to halo
    # mass abundances by matching their number density (phi values).
    masses   = abundance_matching(smf, hmf, **kwargs)
    
    # get function for feedback model that relates halo masses and stellar masses
    # for fit, as well as initial parameter guess
    func, p0, bounds = feedback_function(**kwargs)
    
    # fit halo masses to observed stellar masses via feedback function to get SHMR
    params,_ = curve_fit(func, masses[:,1], masses[:,0], p0=p0, maxfev = int(1e+8),
                         bounds=bounds)
    return(params)    

# find (feedback-adjusted) m values for clostest matching phi values (abundance matching)
def abundance_matching(smf, hmf, **kwargs):
    '''
    Match observed stellar masses to unscaled theoretical stellar masses obtained from
    HMF and the feedback model by calculating at which mass their abundances (phi values/
    number density at a given mass)
    '''
    
    phi_matching_ind = find_closest(smf[:,1], hmf[:,1]) # column indices of closest phi values
    
    masses = np.empty([len(smf),2]) 
    masses[:,0] = smf[:,0]                      # masses from observations
    masses[:,1] = hmf[:,0][phi_matching_ind]    # phi-matching masses from unscaled model
    return(masses)

# feedback scaling to masses in HMF according to supernova or supernova+black hole models
def feedback_function(fb = 'none'):
    '''
    Return function that relates stellar masses to halo masses according to
    feedback model. The model parameters are left free and can be obtained via 
    fitting. Also return the initial best guesses.
    Three models implemented:
        none    : no feedback adjustment
        sn      : supernova feedback
        both    : supernova and black hole feedback
    '''    
    m_c =1e+12
    
    if fb == 'none':
        f = lambda m, A : A * m     # model
        p0 = np.array([0.02])       # initital guess
        bounds = (0, [np.inf])
    if fb == 'sn':
        f = lambda m, A, alpha : A * (m/m_c)**alpha * m 
        p0 = np.array([0.02, 1])
        bounds = (0, [np.inf, np.inf])
    if fb == 'both':
        f = lambda m, A, alpha, beta : A * 1/((m/m_c)**(-alpha)+(m/m_c)**(beta)) * m 
        p0 = np.array([0.02, 1, 0.6])
        bounds = (0, [np.inf, np.inf, 1])
    return(f, p0, bounds)

## HELP FUNCTIONS
def find_closest(data,reference):
    '''
    Find indices of closest matching values in reference array compared to
    data (observations) array
    '''
    ind = []
    for i in range(len(data)):
        diff = np.abs(reference-data[i]) 
        ind.append(np.argmin(diff))
    return(ind)



