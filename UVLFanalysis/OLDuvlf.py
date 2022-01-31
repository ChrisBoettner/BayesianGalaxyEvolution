#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 12:40:48 2021

@author: boettner
"""
import numpy as np
from scipy.optimize import curve_fit

## MAIN FUNCTIONS
def calculate_LHMR(lfs, hmfs, feedback):
    '''
    Calculate the luminosity-to-halo mass relation between theoretical HMF and 
    observed UVLFs (for all redshifts).
    Three feedback models: 'none', 'sn', 'both'
    
    Uses data to calculate scaling factor and power-law slopes. Critical luminosity is 
    pre-set.
    
    Abundances (phi values) below 1e-6 are cut off because they cant be measured
    reliably.
    
    Returns:
    params      :   set of fitted parameter (A, alpha, beta)
    smf_m       :   modelled LF obtained from scaling HMF
    ssr         :   sum of squared residuals between scaled HMF and observational data
    '''
    
    params = []; sfe = []; sfrf = []
    lf_mod = []; ssr = []
    
    for i in range(len(lfs)):
        lf  = lfs[i][lfs[i][:,1]>1e-6]   # cut unreliable values
        hmf = hmfs[i+1]                  # list starts at z=0 not 1, like smf
        
        ## FIT PARAMETERS
        par = fit(lf, hmf, fb=feedback) 
        params.append(par)
        
        ## CALCULATE MODELLED LF
        f_fb,_,_  = feedback_function(feedback) # get feedback model function   
        # scale halo masses to luminosities based on feedback model and fitted parameters
        lf_m      = hmf.copy()
        lf_m[:,0] = f_fb(lf_m[:,0], *par)
        lf_mod.append(lf_m)
        
        ## calculate SFR and SFRF estimate
        sfe_m, sfrf_m = calculate_sf(par, f_fb, hmf)
        
        sfe.append(sfe_m)
        sfrf.append(sfrf_m)
        
        
        ## CALCULATE PHI RESIDUALS
        # find indices for closest matching mass (x) values
        # for observed LF and modelled LF calculated using LHMR
        m_matching_ind = find_closest(lf[:,0], lf_m[:,0])
        # calculate sum of squared differences between phi (y) values for
        # observed LF and modelled LF for same masses (x values)
        res = (lf[:,1]-lf_m[:,1][m_matching_ind])
        ssr.append(np.sum(res**2))
        
    return(params, sfe, lf_mod, sfrf, ssr)

def calculate_sf(par, f_fb, hmf):
    '''
    Calculate star-formation efficiency (SFE) and star-formation rate (SFR)
    from scaling (fit) parameter A using Kennicutt relation and cosmic baryon 
    fraction, and including feedback.
    L = c_Kennicutt * SFR
      = c_Kennicutt * SFE * m_gas
      = c_Kennicutt * SFE * Omega_b/Omega_m * f_fb(m_h)
      = A * f_fb(m_h)
    '''
    A = par[0]
    
    omega_b     = 0.022/0.7**2  # h = H_0/100 = 0.7
    omega_m     = 0.309
    c_Kennicutt = 7.8e-29       # Chabrier IMF corrected
    
    sfe = omega_m/omega_b*A*c_Kennicutt
    
    sfr = hmf.copy()
    sfr[:,0] = f_fb(sfr[:,0], *par)*c_Kennicutt 
    return(sfe, sfr)

## SUBFUNCTIONS
def fit(lf, hmf, **kwargs):
    '''
    Calculate the fitting parameters for the function that relates luminosity
    to halo masses.
    FEEDBACK MODEL  : PARAMETERS
    none            : A
    sn              : A, alpha
    both            : A, alpha, beta
    '''    
    
    # use abundance matching to match observed stellar mass abundances to halo
    # mass abundances by matching their number density (phi values).
    matched_values   = abundance_matching(lf, hmf, **kwargs)
    
    # get function for feedback model that relates halo masses and stellar masses
    # for fit, as well as initial parameter guess
    func, p0, bounds = feedback_function(**kwargs)
    
    # fit halo masses to observed stellar masses via feedback function to get SHMR
    params,_ = curve_fit(func, matched_values[:,1], matched_values[:,0]//1e+16,
                         p0=p0, maxfev = int(1e+8), bounds=bounds)
    params[0] = params[0]*1e+16
    #print('WARNING: TEMPORARY FIT CODE')
    #params = [np.sum(matched_values[:,1]* matched_values[:,0])/np.sum(matched_values[:,1]**2)]
    return(params)    

# find (feedback-adjusted) m values for clostest matching phi values (abundance matching)
def abundance_matching(lf, hmf, **kwargs):
    '''
    Match observed luminosities to unscaled theoretical stellar masses obtained from
    HMF and the feedback model by calculating at which mass their abundances (phi values/
    number density at a given mass)
    '''
    
    phi_matching_ind = find_closest(lf[:,1], hmf[:,1]) # column indices of closest phi values
    
    matched_values = np.empty([len(lf),2]) 
    matched_values[:,0] = lf[:,0]                      # masses from observations
    matched_values[:,1] = hmf[:,0][phi_matching_ind]   # phi-matching masses from unscaled model
    return(matched_values)

# feedback scaling to masses in HMF according to supernova or supernova+black hole models
def feedback_function(fb = 'none'):
    '''
    Return function that relates luminosity to halo masses according to
    feedback model. The model parameters are left free and can be obtained via 
    fitting. Also return the initial best guesses.
    Three models implemented:
        none    : no feedback adjustment
        sn      : supernova feedback
        both    : supernova and black hole feedback
    '''    
    l_c =np.power(10,10.5)
    
    if fb == 'none':
        f = lambda l, A : A * l           # model
        p0 = np.array([1])                # initital guess
        bounds = (0, [np.inf,])           # fitting bounds
    if fb == 'sn':
        f = lambda l, A, alpha : A * (l/l_c)**alpha * l 
        p0 = np.array([1, 1])
        bounds = (0, [np.inf, np.inf])
    if fb == 'both':
        f = lambda l, A, alpha, beta : A * 1/((l/l_c)**(-alpha)+(l/l_c)**(beta)) * l 
        p0 = np.array([1, 1, 0.7])
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



