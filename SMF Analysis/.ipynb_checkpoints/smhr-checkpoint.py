#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 14:29:52 2021

@author: boettner
"""
from matplotlib import rc_file
rc_file('plots/settings.rc')  # <-- the file containing your settings

import numpy as np
import matplotlib.pyplot as plt

## IMPORT DATA
# import
Davidson    = np.load('data/Davidson2017SMF.npz')
Duncan      = np.load('data/Duncan2014SMF.npz')
Song        = np.load('data/Song2016SMF.npz')
Bhatawdekar = np.load('data/Bhatawdekar2018SMF.npz')
# turn log values to actual values for fitting, and sort to correct redshift
z1  = np.power(10, Davidson['0'])                                            [:,:2]
z2  = np.power(10, Davidson['1'])                                            [:,:2]
z3  = np.power(10, Davidson['2'])                                            [:,:2]
z4  = np.power(10, np.concatenate([Davidson['3'],Duncan['0']]))              [:,:2]
z5  = np.power(10, Duncan['1'])                                              [:,:2]
z6  = np.power(10, np.concatenate([Duncan['2'],Song['0'],Bhatawdekar['0']])) [:,:2]
z7  = np.power(10, np.concatenate([Song['1'],Bhatawdekar['1']]))             [:,:2]
z8  = np.power(10, np.concatenate([Song['2'],Bhatawdekar['2']]))             [:,:2]
z9  = np.power(10, Bhatawdekar['3'])                                         [:,:2]
smfs     = [z1,z2,z3,z4,z5,z6,z7,z8,z9]
hmf_log  = np.load('data/HMF.npz'); hmfs = {str(i):np.power(10,hmf_log[str(i)]) for i in range(20)}

## MAIN FUNCTION
def calculate_SHMR(smfs, hmfs, feedback):
    '''
    Calculate the stellar-to-halo mass relation between theoretical HMF and 
    observed SMF (for all redshifts).
    Three feedback models: 'none', 'sn', 'both'
    
    Uses data to calculate normalization factor, but feedback adjustments are
    pre-set.
    
    Returns:
    norm        :   time-dependent, but mass-independent normalisation factor
    fb_adj      :   feedback adjustment factor (depends on halo mass but not time),
                    depends on chosen model
    ssr         :   sum of squared residuals between scaled HMF and observational data
    '''
    norm = []; fb_adj = []; ssr = []
    
    for i in range(9):
        n, f_a = scaling(smfs[i],hmfs[str(i+1)], fb=feedback) # calc SHMR factors
        norm.append(n); fb_adj.append(f_a)
        
        ## CALCULATE PHI RESIDUALS
        # find indices for closest matching mass (x) values
        # for observed SMF and modelled SMF calculated using SHMR
        smf_obs      = smfs[i][smfs[i][:,1]>1e-6] # cut unreliable values
        smf_mod      = hmfs[str(i+1)].copy()
        smf_mod[:,0] = n*f_a*hmfs[str(i+1)][:,0] 
        
        m_matching_ind = find_closest(smf_obs[:,0], smf_mod[:,0])
        # calculate sum of squared differences between phi (y) values for
        # observed SMF and modelled SMF for same masses (x values)
        res = (smf_obs[:,1]-hmfs[str(i+1)][:,1][m_matching_ind])
        ssr.append(np.sum(res**2))
    return(norm, fb_adj, ssr)

## SUBFUNCTIONS
def scaling(smf, hmf, **kwargs):
    '''
    Calculate the time-dependent normalization factor to match observational 
    data to feedback-adjusted(!) mass function obtained from HMF.
    '''    
    # calculate feedback-related adjustment factors for different halo masses
    # to obtain feedback-adjusted unscaled mass function
    fb_adj      = feedback_adjustment(hmf, **kwargs)
    adj_mf      = hmf.copy()
    adj_mf[:,0] = fb_adj*hmf[:,0]
     
    # use abundance matching to match observed stellar masses to unscaled 
    # (non-normalized) modelled stellar masses to fit normalization factor
    # by matching their number density (phi values)
    masses = abundance_matching(smf, adj_mf, **kwargs)
    
    # calculate time-dependent normalization factor through least squares minimization
    # between observational data and feedback-adjusted masses
    norm = np.linalg.lstsq(masses[:,1,np.newaxis],masses[:,0],rcond=None)[0][0]  
    return(norm, fb_adj)

# find (feedback-adjusted) m values for clostest matching phi values (abundance matching)
def abundance_matching(smf,adj_mf, **kwargs):
    '''
    Match observed stellar masses to unscaled theoretical stellar masses obtained from
    HMF and the feedback model by calculating at which mass their abundances (phi values/
    number density at a given mass)
    
    Abundances (phi values) below 1e-6 are cut off because they cant be measured
    reliably.
    '''
    smf = smf[smf[:,1]>1e-6] # cutoff low values
    
    phi_matching_ind = find_closest(smf[:,1], adj_mf[:,1]) # column indices of closest phi values
    
    masses = np.empty([len(smf),2]) 
    masses[:,0] = smf[:,0]                      # masses from observations
    masses[:,1] = adj_mf[:,0][phi_matching_ind] # phi-matching masses from unscaled model
    return(masses)

# feedback scaling to masses in HMF according to supernova or supernova+black hole models
def feedback_adjustment(hmf, fb = 'none'):
    '''
    Calculate (halo mass-dependent) adjustment factors for matching the HMF to 
    the SMF, caused by feedback mechanisms.
    Three models implemented:
        none    : no feedback adjustment
        sn      : supernova feedback
        both    : supernova and black hole feedback
    '''    
    if fb == 'none':
        fb_factor = np.ones(len(hmf))   
    if fb == 'sn':
        fb_factor = 1/(hmf[:,0]/1e+12)**(-1.5)
    if fb == 'both':
        fb_factor = 1/((hmf[:,0]/1e+12)**(-1.5) + (hmf[:,0]/1e+12)**0.7)
    return(fb_factor)

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

###############################################################################

get_results = lambda fb: calculate_SHMR(smfs, hmfs, fb)

## plotting
plt.close('all')

# plot stellar mass functions
fig, ax = plt.subplots(3,3); ax = ax.flatten()
for i in range(9):        
    ax[i].scatter(smfs[i][:,0],smfs[i][:,1])
    ax[i].set_xscale('log')
    ax[i].set_yscale('log')              
    ax[i].set_xlim([5e+6,2e+12])
    ax[i].set_ylim([1e-6,1e+6])   
fig.supxlabel('$M_*/M_\odot$')
fig.supylabel('$\phi(M_*)$ [cMpc$^{-1}$ dex$^{-1}$]')
for fb in ['none','sn','both']:
    norm, fb_adj, _ = get_results(fb)
    for i in range(9):
        ax[i].plot(norm[i]*fb_adj[i]*hmfs[str(i+1)][:,0], hmfs[str(i+1)][:,1]) 


# plot time factor
fig, ax = plt.subplots()
ax.set_xlabel('$z$')
ax.set_ylabel('time scale factor')
for fb in ['none','sn','both']:
    norm, _, _ = get_results(fb)
    ax.scatter(range(1,10), norm)

# plot change in goodness of fit (compare residuals)
fig, ax = plt.subplots()
ax.set_xlabel('$z$')
ax.set_ylabel('sum of squared residuals')
for fb in ['none','sn','both']:
    _, _, ssr = get_results(fb)
    ax.scatter(range(1,10), ssr)
    ax.set_yscale('log')


# plot scaling as fnct of halo mass
# first choose 'best fit' somehow, then do this
'''
fig, ax = plt.subplots()
for i in range(9):
    ax.plot(hmf[str(i+1)][:,0], time_factor[i]*fb_factor[i], label=str(i+1))
ax.legend()
ax.set_xlim([1e+9,1e+15])
ax.set_ylim([2e-7,5e-1])
ax.set_xlabel('$M_h$')
ax.set_ylabel('scale factor')
ax.set_xscale('log')
ax.set_yscale('log')  
'''

