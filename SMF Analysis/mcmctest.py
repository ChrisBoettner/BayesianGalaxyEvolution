# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 13:19:38 2021

@author: boettner
"""

from matplotlib import rc_file
rc_file('plots/settings.rc')  # <-- the file containing your settings

import numpy as np
import matplotlib.pyplot as plt

from smhr import calculate_SHMR, abundance_matching
from data_processing import group, z_ordered_data

def load():
    # get z=1,2,3,4 for Davidson, z=1,2,3 for Ilbert
    davidson    = np.load('data/Davidson2017SMF.npz'); davidson = {i:davidson[j] for i,j in [['0','2'],['1','4'],['2','6'],['3','8']]}
    ilbert      = np.load('data/Ilbert2013SMF.npz');   ilbert = {i:ilbert[j] for i,j in [['0','2'],['1','4'],['2','6']]}
    duncan      = np.load('data/Duncan2014SMF.npz')      
    song        = np.load('data/Song2016SMF.npz')       
    bhatawdekar = np.load('data/Bhatawdekar2018SMF.npz')
    stefanon    = np.load('data/Stefanon2021SMF.npz')
    hmfs        = np.load('data/HMF.npz'); hmfs = [hmfs[str(i)] for i in range(20)]   
    
    ## TURN DATA INTO GROUP OBJECTS, INCLUDING PLOT PARAMETER
    davidson    = group(davidson,    [1,2,3,4]   ).plot_parameter('black', 'o', 'Davidson2017')
    ilbert      = group(ilbert,      [1,2,3]     ).plot_parameter('black', 'H', 'Ilbert2013')
    duncan      = group(duncan,      [4,5,6,7]   ).plot_parameter('black', 'v', 'Duncan2014')
    song        = group(song,        [6,7,8]     ).plot_parameter('black', 's', 'Song2016')
    bhatawdekar = group(bhatawdekar, [6,7,8,9]   ).plot_parameter('black', '^', 'Bhatawdekar2019')
    stefanon    = group(stefanon,    [6,7,8,9,10]).plot_parameter('black', 'X', 'Stefanon2021')
    groups      = [davidson, ilbert, duncan, song, bhatawdekar, stefanon]
    
    ## DATA SORTED BY REDSHIFT
    smfs = z_ordered_data(groups)
    # undo log for easier fitting
    raise10 = lambda list_log: [10**list_log[i] for i in range(len(list_log))]
    smfs = raise10(smfs)
    hmfs = raise10(hmfs)
    
    ##
    i=0
    smf = smfs[i][smfs[i][:,1]>1e-6] # cut unreliable values
    hmf = hmfs[i+1]  
    return(smf,hmf)

smf, hmf = load()
masses = abundance_matching(smf,hmf)

def feedback_function(fb = 'none'):
    m_c =1e+12
    if fb == 'none':
        f = lambda m, A : A * m     # model
    if fb == 'sn':
        f = lambda m, A, alpha : A * (m/m_c)**alpha * m 
    if fb == 'both':
        f = lambda m, A, alpha, beta : A * 1/((m/m_c)**(-alpha)+(m/m_c)**(beta)) 
    return(f)

def log_likelihood(theta, func, x, y):
    y_mod = func(x,*theta)
    y_obs = y
    log_L = -0.5 * np.sum((y_obs-y_mod)**2)
    return(log_L)

from scipy.stats import multivariate_normal, uniform
# def log_prior(theta, mean, cov):   
#     log_l = multivariate_normal(mean, cov).logpdf(theta)
#     if np.any(theta<0):
#         return(-np.inf)
#     if theta[2]>1:
#         return(-np.inf)
#     return(log_l)

def log_prior(theta, mean, cov):   
    
    p_A       = uniform.pdf(theta[0])
    p_alpha   = uniform(scale=2).pdf(theta[1])
    p_beta    = uniform.pdf(theta[2])
    
    log_l = p_A*p_alpha*p_beta
    if log_l==0:
        return(-np.inf)
    return(log_l)

def log_probability(theta, x, y, func, means, stds):
    l_prior = log_prior(theta, means, stds)
    log_L   = log_likelihood(theta, func, x, y)
    return(l_prior + log_L)

import emcee

mean  = [0.03, 1.3, 0.4]
stds  = np.array([0.007, 0.3, 0.2])
cov   = np.diag(stds**2)

mean_in = [0.03,1.3,0.4]

pos = multivariate_normal(mean = mean_in, cov=10*cov).rvs(size=50, random_state=None)
pos = pos[(np.all(pos>0, axis=1)) & (pos[:,2]<1) ]
print(len(pos))
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(masses[:,1], masses[:,0]/masses[:,1],
                                           feedback_function('both'), mean, cov )
)
sampler.run_mcmc(pos, 50000, progress=True);

tau = sampler.get_autocorr_time()
flat_samples = sampler.get_chain(discard=5*np.amax(tau).astype(int), flat=True)

import corner
fig = corner.corner(flat_samples);