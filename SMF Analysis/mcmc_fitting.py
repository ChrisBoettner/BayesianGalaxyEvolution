#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 12:13:22 2021

@author: boettner
"""

import numpy as np
from scipy.stats import uniform
import emcee

save_path = '/data/users/boettner/SMF/mcmc_runs/'

## MAIN FUNCTION
def mcmc_fit(smf, hmf, smf_model, mode = 'temp', z = 0):
    '''
    Calculate parameter that match observed SMFs to modelled SMFs using MCMC
    fitting by maximizing the logprobability function which is the product of the
    loglikelihood and logprior.
    IMPORTANT   : When fitting the sn feedback model, only values up th the critical
                  mass are included.
    '''
    # initalize saving for mcmc runs
    if mode == 'temp':
        savefile = None
    else:
        savefile = emcee.backends.HDFBackend(save_path + smf_model.feedback_model.name + str(z) + '.h5')
    
    # define prior as uniform prior (unique uniform prior per parameter in
    # feedback model)
    prior     = lambda params: log_uniform_prior(params, smf_model.feedback_model.name)
    
    # select initial walker positions from uniform distribution
    distribution, param_num = uniform_dist(smf_model.feedback_model.name)
    walker_pos = distribution.rvs(size=[50,param_num], random_state=None)
    nwalkers, ndim = walker_pos.shape
    

    if mode == ('saving' or 'temp'):
        # create MCMC sampler and run MCMC 
        sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                        log_probability, args=(smf_model, smf, prior),
                                        backend=savefile)
        sampler.run_mcmc(walker_pos, 5000, progress=True)
    if mode == 'loading':
        # load from savefile 
        sampler = savefile
    
    # get autocorrelationtime and discard burn-in of mcmc walk 
    tau = sampler.get_autocorr_time()
    flat_samples = sampler.get_chain(discard=5*np.amax(tau).astype(int), flat=True)
    
    # calculate median of parameter from MCMC walks and value of cost function
    # at the calculated parameter
    par  = np.median(flat_samples, axis=0)
    cost = log_likelihood(par, smf_model, smf)/(-0.5)    
    
    # create data for modelled smf (for plotting)
    modelled_smf = hmf.copy()
    modelled_smf[:,1] = smf_model.function(hmf[:,0], par)
    return(par, modelled_smf, cost)


## MCMC HELP FUNCTIONS
def log_likelihood(params, smf_model, smf):
    '''
    Loglikelihood function: This is where the modelling/physics comes in.
    Calculate loglikelihood by assuming that  difference between the (log of the)
    observed and modelled value are distributed according to a Gaussian (without errors). 
    Calculate sum of squared errors (ssr) for this. 
    IMPORTANT :   We minimize the log of the phi_obs and phi_mod, instead of the
                  values themselves. Otherwise the low-mass end would have much higher
                  constribution due to the larger number density.
    '''
    #observed values
    m_obs   = smf[:,0]
    phi_obs = smf[:,1]
    
    # evaluate model function at observed masses
    phi_mod = smf_model.function(m_obs, params)
    
    # if A gets smaller than 0, phi_mod becomes negative and residuals
    # can't be computed correctly, return -inf instead (meaning total probability
    # will be -inf and step will be rejected)
    if params[0]<0:
        return(-np.inf)
    
    # calc residuals (IMPORTANT: log values!)   
    res     = np.log10(phi_obs)- np.log10(phi_mod)
    
    log_L = -0.5 * np.sum(res**2) # Assume Gaussian dist of values
    return(log_L)

def log_uniform_prior(params, feedback_name):
    '''
    The prior for the distribution. For now, just use a uniform prior. Use
    scipy function no create n independent uniform distributions, where n is 
    the number of parameters in the feedback model (with different ranges). 
    '''
    
    # create distribution (which are actually n independent uniform distributions)
    distribution,_  = uniform_dist(feedback_name)
    
    probabilities = distribution.pdf(params) # calc individiual probs for current parameter guesses 
    l = np.prod(probabilities)   # total prob is product of individual probs
        
    if l==0: # if parameter guesses are outside of uniform distribution, reject step
       return(-np.inf)
    return(np.log(l))

def log_probability(params, smf_model, smf_observed, prior):
    '''
    Calculate total probability (which will be maximized by MCMC fitting).
    Total probability is given by likelihood*prior_probability, meaning
    logprobability = loglikelihood+logprior

    '''
    l_prior = prior(params)
    log_L   = log_likelihood(params, smf_model, smf_observed)
    return(l_prior + log_L)


## PRIOR HELP FUNCTIONS
def uniform_dist(feedback_name):
    '''
    Create n independent uniform distributions, where n is the number of parameters
    in the model. The scale parameter gives the range for each distribution of parameter.
    Specifically enforces that all values are larger than 0 (and smaller than their
    respective scale value.)
    '''
    if feedback_name == 'none':
        scale = [1]       # A must be between zero and one
    if feedback_name == 'sn':
        scale = [1,10]    # assume alpha will be smaller than 10 
    if feedback_name == 'both':
        scale = [1,10,10] # assume beta will also be smaller than 10 
    
    multi_uniform = uniform(scale = scale)
    param_num     = len(scale)
    return(multi_uniform, param_num)