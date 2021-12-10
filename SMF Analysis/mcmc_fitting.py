#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 12:13:22 2021

@author: boettner
"""

import numpy as np
from scipy.stats import uniform, rv_histogram
import emcee

import os
from multiprocessing import Pool

## MAIN FUNCTION
def mcmc_fit(smf_model, prior = None, mode = 'temp'):
    '''
    Calculate parameter that match observed SMFs to modelled SMFs using MCMC
    fitting by maximizing the logprobability function which is the product of the
    loglikelihood and logprior.
    IMPORTANT   : When fitting the sn feedback model, only values up th the critical
                  mass are included.
    '''
    # initalize saving for mcmc runs
    save_path = '/data/users/boettner/SMF/mcmc_runs/'
    if mode == 'temp':
        savefile = None
    else:
        filename = save_path + smf_model.feedback_model.name + str(smf_model.z) + 'prior.h5'
        savefile = emcee.backends.HDFBackend(filename)
    
    # select initial walker positions near initial guess
    initial_guess = np.array(smf_model.feedback_model.initial_guess)
    ndim       = len(initial_guess)
    nwalkers   = 50
    walker_pos = initial_guess*(1+0.1*np.random.rand(nwalkers,ndim))

    if (mode == 'saving') or (mode=='temp'):
        if mode == 'saving' and os.path.exists(filename):
            os.remove(filename) # clear file before start writing to it
        # create MCMC sampler and run MCMC
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                            log_probability, args=(smf_model, prior),
                                            backend=savefile, pool=pool)
            sampler.run_mcmc(walker_pos, 3000, progress=True)
    if mode == 'loading':
        # load from savefile 
        sampler = savefile
    
    # get autocorrelationtime and discard burn-in of mcmc walk 
    tau       = np.array(sampler.get_autocorr_time())
    posterior = sampler.get_chain(discard=5*np.amax(tau).astype(int), flat=True)
    
    # calculate median of parameter from MCMC walks and value of cost function
    # at the calculated parameter
    par  = np.median(posterior, axis=0)  
    
    # create data for modelled smf (for plotting)
    modelled_smf = np.copy(smf_model.hmf)
    modelled_smf[:,1] = smf_model.function(smf_model.hmf[:,0], par)
    return(par, modelled_smf, posterior)


## MCMC HELP FUNCTIONS
def log_probability(params, smf_model, prior):
    '''
    Calculate total probability (which will be maximized by MCMC fitting).
    Total probability is given by likelihood*prior_probability, meaning
    logprobability = loglikelihood+logprior

    '''
    
    # check if parameter are within bounds
    if not within_bounds(params, *smf_model.feedback_model.bounds):
        return(-np.inf)
    
    # prior
    l_prior    = log_prior(params, prior) 
    
    # likelihood
    log_L   = log_likelihood(params, smf_model)
    return(l_prior + log_L)

def log_likelihood(params, smf_model):
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
    m_obs   = smf_model.observations[:,0]
    phi_obs = smf_model.observations[:,1]
    
    # evaluate model function at observed masses
    phi_mod = smf_model.function(m_obs, params)
    
    # for bad parameter combinations, inversion of m_star will fail so that
    # m_h can't be calculated and phi_mod can't be estimated. If that is the case
    # reject step
    if np.any(np.isnan(phi_mod)):
        return(-np.inf)
    
    # calc residuals (IMPORTANT: log values!)   
    res     = np.log10(phi_obs)- np.log10(phi_mod)
    
    log_L = -0.5 * np.sum(res**2) # Assume Gaussian dist of values
    return(log_L)

def log_prior(params, prior_hist):
    '''
    Uses the individual prior distributions for each parameter to calculate the
    value of that parameter. Assuming the parameter are independent, the total 
    probability is then the product of the individual probabilities.
    '''
    indiv_prob = [prior_hist[i].pdf(params[i]) for i in range(len(params))]
    total_prob = np.prod(indiv_prob) # assuming independent parameter,
                                     # total prob is product of indiv probs
    if total_prob == 0: 
        return(0) # otherwise log will throw an error
    return(np.log10(total_prob))

## PRIOR FUNCTIONS
def dist_from_hist(smf_model, dist):
    '''
    Creates prior from a sample distribution (derived from a previous mcmc run).
    For now, we assume that all parameter are independent and their marginal 1D
    distributions reflect the full probability for the parameter.
    Returns list of probability distributions for each parameter.
    
    IMPORTANT: If sample distribution is None type, assume uniform distribution. 
    
    '''
    if dist is None:
        return(uniform_prior(smf_model))
    param_num = dist.shape[1]
    dists     = []
    for i in range(param_num):
        lower_bound = smf_model.feedback_model.bounds[0][i]
        upper_bound = smf_model.feedback_model.bounds[1][i]
        hist        = np.histogram(dist[:,i],range=[lower_bound,upper_bound],
                                   density=True, bins = 100)
        dists.append(rv_histogram(hist))
    return(dists)

def uniform_prior(smf_model):
    '''
    Create uniform prior from n independent uniform distributions, where n is 
    the number of parameters in the model.
    Returns list of probability distributions for each parameter. 
    '''
    if smf_model.z>1:
        print("Warning: using uniform prior for z>1")
        
    param_num = len(smf_model.feedback_model.initial_guess)
    dists     = []
    for i in range(param_num):
        lower_bound = smf_model.feedback_model.bounds[0][i]
        upper_bound = smf_model.feedback_model.bounds[1][i]
        dists.append(uniform(loc = lower_bound, scale = upper_bound))
    return(dists)

        
## HELP FUNCTIONS
def within_bounds(values, lower_bounds, upper_bounds):
    '''
    Checks if list of values is within lower and upper bounds (strictly
    within bounds). All three arrays must have same length.
    '''
    is_within = []
    for i in range(len(values)):
        is_within.append((values[i] > lower_bounds[i]) & (values[i] < upper_bounds[i]))
        
    # return True if all elements are within bounds, otherwise False
    if all(is_within)==True:
        return(True)
    return(False)