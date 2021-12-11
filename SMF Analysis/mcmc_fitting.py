#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 12:13:22 2021

@author: boettner
"""

import numpy as np
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
    if mode == 'temp':
        savefile = None
    else:
        save_path = '/data/users/boettner/SMF/mcmc_runs/'
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
                                            backend=savefile, pool = pool)
            sampler.run_mcmc(walker_pos, 50000, progress=True)
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
    
    # PRIOR
    # check if parameter are within bounds
    if not within_bounds(params, *smf_model.feedback_model.bounds):
        return(-np.inf)
    # prior prob
    l_prior    = log_prior(params, prior) 
    
    # LIKELIHOOD
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
    hists = prior_hist[0]
    edges = prior_hist[1]
    
    # find indices for bins in histogram that the params belong to
    ind = []
    for i in range(len(params)):
        ind.append(np.argwhere(params[i]<edges[i])[0][0]-1)
    
    # if probabilities are assumed independent: get probability for each param
    # from corresponding histogram and then multiply all together to get total
    # probability
    if len(hists)>1:
        indiv_prob = [hists[i][ind[i]] for i in range(len(params))]
        total_prob = np.prod(indiv_prob)
        
    # if probabilities are not assumed independent: get total probability form
    # n-dimensional histogram    
    else: 
        total_prob = hists[0][tuple(ind)]
                                         
    if total_prob == 0: 
        return(-np.inf) # otherwise log will throw an error
    return(np.log10(total_prob))

## PRIOR FUNCTIONS
def dist_from_hist_nd(smf_model, dist):
    '''
    Create n-dimensional histogram from a sample distribution 
    (derived from a previous mcmc run).
    Returns normalized histogram (probabilities) and edges of histogram, both as
    lists for compatiblity with other prior functions.
    
    IMPORTANT: If sample distribution is None type, assume uniform distribution. 
    '''
    if dist is None:
        return(uniform_prior(smf_model, dist))
    if dist.shape[1] == 1:
        return(dist_from_hist_1d(smf_model, dist))
    
    bounds= list(zip(*smf_model.feedback_model.bounds))
    
    hist_nd, edges = np.histogramdd(dist, bins=100, range=bounds, density=True)
    return([hist_nd], edges)
    

def dist_from_hist_1d(smf_model, dist):
    '''
    Create n histograms from a sample distribution (derived from a previous 
    mcmc run). Here we assume that all parameter are independent and their 
    marginal 1D distributions reflect the full probability for the parameter.
    Returns normalized histograms (probabilities) and edges of histograms, both
    as lists.
    
    IMPORTANT: If sample distribution is None type, assume uniform distribution. 
    
    '''
    if dist is None:
        return(uniform_prior(smf_model, dist))
    hists     = []
    edges     = []
    for i in range(dist.shape[1]):
        lower_bound  = smf_model.feedback_model.bounds[0][i]
        upper_bound  = smf_model.feedback_model.bounds[1][i]
        hist, edge   = np.histogram(dist[:,i], range=(lower_bound,upper_bound),
                                    density=True, bins = 100)
        hists.append(hist)
        edges.append(edge)
    return(hists, edges)

def uniform_prior(smf_model, dist):
    '''
    Create n histograms that match n independent uniform prior.
    Returns normalized histograms (probabilities) and edges of histograms, both
    as lists.
    '''
    if smf_model.z>1:
        print("Warning: using uniform prior for z>1")
        
    param_num = len(smf_model.feedback_model.initial_guess)
    hists     = []
    edges     = []
    for i in range(param_num):
        lower_bound = smf_model.feedback_model.bounds[0][i]
        upper_bound = smf_model.feedback_model.bounds[1][i]
        hist        = np.array([1/(upper_bound-lower_bound)])
        edge        = np.array([lower_bound, upper_bound])
        hists.append(hist)
        edges.append(edge)
    return(hists, edges)

        
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