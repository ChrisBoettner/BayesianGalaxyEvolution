#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 12:13:22 2021

@author: boettner
"""

import numpy as np
import emcee

from help_functions import geometric_median

import os
from multiprocessing import Pool

## MAIN FUNCTION
def mcmc_fit(lf_model, prior, prior_name, mode = 'temp'):
    '''
    Calculate parameter that match observed LFs to modelled LFs using MCMC
    fitting by maximizing the logprobability function which is the product of the
    loglikelihood and logprior.
    '''
    # initalize saving for mcmc runs
    if mode == 'temp':
        savefile = None
    else:
        # use correct file path depending on system
        save_path = '/data/p305250/UVLF/mcmc_runs/' + lf_model.directory +'/'
        if os.path.isdir(save_path): # if path exists use this one (cluster structure)
            pass 
        else: # else use path for home computer
            save_path = '/home/chris/Desktop/mcmc_runs/UVLF/' + lf_model.directory +'/'            
        filename = save_path + lf_model.filename +'.h5'
        savefile = emcee.backends.HDFBackend(filename)
    
    # select initial walker positions near initial guess
    initial_guess = np.array(lf_model.feedback_model.initial_guess)
    ndim       = len(initial_guess)
    nwalkers   = 50
    walker_pos = initial_guess*(1+0.1*np.random.rand(nwalkers,ndim))
    
    # make prior a global variable so it doesn"t have to be called in 
    # log_probability explicitly. This helps with parallization and makes it
    # a lot faster
    # see https://emcee.readthedocs.io/en/stable/tutorials/parallel/
    global prior_global
    prior_global = prior

    if (mode == 'saving') or (mode=='temp'):
        if mode == 'saving' and os.path.exists(filename):
            os.remove(filename) # clear file before start writing to it
        # create MCMC sampler and run MCMC
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                            log_probability, args=(lf_model,),
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
    #par  = np.median(posterior,axis=0)
    par  = geometric_median(posterior)  
    
    return(par, posterior)

## MCMC HELP FUNCTIONS
def log_probability(params, lf_model):
    '''
    Calculate total probability (which will be maximized by MCMC fitting).
    Total probability is given by likelihood*prior_probability, meaning
    logprobability = loglikelihood+logprior
    '''
    
    # PRIOR
    # check if parameter are within bounds
    if not within_bounds(params, *lf_model.feedback_model.bounds):
        return(-np.inf)
    # prior prob
    l_prior    = log_prior(params, prior_global) 
    
    # LIKELIHOOD
    log_L   = log_likelihood(params, lf_model)  
    return(l_prior + log_L)

def log_likelihood(params, lf_model):
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
    l_obs   = lf_model.observations[:,0]
    phi_obs = lf_model.observations[:,1]
    
    # evaluate model function at observed masses
    phi_mod = lf_model.function(l_obs, params)
    
    # for bad parameter combinations, inversion of m_star will fail so that
    # m_h can't be calculated and phi_mod can't be estimated. If that is the case
    # reject step
    if not np.all(np.isfinite(phi_mod)):
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
    #import pdb; pdb.set_trace()
    # find indices for bins in histogram that the params belong to
    bin_widths = [e[1]-e[0] for e in edges]
    ind = [int(params[i]/bin_widths[i]) for i in range(len(params))]
    
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
def dist_from_hist_nd(lf_model, dist, dist_bounds):
    '''
    Create n-dimensional histogram from a sample distribution 
    (derived from a previous mcmc run).
    Returns normalized histogram (probabilities) and edges of histogram, both as
    lists for compatiblity with other prior functions. Also returns new set of 
    bounds for the histogram, to be used on the next iteration.
    
    IMPORTANT: If number of parameters of model are fewer than the columns in
               dist, we marginalise over the later columns of dist.
    
    IMPORTANT: If sample distribution is None type, assume uniform distribution. 
    '''
    if dist is None:
        return(uniform_prior(lf_model, dist, dist_bounds))
    if dist.shape[1] == 1:
        return(dist_from_hist_1d(lf_model, dist, dist_bounds))
    
    param_num = len(lf_model.feedback_model.initial_guess)
    
    dist_bounds_new = list(zip(*lf_model.feedback_model.bounds))
    if dist_bounds is None: # if bounds are not provided, use the ones from model
        dist_bounds    = dist_bounds_new

    # create histogram    
    hist_nd, edges = np.histogramdd(dist, bins=100, range=dist_bounds)
    # make empty spots have 0.1% of actual prob, so that these are not completely ignored
    hist_nd[hist_nd == 0] = 0.001*np.sum(hist_nd)/np.sum(hist_nd == 0)
    # normalize
    hist_nd = hist_nd/np.sum(hist_nd)
    
    # marginalise over parameters if new model has fewer parameter than given
    # in dist (assume later columns in model are to be marginalised over)
    marg_variables = hist_nd.ndim - param_num
    if marg_variables>0:
        for n in range(marg_variables):
            hist_nd = np.sum(hist_nd, axis = -1)
        edges = edges[:-marg_variables]
    return([[hist_nd], edges], dist_bounds_new)
    

def dist_from_hist_1d(lf_model, dist, dist_bounds):
    '''
    Create n histograms from a sample distribution (derived from a previous 
    mcmc run). Here we assume that all parameter are independent and their 
    marginal 1D distributions reflect the full probability for the parameter.
    Returns normalized histograms (probabilities) and edges of histograms, both
    as lists. Also returns new set of bounds for the histogram, to be used on 
    the next iteration.
    
    IMPORTANT: If number of parameters of model are fewer than the columns in
               dist, we assume the additional columns of dist are to be ignored.
    
    IMPORTANT: If sample distribution is None type, assume uniform distribution. 
    
    '''
    if dist is None:
        return(uniform_prior(lf_model, dist, dist_bounds))
    hists     = []
    edges     = []
    
    param_num = len(lf_model.feedback_model.initial_guess)
    dist_bounds_new = list(zip(*lf_model.feedback_model.bounds))
    if dist_bounds is None: # if bounds are not provided, use the ones from model
        dist_bounds    = dist_bounds_new
    
    # if dist.shape[1] > param_num, ignore the later columns of dist
    for i in range(param_num):
        lower_bound = dist_bounds[i][0]
        upper_bound = dist_bounds[i][1]
        hist, edge   = np.histogram(dist[:,i], range=(lower_bound,upper_bound),
                                    bins = 100)
        # make empty spots have 0.1% of actual prob, so that these are not completely ignored
        hist[hist == 0] = 0.001*np.sum(hist)/np.sum(hist == 0)
        # normalize
        hist = hist/np.sum(hist)
        
        hists.append(hist)
        edges.append(edge)
    return([hists, edges], dist_bounds)

def uniform_prior(lf_model, dist, dist_bounds):
    '''
    Create n histograms that match n independent uniform prior.
    Returns normalized histograms (probabilities) and edges of histograms, both
    as lists. Also returns new set of bounds for the histogram, to be used on 
    the next iteration.
    '''
    hists     = []
    edges     = []
    param_num = len(lf_model.feedback_model.initial_guess)
    dist_bounds_new = list(zip(*lf_model.feedback_model.bounds))
    if dist_bounds is None: # if bounds are not provided, use the ones from model
        dist_bounds    = dist_bounds_new

    for i in range(param_num):
        lower_bound = dist_bounds[i][0]
        upper_bound = dist_bounds[i][1]
        hist        = np.array([1/(upper_bound-lower_bound)])
        edge        = np.array([lower_bound, upper_bound])
        hists.append(hist)
        edges.append(edge)
    return([hists, edges], dist_bounds)

        
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
