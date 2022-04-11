#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:23:57 2022

@author: chris
"""

import numpy as np
import emcee

import os
from multiprocessing import Pool
from pathlib import Path

from scipy.optimize import dual_annealing, minimize, brute

from model.calibration.leastsq_fitting import cost_function
from model.helper import within_bounds



################ MAIN FUNCTIONS ###############################################
def mcmc_fit(model, prior, saving_mode,
             chain_length = 10000, num_walkers = 250, 
             autocorr_discard = True, parameter_calc = True,
             progress = True):
    '''
    Calculate parameter that match observed numbder density function (LF/SMF) 
    to modelled functionss using MCMC fitting by maximizing the logprobability 
    function which is the product of the loglikelihood and logprior.
    
    chain_length is the length of the Markov chains.
    num_walkers is the number of walkers for mcmc sampling.
    If autocorr_discard is True, calculate and discard the burn-in part of the 
    chain.
    If parameter_calc is True, calculate best-fit parameter (MAP estimator).
    If progress is True, show progress bar of mcmc.
    '''
    # initalize saving for mcmc runs
    if saving_mode == 'temp':
        savefile = None
    else:
        save_path = model.directory       
        Path(save_path).mkdir(parents=True, exist_ok=True) # create dir if it doesn't exist
        filename  = save_path + model.filename +'.h5'
        savefile  = emcee.backends.HDFBackend(filename)
    
    # select initial walker positions near initial guess
    initial_guess = np.array(model.feedback_model.initial_guess)
    ndim       = len(initial_guess)
    nwalkers   = num_walkers
    walker_pos = initial_guess*(1+0.1*np.random.rand(nwalkers,ndim))
    
    # make prior a global variable so it doesn"t have to be called in 
    # log_probability explicitly. This helps with parallization and makes it
    # a lot faster
    # see https://emcee.readthedocs.io/en/stable/tutorials/parallel/
    global prior_global
    prior_global = prior
    if (saving_mode == 'saving') or (saving_mode=='temp'):
        if saving_mode == 'saving' and os.path.exists(filename):
            os.remove(filename) # clear file before start writing to it
        # create MCMC sampler and run MCMC
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                            log_probability, args=(model,),
                                            backend=savefile, pool = pool)
            sampler.run_mcmc(walker_pos, chain_length, progress=progress)
    if saving_mode == 'loading':
        # load from savefile 
        sampler = savefile
    
    # get autocorrelationtime and discard burn-in of mcmc walk 
    if autocorr_discard:
        tau = np.array(sampler.get_autocorr_time())
    else:
        tau = 0
    posterior_samp = sampler.get_chain(discard=5*np.amax(tau).astype(int), flat=True)
    
    # calculate best fit parameter
    if parameter_calc:
        bounds = list(zip(np.percentile(posterior_samp, 16, axis = 0),
                           np.percentile(posterior_samp, 84, axis = 0)))
        params = calculate_MAP_estimator(prior_global, model, method = 'annealing',
                                         bounds = bounds,
                                         x0 = np.median(posterior_samp,axis = 0))
    else:
        params = None

    return(params, posterior_samp)

################ PROBABILITY FUNCTIONS ########################################
def log_probability(params, model):
    '''
    Calculate total probability (which will be maximized by MCMC fitting).
    Total probability is given by likelihood*prior_probability, meaning
    logprobability = loglikelihood+logprior
    '''
    
    # check if parameter are within bounds
    if not within_bounds(params, *model.feedback_model.bounds):
        return(-np.inf)
    
    # PRIOR
    l_prior    = log_prior(params, prior_global) 
    # LIKELIHOOD
    l_likelihood   = log_likelihood(params, model)
    return(l_prior + l_likelihood)

def log_likelihood(params, model):
    '''
    Sum of squares of residuals. See leastsq_fitting.cost_function for more in-
    formation
    '''
    return(-cost_function(params, model))

def log_prior(params, prior_hist):
    '''
    Uses the individual prior distributions for each parameter to calculate the
    value of that parameter. Assuming the parameter are independent, the total 
    probability is then the product of the individual probabilities.
    '''
    hists = prior_hist[0]
    edges = prior_hist[1]

    # find indices for bins in histogram that the params belong to
    bin_widths = [e[1]-e[0] for e in edges]
    ind = np.array([int(params[i]/bin_widths[i]) for i in range(len(params))])
    ind[ind < 0]     = 0 
    ind_upper_lim    = (len(edges[0])-2) # highest ind in hist
    ind[ind > ind_upper_lim] = ind_upper_lim
    
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

################ PRIOR FUNCTIONS ##############################################
def dist_from_hist_nd(model, dist, dist_bounds):
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
        return(uniform_prior(model, dist, dist_bounds))
    if dist.shape[1] == 1:
        return(dist_from_hist_1d(model, dist, dist_bounds))
    
    param_num = len(model.feedback_model.initial_guess)
    
    dist_bounds_new = list(zip(*model.feedback_model.bounds))
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
    

def dist_from_hist_1d(model, dist, dist_bounds):
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
        return(uniform_prior(model, dist, dist_bounds))
    hists     = []
    edges     = []
    
    param_num = len(model.feedback_model.initial_guess)
    dist_bounds_new = list(zip(*model.feedback_model.bounds))
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

def uniform_prior(model, dist, dist_bounds):
    '''
    Create n histograms that match n independent uniform prior.
    Returns normalized histograms (probabilities) and edges of histograms, both
    as lists. Also returns new set of bounds for the histogram, to be used on 
    the next iteration.
    '''
    hists     = []
    edges     = []
    param_num = len(model.feedback_model.initial_guess)
    dist_bounds_new = list(zip(*model.feedback_model.bounds))
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

        
################ PARAMETER ESTIMATION #########################################
def calculate_MAP_estimator(prior, model, method = 'annealing', bounds = None,
                            x0 = None):
    '''
    Calculate 'best-fit' value of parameter by searching for the global minimum
    of the posterior distribution (Maximum A Posteriori estimator).
    '''
    
    if bounds is None:
        bounds = list(zip(*model.feedback_model.bounds))
    
    def neg_log_prob(params):
        val = (log_prior(params, prior) + log_likelihood(params, model))*(-1) 
        if not np.isfinite(val): # huge val, but not infite so that optimization works
            val = 1e+30 
        return(val)
    
    if method == 'minimize':
        if x0 is None:
            raise ValueError('x0 must be specificed for \'minimize\' method.')
        optimization_res = minimize(neg_log_prob, x0 = x0, bounds = bounds)
    
    elif method == 'annealing':
        optimization_res = dual_annealing(neg_log_prob, 
                                          bounds = bounds,
                                          maxiter = 100)
    elif method == 'brute':
        optimization_res = brute(neg_log_prob, 
                                 ranges = bounds,
                                 Ns = 100)        
        
    par = optimization_res.x
    if not optimization_res.success:
        print('Warning: MAP optimization did not succeed')
    return(par)