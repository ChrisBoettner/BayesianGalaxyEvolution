#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:23:57 2022

@author: chris
"""

import numpy as np
import emcee
from progressbar import FormatLabel, NullBar

import os
from multiprocessing import Pool
from pathlib import Path

from scipy.optimize import dual_annealing, minimize, brute

from model.calibration.leastsq_fitting import cost_function
from model.helper import within_bounds, custom_progressbar

################ MAIN FUNCTIONS ###############################################


def mcmc_fit(model, prior, saving_mode,
             num_walker=250, min_chain_length = 10000, tolerance=0.01,
             autocorr_chain_multiple = 50, autocorr_discard=10, 
             parameter_calc=True, parallel=True, progress=True):
    '''
    Calculate parameter that match observed numbder density function (LF/SMF)
    to modelled functions using MCMC fitting by maximizing the log probability
    function which is the product of the log likelihood and log prior.

    num_walkers is the number of walkers for mcmc sampling.
    min_chain_length is the minimum length of the Markov chains for it to be
    considered converged (as a multiple of the autocorrelation time).

    Parameters
    ----------
    model : ModelResult instance
        The ModelResult that is used for the fit.
    prior : array
        A numpy array output of np.histogram used to calculate the prior value.
    saving_mode : str
        Choose if Markov chain is supposed to be saved/loaded or not stored.
        Options are 'saving', 'loading' and temp.
    num_walker : int, optional
        Number of walkers for mcmc sampling. The default is 250.
    min_chain_length : int, optional
        Minimum length of the Markov chains (after discarding burn-in).
        The default is 10000.
    tolerance : float, optional
        Allowed relative deviation from one autocorrelation estimate to the
        next for the autocorrelation to be considered converged. The default 
        is 0.01.
    autocorr_chain_multiple : int, optional
        Minimum (length of chain/maximum autocorr time) for autocorr estimate
        to be considered trustworthy. The default is 50
    autocorr_discard : int, optional
        Multiple of maximum autocorrelation time that is regarded as burn-in
        and discarded (turn off with 0). The default is 10
    parameter_calc : bool, optional
        Choose if best fit parameter (MAP estimator) is supposed to be 
        calculated. The default is True.
    parallel : bool, optional
        Choose if calculation is supposed to be done in parallel. The default 
        is True.
    progress : bool, optional
        Choose if progress bar is supposed to be shown. The default is True.

    Raises
    ------
    FileNotFoundError
        If mode is 'loading' and mcmc chain is supposed to be loaded from a 
        file that does not exist.
    ValueError
        If saving_mode is not known.
    McmcConvergenceError
        If autocorrelation estimate does not converge in maximum chain length.

    Returns
    -------
    None.

    '''
    # initalize saving for mcmc runs
    if saving_mode == 'temp':
        savefile = None
    elif saving_mode in ['saving', 'loading']:
        save_path = model.directory
        # create dir if it doesn't exist
        Path(save_path).mkdir(parents=True, exist_ok=True)
        filename = save_path + model.filename.at_z(model._z) + '.h5'
        if (saving_mode == 'loading') and not Path(filename).is_file():
            raise FileNotFoundError('mcmc data file does not exist.')
        savefile = emcee.backends.HDFBackend(filename)
    else:
        raise ValueError('saving_mode not known.')

    # select initial walker positions near initial guess
    initial_guess = np.array(model.feedback_model.at_z(model._z).initial_guess)
    ndim = len(initial_guess)
    nwalkers = num_walker
    walker_pos = initial_guess * (1 + 0.1 * np.random.rand(nwalkers, ndim))

    # make prior and model object a global variable so it doesn"t have to be 
    # called in log_probability explicitly. This helps with parallization and 
    # makes it a lot faster,
    # see https://emcee.readthedocs.io/en/stable/tutorials/parallel/
    global mod_global
    mod_global = model
    global prior_global
    prior_global = prior

    if saving_mode in ['saving', 'temp']:     
        # define progressbar
        if progress:
            ProgressBar = custom_progressbar()
        else:
            ProgressBar = NullBar()
        
        if saving_mode == 'saving' and os.path.exists(filename):
            os.remove(filename)  # clear file before start writing to it
        with Pool() as pool:
            # create MCMC sampler
            if parallel:      
                    sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                                    log_probability,
                                                    backend = savefile,
                                                    pool=pool)
            else:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                                backend = savefile)
                
            # run burn-in until autocorrelation converges
            max_iterations = int(5e+5)
            convergence_flag = False
            old_tau = np.inf            
            mcmc = sampler.sample(walker_pos, iterations=max_iterations)
            
            if progress:
                ProgressBar.widgets[0] = f'z = {model._z}'
            for sample in ProgressBar(mcmc):
                # only check convergence every 1000 steps
                if sampler.iteration % 100:
                    continue
                # compute the autocorrelation time so far
                tau = sampler.get_autocorr_time(tol=0)      
                # check convergence
                rel_deviation = np.abs(old_tau - tau) / tau
                converged  = np.all(rel_deviation < tolerance)
                converged &= np.all(tau * autocorr_chain_multiple 
                                    < sampler.iteration)
                
                if progress:
                    autocorr_update = f'Autocorrelation estimate: {tau}'
                    ProgressBar.widgets[-1] = FormatLabel(autocorr_update)
                
                discard = autocorr_discard*np.amax(tau) 
                if converged and (sampler.iteration-discard>=min_chain_length):
                    convergence_flag = True
                    break
                else:
                    old_tau = tau
            if not convergence_flag:
                raise McmcConvergenceError('Autocorrelation estimate did not'
                                           ' converge.')
                
    elif saving_mode == 'loading':
        # load from savefile
        sampler = savefile
        tau = sampler.get_autocorr_time()   
    else:
        raise ValueError('saving_mode not known.')

    # discard burn-in of mcmc walk
    posterior_samp = sampler.get_chain(
                      discard= autocorr_discard*np.amax(tau).astype(int),
                      flat=True)

    # calculate best fit parameter (MAP) using annealing
    if parameter_calc:
        bounds = list(zip(np.percentile(posterior_samp, 16, axis=0),
                          np.percentile(posterior_samp, 84, axis=0)))
        params = calculate_MAP_estimator(prior_global, model,
                                         method='annealing',
                                         bounds=bounds,
                                         x0=np.median(
                                             posterior_samp, 
                                             axis=0))
    else:
        params = None

    return(params, posterior_samp)

################ PROBABILITY FUNCTIONS ########################################


def log_probability(params):
    '''
    Calculate total probability (which will be maximized by MCMC fitting).
    Total probability is given by likelihood*prior_probability, meaning
    logprobability = loglikelihood+logprior
    '''

    # check if parameter are within bounds
    if not within_bounds(params, *mod_global.feedback_model.at_z(
                        mod_global._z).bounds):
        return(-np.inf)

    # PRIOR
    l_prior = log_prior(params, prior_global)
    # LIKELIHOOD
    l_likelihood = log_likelihood(params, mod_global)
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
    bin_widths = [e[1] - e[0] for e in edges]
    ind = np.array([int(params[i] / bin_widths[i])
                   for i in range(len(params))])
    ind[ind < 0] = 0
    ind_upper_lim = (len(edges[0]) - 2)  # highest ind in hist
    ind[ind > ind_upper_lim] = ind_upper_lim

    # if probabilities are assumed independent: get probability for each param
    # from corresponding histogram and then multiply all together to get total
    # probability
    if len(hists) > 1:
        indiv_prob = [hists[i][ind[i]] for i in range(len(params))]
        total_prob = np.prod(indiv_prob)

    # if probabilities are not assumed independent: get total probability form
    # n-dimensional histogram
    else:
        total_prob = hists[0][tuple(ind)]

    if total_prob == 0:
        return(-np.inf)  # otherwise log will throw an error
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

    param_num = len(model.feedback_model.at_z(model._z).initial_guess)

    dist_bounds_new = list(zip(*model.feedback_model.at_z(model._z).bounds))
    dist_bounds = dist_bounds_new
    
    # if new model has fewer parameter, marginalise over leftover parameter
    if param_num<dist.shape[1]:
        dist = dist[:,:-(dist.shape[1]-param_num)]
    elif param_num > dist.shape[1]:
        raise NotImplementedError('Parameter number of model should not increase.')
    else:
        pass
    
    # create histogram
    hist_nd, edges = np.histogramdd(dist, bins=500, range=dist_bounds)
    # make empty spots have 0.1% of actual prob, so that these are not
    # completely ignored
    if np.any(hist_nd == 0):
        hist_nd = hist_nd.astype(float)
        hist_nd[hist_nd == 0] = 0.001 * np.sum(hist_nd) / np.sum(hist_nd == 0)
    # normalize
    hist_nd = hist_nd / np.sum(hist_nd)
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
    hists = []
    edges = []

    param_num = len(model.feedback_model.at_z(model._z).initial_guess)
    dist_bounds_new = list(zip(*model.feedback_model.at_z(model._z).bounds))
    if dist_bounds is None:  # if bounds are not provided, use the ones from model
        dist_bounds = dist_bounds_new

    # if dist.shape[1] > param_num, ignore the later columns of dist
    for i in range(param_num):
        lower_bound = dist_bounds[i][0]
        upper_bound = dist_bounds[i][1]
        hist, edge = np.histogram(dist[:, i], range=(lower_bound, upper_bound),
                                  bins=500)
        # make empty spots have 0.1% of actual prob, so that these are not
        # completely ignored
        if np.any(hist == 0):
            hist = hist.astype(float)
            hist[hist == 0] = 0.001 * np.sum(hist) / np.sum(hist == 0)
        # normalize
        hist = hist / np.sum(hist)

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
    hists = []
    edges = []
    param_num = len(model.feedback_model.at_z(model._z).initial_guess)
    dist_bounds_new = list(zip(*model.feedback_model.at_z(model._z).bounds))
    if dist_bounds is None:  # if bounds are not provided, use the ones from model
        dist_bounds = dist_bounds_new

    for i in range(param_num):
        lower_bound = dist_bounds[i][0]
        upper_bound = dist_bounds[i][1]
        hist = np.array([1 / (upper_bound - lower_bound)])
        edge = np.array([lower_bound, upper_bound])
        hists.append(hist)
        edges.append(edge)
    return([hists, edges], dist_bounds)


################ ERROR HANDLING ###############################################
class McmcConvergenceError(Exception):
    '''Raised when autocorrelation estimate did not converge.'''
    pass

################ PARAMETER ESTIMATION #########################################
def calculate_MAP_estimator(prior, model, method='annealing', bounds=None,
                            x0=None):
    '''
    Calculate 'best-fit' value of parameter by searching for the global minimum
    of the posterior distribution (Maximum A Posteriori estimator).
    '''

    if bounds is None:
        bounds = list(zip(*model.feedback_model.at_z(model._z).bounds))

    def neg_log_prob(params):
        val = (log_prior(params, prior) + log_likelihood(params, model)) * (-1)
        if not np.isfinite(
                val):  # huge val, but not infite so that optimization works
            val = 1e+30
        return(val)

    if method == 'minimize':
        if x0 is None:
            raise TypeError('x0 must be specificed for \'minimize\' method.')
        optimization_res = minimize(neg_log_prob, x0=x0, bounds=bounds)

    elif method == 'annealing':
        optimization_res = dual_annealing(neg_log_prob,
                                          bounds=bounds,
                                          maxiter=1000)
    elif method == 'brute':
        optimization_res = brute(neg_log_prob,
                                 ranges=bounds,
                                 Ns=100)
    else:
        raise ValueError('method not known.')

    par = optimization_res.x
    if not optimization_res.success:
        print('Warning: MAP optimization did not succeed')
    return(par)
