#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 11:51:46 2022

@author: chris
"""
import timeit

from model.data.load import load_ndf_data, load_hmf_functions
from model.model import choose_model
from model.calibration.parameter import save_parameter
from model.helper import is_sublist, make_list
from model.quantity_options import get_quantity_specifics


def load_model(quantity_name, physics_name, data_subset=None,
               prior_name=None, redshift=None, **kwargs):
    '''
    Load saved (MCMC) model. Built for simplicity so that physics_name is
    associated with specific prior, but can be changed if needed.
    Loads parameter from file.
    Can choose to load and run on subset of data, just put in list of data set
    names (of form AuthorYear).
    '''
    cutoff = get_quantity_specifics(quantity_name)['cutoff']
    groups, log_ndfs = load_ndf_data(quantity_name, cutoff, data_subset)
    log_hmfs = load_hmf_functions()
    if redshift is None:
        redshift = list(log_ndfs.keys())

    if prior_name is None:
        if physics_name == 'changing':
            prior_name = 'successive'
        else:
            prior_name = 'uniform'

    Model = choose_model(quantity_name)
    model = Model(redshift, log_ndfs, log_hmfs,
                  quantity_name, physics_name, prior_name,
                  groups=groups,
                  name_addon=data_subset,
                  fitting_method='mcmc',
                  saving_mode='loading',
                  parameter_calc=False,
                  **kwargs)
    return(model)


def save_model(quantity_name, physics_name, data_subset=None,
               prior_name=None, redshift=None, parameter_calc=True,
               **kwargs):
    '''
    Run and save (MCMC) model. Built for simplicity so that physics_name is
    associated with specific prior, but can be changed if needed.
    Also saves parameter to same folder.
    Can choose to load and run on subset of data, just put in list of data set
    names (of form AuthorYear).
    '''
    cutoff = get_quantity_specifics(quantity_name)['cutoff']
    groups, log_ndfs = load_ndf_data(quantity_name, cutoff, data_subset)
    log_hmfs = load_hmf_functions()
    if redshift is None:
        redshift = list(log_ndfs.keys())

    if prior_name is None:
        if physics_name == 'changing':
            prior_name = 'successive'
        else:
            prior_name = 'uniform'

    print(quantity_name, prior_name, physics_name)
    start = timeit.default_timer()

    Model = choose_model(quantity_name)
    model = Model(redshift, log_ndfs, log_hmfs,
                  quantity_name, physics_name, prior_name,
                  groups=groups,
                  name_addon=data_subset,
                  fitting_method='mcmc',
                  saving_mode='saving',
                  parameter_calc=parameter_calc,
                  **kwargs)
    save_parameter(model, data_subset)
    end = timeit.default_timer()
    print('\nDONE')
    print('Total Time: ' + str((end - start) / 3600) + ' hours')
    return(model)


def run_model(quantity_name, physics_name, fitting_method='least_squares',
              data_subset=None, prior_name=None, redshift=None,
              saving_mode='temp', parameter_calc=True,
              **kwargs):
    '''
    Run a model calibration without saving. Default is least_squares fit (without
    mcmc), but can be changed. If fitting_method is mcmc, uses low chain_length and
    num_walker so that it finishes quickly (autocorr_discard disabled by
    default, also adjustable).
    Can choose to load and run on subset of data, just put in list of data set
    names (of form AuthorYear).
    '''
    cutoff = get_quantity_specifics(quantity_name)['cutoff']
    groups, log_ndfs = load_ndf_data(quantity_name, cutoff, data_subset)
    log_hmfs = load_hmf_functions()

    if redshift is None:
        redshift = list(log_ndfs.keys())
    else:
        redshift = make_list(redshift)
        if not is_sublist(redshift, list(log_ndfs.keys())):
            return ValueError('redshifts not in dataset.')

    if prior_name is None:
        if physics_name == 'changing':
            prior_name = 'successive'
        else:
            prior_name = 'uniform'

    Model = choose_model(quantity_name)
    model = Model(redshift, log_ndfs, log_hmfs,
                  quantity_name, physics_name, prior_name,
                  groups=groups,
                  fitting_method=fitting_method,
                  name_addon=data_subset,
                  parameter_calc=parameter_calc,
                  saving_mode=saving_mode,
                  **kwargs)
    return(model)
