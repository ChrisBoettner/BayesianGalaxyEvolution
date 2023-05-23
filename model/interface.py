#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 11:51:46 2022

@author: chris
"""
import timeit

from model.data.load import load_ndf_data
from model.model import choose_model
from model.calibration.parameter import save_parameter
from model.helper import is_sublist, make_list
from model.quantity_options import get_quantity_specifics


def load_model(quantity_name, physics_name, data_subset=None,
               prior_name='successive', redshift=None, 
               parameter_calc=False, **kwargs):
    '''
    Load saved (MCMC) model. Built for simplicity so that physics_name is
    associated with specific prior, but can be changed if needed.
    Loads parameter from file.
    Can choose to load and run on subset of data, just put in list of data set
    names (of form AuthorYear).
    '''
    cutoff = get_quantity_specifics(quantity_name)['cutoff']
    groups, log_ndfs = load_ndf_data(quantity_name, cutoff, data_subset)
    if redshift is None:
        redshift = list(log_ndfs.keys())
        
    Model = choose_model(quantity_name)
    model = Model(redshift, log_ndfs,
                  quantity_name, physics_name, prior_name,
                  fitting_method='mcmc',
                  saving_mode='loading',
                  groups=groups,
                  name_addon=data_subset,
                  parameter_calc=parameter_calc,
                  **kwargs)
    return(model)


def save_model(quantity_name, physics_name, data_subset=None,
               prior_name='successive', redshift=None, parameter_calc=True,
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
    if redshift is None:
        redshift = list(log_ndfs.keys())

    print('Quantity: '      + quantity_name + ' | ' +
          'Prior Model: '   + prior_name + ' | ' +
          'Physics Model: ' + physics_name + '\n')
    start = timeit.default_timer()

    Model = choose_model(quantity_name)
    model = Model(redshift, log_ndfs,
                  quantity_name, physics_name, prior_name,
                  fitting_method='mcmc',
                  saving_mode='saving',
                  groups=groups,
                  name_addon=data_subset,
                  parameter_calc=parameter_calc,
                  **kwargs)
    
    if parameter_calc:
        save_parameter(model, data_subset)
    end = timeit.default_timer()
    print('\nDONE')
    print('Total Time: ' + str((end - start) / 3600) + ' hours')
    return(model)


def run_model(quantity_name, physics_name, fitting_method='mcmc',
              data_subset=None, prior_name='successive', redshift=None,
              saving_mode='temp', parameter_calc=True,
              **kwargs):
    '''
    Run a model calibration without saving.
    Can choose to load and run on subset of data, just put in list of data set
    names (of form AuthorYear).
    '''
    cutoff = get_quantity_specifics(quantity_name)['cutoff']
    groups, log_ndfs = load_ndf_data(quantity_name, cutoff, data_subset)

    if redshift is None:
        redshift = list(log_ndfs.keys())
    else:
        redshift = make_list(redshift)
        if not is_sublist(redshift, list(log_ndfs.keys())):
            return ValueError('redshifts not in dataset.')

    Model = choose_model(quantity_name)
    model = Model(redshift, log_ndfs,
                  quantity_name, physics_name, prior_name,
                  fitting_method=fitting_method,
                  saving_mode=saving_mode,
                  groups=groups,
                  name_addon=data_subset,
                  parameter_calc=parameter_calc,
                  **kwargs)
    return(model)
