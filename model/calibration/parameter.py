#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 17:52:32 2022

@author: chris
"""
import numpy as np

################ SAVE FUNCTIONS ###############################################


def save_parameter(ModelResult, name_addon=None):
    '''
    Save best fit parameter as numpy npz (dictonary of form str(z):parameter)
    to folder that usually contains distribution data.
    '''
    redshifts = ModelResult.redshift
    parameter = ModelResult.parameter.data
    if ModelResult.parameter.is_None():
        raise ValueError(
            'parameter dict is empty, probably weren\'t calculated.')

    parameter = {str(z): p for z, p in parameter.items()}

    save_path = ModelResult.directory
    if name_addon:
        filename = ModelResult.prior_name + \
            ''.join(name_addon) + '_parameter' '.npz'
    else:
        filename = ModelResult.prior_name + '_parameter' '.npz'
    np.savez(save_path + filename, **parameter)
    return


def load_parameter(ModelResult, name_addon=None):
    '''
    Load best fit parameter.
    '''
    save_path = ModelResult.directory
    if name_addon:
        filename = ModelResult.prior_name + \
            ''.join(name_addon) + '_parameter' '.npz'
    else:
        filename = ModelResult.prior_name + '_parameter' '.npz'
    parameter = np.load(save_path + filename)
    return(parameter)
