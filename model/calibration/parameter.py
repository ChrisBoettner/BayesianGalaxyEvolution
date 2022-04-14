#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 17:52:32 2022

@author: chris
"""
import numpy as np

################ SAVE FUNCTIONS ###############################################
def save_parameter(ModelResult, name_addon = None):
    '''
    Save best fit parameter as numpy npz (dictonary of form str(z):parameter)
    to folder that usually contains distribution data.
    '''
    redshifts = ModelResult.redshift
    parameter = ModelResult.parameter.data
    if np.any(None in [p for par in parameter for p in par]):
        raise ValueError('Parameter are None type, probably weren\'t calculated.')
    
    parameter = {str(redshifts[i]):parameter[i] for i in range(len(redshifts))}
    
    save_path = ModelResult.model.at_z(ModelResult.redshift[0]).directory
    if name_addon:
        filename = ModelResult.prior_name + ''.join(name_addon) + '_parameter' '.npz'
    else:
        filename = ModelResult.prior_name + '_parameter' '.npz'
    np.savez(save_path + filename, **parameter)
    return

def load_parameter(ModelResult, name_addon = None):
    '''
    Load best fit parameter.
    '''
    save_path = ModelResult.model.at_z(ModelResult.redshift[0]).directory
    if name_addon:
        filename = ModelResult.prior_name + ''.join(name_addon) + '_parameter' '.npz'
    else:
        filename = ModelResult.prior_name + '_parameter' '.npz'
    parameter = np.load(save_path + filename)
    return(parameter)