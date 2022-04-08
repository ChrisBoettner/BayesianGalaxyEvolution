#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 17:27:51 2022

@author: chris
"""
import numpy as np

from model.calibration.modelling import fit_model

################ MAIN CLASSES #################################################
class CalibrationResult():
    '''
    Container object that contains all important information about modelled
    ndf.
    '''
    def __init__(self, redshifts, ndfs, hmfs, quantity_name, feedback_name, 
                 prior_name, fitting_method, saving_mode, name_addon = None, **kwargs):
        parameter, distribution, model = fit_model(redshifts, ndfs, hmfs,
                                                   quantity_name, feedback_name,
                                                   prior_name, fitting_method,
                                                   saving_mode,
                                                   name_addon = name_addon,
                                                   **kwargs)
        self.quantity_name = quantity_name
        self.feedback_name = feedback_name
        self.prior_name    = prior_name
        
        self.redshift      = redshifts
        
        self.parameter     = data(parameter,    redshifts = self.redshift)
        self.distribution  = data(distribution, redshifts = self.redshift)
        
        self.model         = data(model,        redshifts = self.redshift)
        
    def plot_parameter(self, color, marker, linestyle, label):
        # style parameter for fit
        self.color     = color
        self.marker    = marker
        self.linestyle = linestyle
        self.label     = label
        return(self)
 
class data():
    # easily retrieve data at certain redshift
    def __init__(self, data, redshifts):
        self.data      = data
        self.redshifts = np.array(redshifts)
    def at_z(self, z):
        if z not in self.redshifts:
            raise ValueError('Redshift not in data')
        else:
            #import pdb; pdb.set_trace()
            return(self.data[np.argwhere(self.redshifts == z)[0][0]])

################ SAVE FUNCTIONS ###############################################
def save_parameter(CalibrationResult):
    '''
    Manually save best fit parameter as numpy npz (dictonary of form str(z):parameter)
    to folder that usually contains distribution data.
    '''
    redshifts = CalibrationResult.redshift
    parameter = CalibrationResult.parameter.data
    if np.any(None in [p for par in parameter for p in par]):
        raise ValueError('Parameter are None type, probably weren\'t calculated.')
    
    parameter = {str(redshifts[i]):parameter[i] for i in range(len(redshifts))}
    
    save_path = CalibrationResult.model.at_z(CalibrationResult.redshift[0]).directory
          
    filename = CalibrationResult.prior_name + '_parameter' '.npz'
    np.savez(save_path + filename, **parameter)
    return