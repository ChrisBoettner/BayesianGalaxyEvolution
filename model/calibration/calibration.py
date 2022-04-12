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
                 prior_name, fitting_method, saving_mode, name_addon = None,
                 groups = None,
                 **kwargs):
        parameter, distribution, model = fit_model(redshifts, ndfs, hmfs,
                                                   quantity_name, feedback_name,
                                                   prior_name, fitting_method,
                                                   saving_mode,
                                                   name_addon = name_addon,
                                                   **kwargs)
        self.quantity_name = quantity_name
        self.feedback_name = feedback_name
        self.prior_name    = prior_name
        
        self.redshift      = np.array(redshifts)
        self.distribution  = data(distribution, redshifts = self.redshift)
        self.model         = data(model,        redshifts = self.redshift)
        self.parameter     = data(parameter,    redshifts = self.redshift)
        if saving_mode == 'loading':
            parameter      = load_parameter(self, name_addon)
            parameter      = [p for p in parameter.values()]
            self.parameter = data(parameter, redshifts = self.redshift)
            
        self.groups        = groups
        
        # default plot parameter per feedback_model
        if self.feedback_name == 'none':
                self.plot_parameter('black', 'o', '-',  'No Feedback')
        if self.feedback_name == 'stellar':
                self.plot_parameter('C1',    's', '--', 'Stellar Feedback')
        if self.feedback_name == 'stellar_blackhole':
                self.plot_parameter('C2',    'v', '-.', 'Stellar + Black Hole Feedback')
        if self.feedback_name == 'changing':
            self.plot_parameter(['C2']*5+['C1']*6,
                                'o',
                                '-', 
                                'Changing Feedback')
        
    def plot_parameter(self, color, marker, linestyle, label):
        '''Style parameter for fit. '''
        self.color     = color
        self.marker    = marker
        self.linestyle = linestyle
        self.label     = label
        return(self)

    def draw_parameter_sample(self, z, num = 1):
        '''
        Get a sample from feedback parameter distribution at given redshift.
        '''
        # randomly draw from parameter distribution at z 
        random_draw      = np.random.choice(self.distribution.at_z(z).shape[0],
                                            size = num)
        parameter_sample = self.distribution.at_z(z)[random_draw]
        return(parameter_sample)
 
    def calculate_quantity_distribution(self, log_halo_mass, z, num = int(1e+5)):
        '''
        At a given redshift, calculate distribution of observable quantity 
        (mstar/Muv) for a given halo mass by drawing parameter sample and
        calculating value for each one (number of draws adaptable.)
        '''     
        parameter_sample = self.draw_parameter_sample(z, num = num)
        
        log_quantity_dist = []
        for p in parameter_sample:
            log_quantity_dist.append(self.model.at_z(z).
                                     feedback_model.
                                     calculate_log_observable(log_halo_mass, *p))
        return(np.array(log_quantity_dist))    
 
    def calculate_halo_mass_distribution(self, log_quantity, z, num = int(1e+5)):
        '''
        At a given redshift, calculate distribution of halo mass for a given 
        observable quantity (mstar/Muv) by drawing parameter sample and
        calculating value for each one (number of draws adaptable.)
        '''  
        parameter_sample = self.draw_parameter_sample(z, num = num)
        
        log_halo_mass_dist = []
        for p in parameter_sample:
            log_halo_mass_dist.append(self.model.at_z(z).
                                     feedback_model.
                                     calculate_log_halo_mass(log_quantity, *p))
        return(np.array(log_halo_mass_dist))

    def calculate_ndf(self, log_quantity, z, parameter):
        '''
        Calculate the number density function (SMF/LF) for a given input quantity,
        redshift and feedback model parameter.
        '''
        return(self.model.at_z(z).log_ndf(log_quantity, parameter))

    def calculate_ndf_curve(self, z, parameter, quantity_range = None):
        '''Calculate number density function over representative range. '''
        if quantity_range is None:
            if self.quantity_name == 'Muv':
                quantity_range = np.linspace(-23.421,-12.4623,100)
            if self.quantity_name == 'mstar':
                quantity_range = np.linspace(7.424,11.974,100)

        ndf = self.calculate_ndf(quantity_range, z, parameter)
        return(quantity_range, ndf)

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
def save_parameter(CalibrationResult, name_addon = None):
    '''
    Save best fit parameter as numpy npz (dictonary of form str(z):parameter)
    to folder that usually contains distribution data.
    '''
    redshifts = CalibrationResult.redshift
    parameter = CalibrationResult.parameter.data
    if np.any(None in [p for par in parameter for p in par]):
        raise ValueError('Parameter are None type, probably weren\'t calculated.')
    
    parameter = {str(redshifts[i]):parameter[i] for i in range(len(redshifts))}
    
    save_path = CalibrationResult.model.at_z(CalibrationResult.redshift[0]).directory
    filename = CalibrationResult.prior_name + ''.join(name_addon) + '_parameter' '.npz'
    
    np.savez(save_path + filename, **parameter)
    return

def load_parameter(CalibrationResult, name_addon = None):
    '''
    Load best fit parameter.
    '''
    save_path = CalibrationResult.model.at_z(CalibrationResult.redshift[0]).directory
    filename  = CalibrationResult.prior_name + name_addon + '_parameter' '.npz'
    parameter = np.load(save_path + filename)
    return(parameter)