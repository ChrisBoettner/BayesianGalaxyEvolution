#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 17:46:44 2022

@author: chris
"""
import warnings

import numpy as np

from model.calibration.modelling import fit_model
from model.calibration.parameter import load_parameter

################ MAIN CLASSES #################################################
class ModelResult():
    '''
    Central model object.
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
        elif self.feedback_name == 'stellar':
                self.plot_parameter('C1',    's', '--', 'Stellar Feedback')
        elif self.feedback_name == 'stellar_blackhole':
                self.plot_parameter('C2',    'v', '-.', 'Stellar + Black Hole Feedback')
        elif self.feedback_name == 'changing':
            self.plot_parameter(['C2']*5+['C1']*6, 'o', '-', 'Changing Feedback')
        else:
            warnings.warn('Plot parameter not defined')
        
    def plot_parameter(self, color, marker, linestyle, label):
        '''Style parameter for plots.'''
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

    def calculate_abundance(self, log_quantity, z, parameter):
        '''
        Calculate the value of number density function (SMF/LF) for a given 
        input quantity, redshift and feedback model parameter.
        '''
        return(self.model.at_z(z).log_ndf(log_quantity, parameter))

    def calculate_ndf(self, z, parameter, quantity_range = None):
        '''Calculate number density function over representative range. '''
        if quantity_range is None:
            # slighlty random values, since this makes plots nicer without
            # having to adjust ticks
            if self.quantity_name == 'Muv':
                quantity_range = np.linspace(-23.42,-12.46,100)
            elif self.quantity_name == 'mstar':
                quantity_range = np.linspace(7.42,11.97,100)
            else:
                raise ValueError('quantity_name not known.')
                
        ndf = self.calculate_abundance(quantity_range, z, parameter)
        return([quantity_range, ndf])
    
    def get_ndf_sample(self, z, num = 100, quantity_range = None):
        '''
        Get a sample of ndf curves (as a list) with parameters randomly drawn
        from the distribution.
        '''
        parameter_sample = self.draw_parameter_sample(z, num = num)
        ndf_sample = []
        for n in range(num):
            ndf = self.calculate_ndf(z, parameter_sample[n], 
                                     quantity_range = quantity_range)
            ndf_sample.append(np.array(ndf).T)
        return(ndf_sample)

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