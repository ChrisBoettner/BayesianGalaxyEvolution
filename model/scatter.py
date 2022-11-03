#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 16:16:39 2022

@author: chris
"""
import numpy as np
from scipy.stats import norm, cauchy

def scatter_model(scatter_name):
    '''
    Select model distribution for the scatter in the quantity - halo mass 
    relation. Distributions are specified through location and scale parameter.
    Currently all distributions are in log space rather than linear
    space.
    Available distributions are
        normal: 
            Gaussian distribution, loc is the expectation value,scale is the
            standard deviation.
        cauchy:
            Cauchy distribution, characteristic for its heavy tails. Better
            model to include large outliers. Pathological distribution in the
            sense that expectation value and variance are not defined. For
            integral (also those for our model) to converge, some cutoff must
            be set for the distribution (done in the relevant method in the
            ModelResult class). loc is the mode/median, scale describes spread.
    
    '''
    
    if scatter_name == 'normal':
        scatter = norm
    elif scatter_name == 'cauchy':
        raise NotImplementedError(
                'cauchy doesnt quite work yet because of covergence issues \n'
                'you should probably implement your idea with the switch to '
                'a gaussian.')
        scatter = cauchy
    else:
        raise NotImplementedError('scatter_name not known')
    return(scatter)


# class Joint_probability_distribution():
# you were mistaken, no need to calculate normalisation
#     def __init__(self, model, scatter_name, scatter_parameter=None):
#         self.model             = model
#         self.scatter_name      = scatter_name
#         self.scatter           = scatter_model(scatter_name)
#         self.scatter_parameter = scatter_parameter
    
#     def _calculate_normalisation(self, z, parameter, scatter_parameter = None):
#         value_grid = 0
#         return(value_grid)
        
#     def _unnormalized_probablility(self, log_quantity, log_halo_mass, z, 
#                                    parameter,  scatter_parameter = None):
        
#         if scatter_parameter is None:
#             scatter_parameter=self.scatter_parameter
        
#         # calculate location parameter for distribution for the input halo
#         # masses
#         log_Q = self.model.physics_model.at_z(z).\
#                            calculate_log_quantity(log_halo_mass, *parameter)
        
#         # calculate unnormalized pdf
#         scatter_contribution =  self.scatter.pdf(x = log_quantity,
#                                                  loc = log_Q,
#                                                  scale = scatter_parameter)
#         hmf_contribution     = np.power(10, self.model.calculate_log_hmf(
#                                                     log_halo_mass, z))
#         return(scatter_contribution*hmf_contribution)
        