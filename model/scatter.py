#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 16:16:39 2022

@author: chris
"""
import numpy as np
from scipy.stats import norm, cauchy
from scipy.integrate import trapezoid

class Joint_distribution():
    def __init__(self, model, scatter_name, scatter_parameter=None):
        '''
        Class to calculate quantities related to scatter in the quantity 
        - halo mass relation. Distributions are specified through location and 
        scale parameter
        Available distributions are
            lognormal (constant scale):
                Gaussian distribution in log quantity space.
            normal (halo mass-dependent scale): 
                Gaussian distribution in linear quantity space. Currently,
                the sigma is implemented as a constant fraction of the 
                Q value.
            cauchy:
                Cauchy distribution in linear quantity space, characteristic 
                for its heavy tails. Better model to include large outliers. 
                Pathological distribution in the sense that expectation value
                and variance are not defined. Currently,
                the sigma is implemented as a constant fraction of the 
                Q value.
        Parameters
        ----------
        model : ModelResult object
            ModelResult, used for calculating halo masses and quantities from
            implemented physics model.
        scatter_name : str
            Name of the scatter model. Currently ['lognormal', 'normal', 
            'cauchy'].
        scatter_parameter: float, optional
            Value for the scale parameter of the scatter model. The default
            value is None, but then must be specified for every method call.
                
        '''
        self.model             = model
        self.scatter_name      = scatter_name
        self.scatter_parameter = scatter_parameter
    
    def calculate_number_density(self, log_quantity, log_halo_mass, z, 
                        parameter,  scatter_parameter=None):
        '''
        Calculate value of joint number density function for given log_quantity
        and log_halo_mass value. z and parameter needed for quantity - halo 
        mass relation.
        '''
        
        if scatter_parameter is None:
            scatter_parameter=self.scatter_parameter
            
        # calculate value of distribution
        scatter =  self.scatter_contribution(log_quantity,  log_halo_mass,
                                              z, parameter, scatter_parameter)
        hmf     = self.hmf_contribution(log_halo_mass, z)
        return(scatter*hmf)
    
    def calculate_quantity_marginal_density(self, log_quantity, z, parameter,
                                            scatter_parameter=None, **kwargs):
        '''
        Calculate value of marginal quantity ndf by integrating over all
        m_h space.
        '''
        if scatter_parameter is None:
            scatter_parameter=self.scatter_parameter
        
        # create points sampled in m_h_space
        log_m_h_space = self.make_log_m_h_space(log_quantity, z, parameter,
                                                **kwargs)
        # calculate ndf values at these m_h points and input quantity values
        number_densities = self.calculate_number_density(log_quantity, 
                                                         log_m_h_space,
                                                         z, parameter,
                                                         scatter_parameter)
        # integrate over sample
        marginal_density =trapezoid(y=number_densities, x=log_m_h_space,
                                          axis=0)
        return(marginal_density)
    
    def scatter_contribution(self, log_quantity, log_halo_mass, z, parameter,
                             scatter_parameter):
        '''
        Calculate contribution of scatter to joint ndf.
        '''
        # calculate location parameter for distribution for the input halo
        # masses
        log_Q = self.calculate_Q(log_halo_mass, z, parameter)
    
        if self.scatter_name == 'lognormal':
            pdf_values =  norm.pdf(x     = log_quantity,
                                   loc   = log_Q,
                                   scale = scatter_parameter)
        elif self.scatter_name == 'normal':
            # linear space
            pdf_values =  norm.pdf(x     = 10**log_quantity,
                                   loc   = 10**log_Q,
                                   scale = scatter_parameter*10**log_Q)
        elif self.scatter_name == 'cauchy':
            # linear space
            pdf_values =  cauchy.pdf(x     = 10**log_quantity,
                                     loc   = 10**log_Q,
                                     scale = scatter_parameter*10**log_Q)
        else:
            raise ValueError('scatter_name not known.')
            
        return(pdf_values)
        
    def hmf_contribution(self, log_halo_mass, z):
        '''
        Calculate contribution of HMF to joint ndf.
        '''
        hmf_values = np.power(10, self.model.calculate_log_hmf(
                                                log_halo_mass, z))
        return(hmf_values)
        
    def calculate_Q(self, log_halo_mass, z, parameter):
        '''
        Calculate Q(m_h) for given halo mass, redshift and quantity - halo
        mass relation parameter. (Convenience method)
        '''
        log_m_h = self.model.physics_model.at_z(z).calculate_log_quantity(
                                            log_halo_mass, *parameter)
        return(log_m_h)
        
    def calculate_Q_inverse(self, log_quantity, z, parameter):  
        '''
        Calculate m_h from q using Q^(-1)(m_h) for given quantity value, 
        redshift and quantity - halo mass relation parameter. 
        (Convenience method)
        '''
        log_Q = self.model.physics_model.at_z(z).calculate_log_halo_mass(
                                            log_quantity, *parameter)
        return(log_Q)
        
    
    def make_log_m_h_space(self, log_quantity, z, parameter, num=int(1e+4),
                           epsilon=1e-8):
        '''
        Create samples in log_m_h space in order to calculate quantity 
        marginal distribution. Points densest around maximum of distribution
        function q=Q(m_h) and then spread outwards to limits of m_h in 
        logarithmic manner. num controls number of data points, while
        epsilon controls how densely point are sampled around maximum of
        scatter distribution.
        '''
        # calculate halo masses using m_h=Q^(-1)(q)
        log_Q_inv  = self.calculate_Q_inverse(log_quantity, z, parameter)
        
        # slight offset used to calculate logarithmic space,
        # smaller value of epsilon means point are sampled more densely around
        # maximum and less densely further out
        log_m_epsilon = log_Q_inv-epsilon
        
        # create logarithmically spaced points to the left and right of m_h
        lower_part = np.geomspace(epsilon, 
                                  log_m_epsilon-self.model.log_min_halo_mass,
                                  num)
        upper_part = np.geomspace(epsilon, 
                                  self.model.log_max_halo_mass-log_m_epsilon,
                                  num)
        
        # transform to m_h_space
        lower_part = log_m_epsilon - lower_part
        upper_part = log_m_epsilon + upper_part
        
        # join spaces (sorted in order)
        log_m_h_space = np.concatenate([lower_part[::-1], upper_part])
        
        # make arrays 2d, needed for correct broadcasting in
        # calculate_quantity_marginal_density method
        if log_m_h_space.ndim == 1:
            log_m_h_space = log_m_h_space[:, np.newaxis]
        return(log_m_h_space)