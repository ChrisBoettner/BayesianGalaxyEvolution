#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 16:16:39 2022

@author: chris
"""
import numpy as np
from scipy.stats import norm, cauchy, skewnorm
from scipy.integrate import trapezoid

import warnings

def calculate_q1_q2_conditional_pdf(Jointdist1, Jointdist2, 
                                    log_q1, log_q2, parameter1, parameter2, z,
                                    **kwargs):
    '''
    Calculate conditional probability p(q1|q2), for a given value of  
    log_q1, log_q2.

    Parameters
    ----------
    Jointdist1 : Joint_distribution
        Joint_distribution instance for log_q1.
    Jointdist2 : Joint_distribution
        Joint_distribution instance for log_q2.
    log_q1 : float or array
        Values for which probability are to be calculated.
    log_q2 : float
        Value on which conditional probability is calcuted.
        MUST BE SCALAR IN CURRENT IMPLEMENTATION.
    parameter1 : array
        Parameter that describe quantity-halo mass relation for log_q1.
    parameter2 : array
        Parameter that describe quantity-halo mass relation for log_q2.
    z : int
        Redshift at which to calculate probability.
    **kwargs : dict
        Additional parameter passed to make_log_m_h_space.

    Raises
    ------
    ValueError
        Raised if log_q2 is not scalar.

    Returns
    -------
    probability: float or array
        Calculated probabilities.

    '''
    
    if not np.isscalar(log_q2):
        raise ValueError('In current implementation, log_q2 must be scalar.')
    
    log_m_h_space = Jointdist1.make_log_m_h_space(log_q1, z, parameter1,
                                                  **kwargs)
    # log_m_h_space2 = Jointdist2.make_log_m_h_space(log_q2, z, parameter2)
    # log_m_h_space  = np.unique(np.sort(np.concatenate([log_m_h_space, 
    #                                                     log_m_h_space2])),
    #                                                         axis=0)
    
    # calculate necessary integrals
    conditional_q1 = Jointdist1.quantity_conditional_density(log_q1, 
                                                             log_m_h_space,
                                                             z, parameter1)
    joint_q2       = Jointdist2.joint_number_density(log_q2, log_m_h_space,
                                                  z, parameter2)
    marginal_q2    = Jointdist2.quantity_marginal_density(log_q2, z, 
                                                          parameter2)
    
    # calculate probability
    integral    = trapezoid(conditional_q1*joint_q2, x=log_m_h_space, axis=0)
    probability = integral/marginal_q2
    
    if len(probability)==1:
        probability = probability[0]
    return(probability)


class Joint_distribution():
    def __init__(self, model, scatter_name, scatter_parameter=None, 
                 skew_parameter=None):
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
        skew_parameter: float, optional
            Value for the skew parameter of the scatter model, if skew is
            included. The default value is None, but then must be specified 
            for every method call if scatter_model requires that parameter.
                
        '''
        self.model             = model
        self.scatter_name      = scatter_name
        self.scatter_parameter = scatter_parameter
        self.skew_parameter    = skew_parameter
    
    def joint_number_density(self, log_quantity, log_halo_mass, z, 
                        parameter,  scatter_parameter=None,
                        skew_parameter=None):
        '''
        Calculate value of joint number density function for given log_quantity
        and log_halo_mass value. z and parameter needed for quantity - halo 
        mass relation.
        '''
        if scatter_parameter is None:
            scatter_parameter=self.scatter_parameter
        if skew_parameter is None:
            skew_parameter=self.skew_parameter
            
        # calculate value of distribution
        scatter =  self.quantity_conditional_density(log_quantity,  
                                                     log_halo_mass,
                                                     z, parameter,
                                                     scatter_parameter,
                                                     skew_parameter)
        hmf     = self.halo_mass_marginal_density(log_halo_mass, z)
        return(scatter*hmf)
    
    def quantity_marginal_density(self, log_quantity, z, parameter,
                                  scatter_parameter=None, 
                                  skew_parameter=None, **kwargs):
        '''
        Calculate value of marginal quantity ndf by integrating over all
        m_h space. kwargs are passed to make_log_m_h_space.
        '''
        if scatter_parameter is None:
            scatter_parameter=self.scatter_parameter
        if skew_parameter is None:
            skew_parameter=self.skew_parameter
        
        # create points sampled in m_h_space
        log_m_h_space = self.make_log_m_h_space(log_quantity, z, parameter,
                                                **kwargs)
        # calculate ndf values at these m_h points and input quantity values
        number_densities = self.joint_number_density(log_quantity, 
                                                     log_m_h_space,
                                                     z, parameter,
                                                     scatter_parameter,
                                                     skew_parameter)
        # integrate over sample
        marginal_density = trapezoid(y=number_densities, x=log_m_h_space,
                                     axis=0)
        return(marginal_density)
    
    def quantity_conditional_density(self, log_quantity, log_halo_mass, z, 
                                     parameter, scatter_parameter=None,
                                     skew_parameter=None, 
                                     skew_correction='median'):
        '''
        Calculate value of conditional density p(q|m_h).
        Use skew_correction for skewlognormal distribution to make log_Q
        desired central quantity.
        '''
        log_quantity = self.model.unit_conversion(log_quantity, 'mag_to_lum')
        
        if scatter_parameter is None:
            scatter_parameter=self.scatter_parameter
        if skew_parameter is None:
            skew_parameter=self.skew_parameter
        
        # calculate location parameter for distribution for the input halo
        # masses
        log_Q = self.calculate_Q(log_halo_mass, z, parameter)
    
        if self.scatter_name == 'lognormal':
            pdf_values =  norm.pdf(x     = log_quantity,
                                   loc   = log_Q,
                                   scale = scatter_parameter)
        elif self.scatter_name == 'skewlognormal':
            if skew_parameter is None:
                raise ValueError('skew_parameter must be specified for '
                                 'skewlognormal model.')
            # correction factor to create the desired central value
            correction_factor = self._skewlognormal_correction_factor(
                                        scatter_parameter, skew_parameter,
                                        skew_correction)
            pdf_values =  skewnorm.pdf(x     = log_quantity,
                                       a     = skew_parameter,
                                       loc   = log_Q + correction_factor,
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
        
    def halo_mass_marginal_density(self, log_halo_mass, z):
        '''
        Calculate value of HMF (which is marginal distribution for halo mass).
        '''
        hmf_values = np.power(10, self.model.calculate_log_hmf(
                                                log_halo_mass, z))
        return(hmf_values)
    
    def halo_mass_conditional_density(self, log_halo_mass, log_quantity, z, 
                                     parameter, scatter_parameter=None,
                                     skew_parameter=None,
                                     **kwargs):
        '''
        Calculate value of conditional density p(m_h|q) using Bayes' theorem.
        **kwargs are passed to quantity_marginal_density.
        '''
        if scatter_parameter is None:
            scatter_parameter=self.scatter_parameter
        if skew_parameter is None:
            skew_parameter=self.skew_parameter
        
        joint_dens         = self.joint_number_density(self, log_quantity, 
                                                       log_halo_mass, z, 
                                                       parameter, 
                                                       scatter_parameter,
                                                       skew_parameter)
        quantity_marg_dens = self.quantity_marginal_density(log_quantity, 
                                                            z, parameter,
                                                            scatter_parameter,
                                                            skew_parameter,
                                                            **kwargs)
        halo_cond_dens     = joint_dens/quantity_marg_dens
        return(halo_cond_dens)
        
    def calculate_Q(self, log_halo_mass, z, parameter):
        '''
        Calculate Q(m_h) for given halo mass, redshift and quantity - halo
        mass relation parameter. (Convenience method)
        '''
        log_Q = self.model.physics_model.at_z(z).calculate_log_quantity(
                                            log_halo_mass, *parameter)
        return(log_Q)
        
    def calculate_Q_inverse(self, log_quantity, z, parameter):  
        '''
        Calculate m_h from q using Q^(-1)(m_h) for given quantity value, 
        redshift and quantity - halo mass relation parameter. 
        (Convenience method)
        '''
        log_quantity = self.model.unit_conversion(log_quantity, 'mag_to_lum')
        log_m_h = self.model.physics_model.at_z(z).calculate_log_halo_mass(
                                            log_quantity, *parameter)
        return(log_m_h)
        
    
    def make_log_m_h_space(self, log_quantity, z, parameter, num=int(1e+3),
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
        log_m_epsilon     = log_Q_inv-epsilon
        # deal with to large values
        max_log_m_epsilon = self.model.log_max_halo_mass - epsilon
        log_m_epsilon[log_m_epsilon>max_log_m_epsilon] = max_log_m_epsilon

        # create logarithmically spaced points to the left and right of m_h
        lower_part = np.geomspace(epsilon, 
                                  log_m_epsilon-self.model.log_min_halo_mass,
                                  num)
        upper_part = np.geomspace(epsilon, 
                                  self.model.log_max_halo_mass-log_m_epsilon,
                                  num)
        
        # transform to m_h_space
        lower_part  = (log_m_epsilon - lower_part)[::-1]
        upper_part  = log_m_epsilon + upper_part
        
        # add middle section that connects two part at same sampling density
        distance  = upper_part[0] - lower_part[-1] 
        samp_dens = upper_part[1] - upper_part[0]
        points    = np.ceil(np.nanmax(distance[samp_dens>0]
                                      /samp_dens[samp_dens>0])).astype(int)
        if points >= num:
            warnings.warn('Number of points created for middle section '
                          'exceeds specified number of points in '
                          'make_log_m_h_space.')
        middle_part = np.linspace(lower_part[-1], upper_part[0], points,
                                  endpoint=False)
        
        # join spaces (sorted in order)
        log_m_h_space = np.concatenate([lower_part, middle_part[1:],
                                        upper_part])
        
        log_m_h_space[np.isnan(log_m_h_space)] = self.model.log_max_halo_mass
        
        # make arrays 2d, needed for correct broadcasting in
        # quantity_marginal_density method
        if log_m_h_space.ndim == 1:
            log_m_h_space = log_m_h_space[:, np.newaxis]
        return(log_m_h_space)
    
    def _skewlognormal_correction_factor(self, scatter_parameter=None, 
                                        skew_parameter=None, 
                                        statistic='median'):
        '''
        Correction factor for skewnormal distribution, since scale parameter
        does not coincide with either mean, median or mode of distribution.
        Choose one of the three statistics using statistic argument and 
        add to loc parameter to create distribution with desired central value.
        The default is median.
        '''
        
        if scatter_parameter is None:
            scatter_parameter=self.scatter_parameter
        if skew_parameter is None:
            skew_parameter=self.skew_parameter
        
        skewnormal_dist = skewnorm(a=skew_parameter, scale=scatter_parameter)
        
        if statistic == 'mean':
            correction_factor = -skewnormal_dist.mean()
        elif statistic == 'median':
            correction_factor = -skewnormal_dist.median()
        elif statistic == 'mode':
            warnings.warn('statistic=mode not properly tested.')
            # approximation taken from Wikipedia  
            delta   = scatter_parameter/np.sqrt(1+scatter_parameter**2)
            mu_z    = np.sqrt(2/np.pi)*delta
            sigma_z = np.sqrt(1-mu_z**2)
            gamma_1 = (4-np.pi)/2 * (mu_z**3/np.power(1-mu_z**2, 1.5)) 
            term_2  = gamma_1*sigma_z/2
            term_3  = (np.sign(skew_parameter)/2
                       * np.exp(2*np.pi/np.abs(skew_parameter)))                   
            correction_factor = skew_parameter*(mu_z - term_2 - term_3)
        else:
            raise NotImplementedError('statistic must be either mean, median '
                                      ' or mode')
        return(correction_factor)
    
    
    
    
    