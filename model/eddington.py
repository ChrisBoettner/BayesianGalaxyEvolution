#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 12:43:36 2022

@author: chris
"""
import numpy as np
from scipy.stats import rv_continuous
from scipy.special import hyp2f1

from model.helper import calculate_limit, make_array

class ERDF(rv_continuous):
    '''
    Scipy implementation of continous probability distribution for 
    Eddington Rate Distribution Function.
    When one of the power laws strongly dominates, calculate approximate 
    value by ignoring the other power law. Threshold when approximation is used
    can be adjusted using log_threshold. The support of the distribution can be
    changed using a and b.
    
    Parameters
    ----------
    log_eddington_star : float
        Characteristic value of the function, where the power law slope switches.
    rho_1 : float
        The first power law slope.
    rho_2 : TYPE
        The second power law slope.
    log_threshold : float, optional
        The difference (in log space) between the power laws before 
        approximation of one power law is used. The default is 10.
    a : float
        The lower end of the support of the distribution (in log space). Values 
        of the pdf for Eddington ratios below this are zero. The default is -11.
    b : float
        The upper end of the support of the distribution (in log space). Values 
        of the pdf for Eddington ratios above this are zero. The default is inf.
        
    '''
    
    def __init__(self, log_eddington_star, rho_1, rho_2, log_threshold=10,
                 a = -11, b = np.inf):
        super().__init__(a = a, b = b) # domain of Eddington Ratio,
                                       # values outside of domain are
                                       # 0
        # define parameters
        self.log_eddington_star = log_eddington_star
        self.rho_1              = rho_1
        self.rho_2              = rho_2
        
        self.log_threshold      = log_threshold
    
        # normalisation: integrate unnormalized erdf from 0 to 
        # upper bound of domain (analytical result)
        normalisation   = 1/calculate_limit(self._unnormalized_cdf, 
                                           self.log_eddington_star+10)
        self.log_normalisation = np.log10(normalisation)
        #self.log_normalisation = 1
    
    def _pdf(self, log_eddington_ratio):
        '''
        Calculate pdf.
        
        '''
        return(np.power(10,self.log_probability(log_eddington_ratio)))
        
        
    def log_probability(self, log_eddington_ratio):
        '''
        Calculate (log of) pdf. For very large differences in exponent, 
        calculate approximate value by ignoring one of the power laws.
        
        '''
        
        # variable subsitution
        x           = make_array(log_eddington_ratio - self.log_eddington_star)
        exponent_1  = self.rho_1*x
        exponent_2  = self.rho_2*x
        
        log_erdf   = np.empty_like(x)
        
        # if one of the two power laws dominates strongly, use approximation
        # to calculate value by ignoring the other value
        flag_1 = ((exponent_1 - exponent_2) > self.log_threshold)
        flag_2 = ((exponent_2 - exponent_1) > self.log_threshold)
        log_erdf[flag_1] = self.log_normalisation - exponent_1[flag_1]
        log_erdf[flag_2] = self.log_normalisation - exponent_2[flag_2]
        
        # otherwise calculate value properly
        flag_3           = np.logical_not(flag_1+flag_2)
        power_law_1      = np.power(10, self.rho_1*x[flag_3])
        power_law_2      = np.power(10, self.rho_2*x[flag_3])
        log_erdf[flag_3] = self.log_normalisation - np.log10(power_law_1
                                                             + power_law_2)
        
        if np.isscalar(log_eddington_ratio):
            return(log_erdf[0])
        else:
            return(log_erdf)
    
    def _cdf(self, log_eddington_ratio):
        '''
        Calculate cdf.
        
        '''
        return(10**self.log_normalisation
               * self._unnormalized_cdf(log_eddington_ratio))
    
    def _unnormalized_cdf(self, log_eddington_ratio):
        '''
        Calculate unnormalized cdf. (Value of antiderivative at 
        log_eddington_ratio - Value at lower bound of support a)
        
        '''
        return(self._antiderivative(log_eddington_ratio)
               - self._antiderivative(self.a))
        
    
    def _antiderivative(self, log_eddington_ratio):
        '''
        Calculate antiderivative of pdf (analytical solution).
        
        '''
        # variable substitutions
        x             = np.power(10, float(log_eddington_ratio 
                                     - self.log_eddington_star))
        exponent_diff = self.rho_1 - self.rho_2
        q             = self.rho_2/exponent_diff
        
        # calculate components for final quantity
        power_law    = x**(-self.rho_2)
        hyper_geo    = hyp2f1(1, -q, 1-q, -x**(exponent_diff))
        
        return(-power_law*hyper_geo/(np.log(10)*self.rho_2))
        
    