#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 12:43:36 2022

@author: chris
"""
from scipy.stats import rv_continuous
from scipy.special import hyp2f1

#from model.helper import make_array

class ERDF(rv_continuous):
    '''
    Scipy implementation of continous probability distribution for 
    Eddington Rate Distribution Function.
    
    '''
    def __init__(self, eddington_star, rho):
        super().__init__(a = 0, b = 1)       # domain of Eddington Ratio,
                                             # values outside of domain are
                                             # 0
        # define parameters
        self.eddington_star = eddington_star
        self.rho            = rho
    
        # normalisation: integrate unnormalized erdf from 0 to 
        # upper bound of domain (analytical result)
        self.normalisation   = 1/(self.b*hyp2f1(1, 1/self.rho, 1+1/self.rho,
                                 -(self.b/self.eddington_star)**self.rho))
    
    def _pdf(self, eddington_ratio):
        '''
        Calculate pdf.
        
        '''
        x = (eddington_ratio/self.eddington_star)**self.rho
        erdf = self.normalisation * 1/(1+x) 
        return(erdf)
    
    def _cdf(self, eddington_ratio):
        '''
        Calculate cdf.
        
        '''
        value = eddington_ratio * hyp2f1(1, 1/self.rho, 1+1/self.rho, 
                                         -(eddington_ratio/
                                           self.eddington_star)**self.rho)
        cdf = self.normalisation * value
        return(cdf)