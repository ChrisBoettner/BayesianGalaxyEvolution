#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 12:43:36 2022

@author: chris
"""
import numpy as np
from scipy.stats import rv_continuous
from scipy.special import hyp2f1

class ERDF(rv_continuous):
    '''
    Scipy implementation of continous probability distribution for 
    Eddington Rate Distribution Function.
    
    '''
    def __init__(self, eddington_star, rho):
        super().__init__(a = 0, b = 1)       # domain of Eddington Ratio
        
        # define parameters
        self.eddington_star = eddington_star
        self.rho            = rho
    
        # normalisation: integrate unnormalized erdf from 0 to 1 (analytical
        # result)
        self.normalisation   = 1/hyp2f1(1, 1/self.rho, 1+1/self.rho,
                                 -1/self.eddington_star**self.rho)
    
    def _pdf(self, eddington_ratio):
        '''
        Calculate pdf.
        
        '''
        # assume no super-Eddington accretion
        if (eddington_ratio<0) or (eddington_ratio>1):
            erdf = 0
        else:
            x = (eddington_ratio/self.eddington_star)**self.rho
            erdf = self.normalisation * 1/(1+x) 
        return(erdf)
    
    def _cdf(self, eddington_ratio):
        '''
        Calculate cdf.
        
        '''
        if eddington_ratio<0:
            cdf = 0
        elif eddington_ratio>1:
            cdf = 1
        else:        
            value = eddington_ratio * hyp2f1(1, 1/self.rho, 1+1/self.rho, 
                                             -(eddington_ratio/
                                               self.eddington_star)**self.rho)
            cdf = self.normalisation * value
        return(cdf)