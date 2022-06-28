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
        Characteristic value of the function, where the power law starts to 
        take effect.
    rho : float
        The power law slope.
    log_threshold : float, optional
        The value (in log space) of the power laws term before 
        approximation of pure power law is used. The default is 10.
    a : float
        The lower end of the support of the distribution (in log space). Values 
        of the pdf for Eddington ratios below this are zero. The default is 
        -inf.
    b : float
        The upper end of the support of the distribution (in log space). Values 
        of the pdf for Eddington ratios above this are zero. The default is 
        inf.

    '''

    def __init__(self, log_eddington_star, rho, log_threshold=3,
                 a=-np.inf, b=np.inf):
        super().__init__(a=a, b=b)  # domain of Eddington Ratio,
        # values outside of domain are
        # 0
        # define parameters
        self.log_eddington_star = log_eddington_star
        self.rho = rho

        self.log_threshold = log_threshold

        # normalisation: integrate unnormalized erdf from 0 to
        # upper bound of domain (analytical result)
        normalisation = 1/calculate_limit(self._unnormalized_cdf,
                                          self.log_eddington_star+5)
        self.log_normalisation = np.log10(normalisation)

    def _pdf(self, log_eddington_ratio):
        '''
        Calculate pdf.

        '''
        return(np.power(10, self.log_probability(log_eddington_ratio)))

    def log_probability(self, log_eddington_ratio):
        '''
        Calculate (log of) pdf. For very large differences in exponent, 
        calculate approximate value by ignoring one of the power laws.

        '''
        # variable subsitution
        x = make_array(log_eddington_ratio - self.log_eddington_star)
        exponent = self.rho*x

        # check where approximation can be used
        power_law_mask = (exponent > self.log_threshold)
        if np.all(power_law_mask):
            log_erdf = self.log_normalisation - exponent

        else:
            log_erdf = np.empty_like(x)
            # if value of power law term is very large, ignore the + 1 and
            # treat as if it was power law directly
            log_erdf[power_law_mask] = (self.log_normalisation
                                        - exponent[power_law_mask])

            # otherwise calculate value properly
            inverse_mask = np.logical_not(power_law_mask)
            power_law = np.power(10, self.rho*x[inverse_mask])
            log_erdf[inverse_mask] = (self.log_normalisation
                                      - np.log10(1 + power_law))

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
        x = log_eddington_ratio - self.log_eddington_star
        exponent = self.rho*x

        # calculate hypergeometrix function for final quantity
        hyper_geo = hyp2f1(1, 1/self.rho, 1+1/self.rho,
                           -np.power(10, exponent))
        return(np.power(10, log_eddington_ratio)*hyper_geo)
