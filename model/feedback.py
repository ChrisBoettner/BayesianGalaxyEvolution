#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:34:20 2022

@author: chris
"""
import numpy as np
from scipy.interpolate import interp1d

from model.helper import make_array, invert_function

def feedback_model(feedback_name, log_m_c, initial_guess=None, bounds=None):
    '''
    Return feedback model that relates SMF and HMF, including model function,
    model name, initial guess and physical parameter bounds for fitting,
    that related SMF and HMF.
    The model function parameters are left free and can be obtained via fitting.
    Three models implemented:
        none    : no feedback adjustment
        sn      : supernova feedback
        both    : supernova and black hole feedback
    '''
    if feedback_name == 'none':
        feedback = NoFeedback(log_m_c, initial_guess[:1],
                              bounds[:, :1])
    elif feedback_name == 'stellar':
        feedback = StellarFeedback(log_m_c, initial_guess[:-1],
                                   bounds[:, :-1])
    elif feedback_name == 'stellar_blackhole':
        feedback = StellarBlackholeFeedback(log_m_c, initial_guess,
                                            bounds)
    else:
        raise ValueError('quantity_name not known.')
    return(feedback)


class StellarBlackholeFeedback(object):
    def __init__(self, log_m_c, initial_guess, bounds):
        '''
        Model for stellar + black hole feedback.

        '''
        self.name = 'stellar_blackhole'
        self.log_m_c = log_m_c               # critical mass for feedback
        self.initial_guess = initial_guess   # initial guess for least_squares fit
        self.bounds = bounds                 # parameter (A, alpha, beta) bounds

        # max halo mass (just used to avoid overflows)
        self._upper_m_h = 50

    def calculate_log_quantity(self, log_m_h, log_A, log_m_c, alpha, beta):
        '''
        Calculate observable quantity from input halo mass and model parameter.
        '''
        if np.isnan(log_m_h).any():
            return(np.nan)
        log_m_h = self._check_overflow(log_m_h)
        
        sn, bh = self._variable_substitution(log_m_h, log_m_c, alpha, beta)
        log_quantity = np.empty_like(bh)
        
        # deal with bh becoming infinite sometimes
        inf_mask = np.isfinite(log_m_h) 
        sn       = sn[inf_mask]
        bh       = bh[inf_mask]
        
        log_quantity[inf_mask] = log_A + log_m_h[inf_mask] - np.log10(sn + bh)
        log_quantity[np.logical_not(inf_mask)] = np.inf
        return(log_quantity)

    def calculate_log_halo_mass(self, log_quantity, log_A, log_m_c, alpha,
                                beta, num = 500):
        '''
        Calculate halo mass from input quantity quantity and model parameter.
        Do this by calculating table of quantity for halo masses near critical
        mass and linearly interpolating inbetween (and extrapolating beyond
        range).
        num controls number of points that halo mass is initially calculated for,
        higher values make result more accurate but also more expensive.
        '''
        # create lookup tables of halo mass and quantities
        log_m_h_lookup = self._make_m_h_space(log_m_c, alpha, beta, num)
        log_quantity_lookup = StellarBlackholeFeedback.calculate_log_quantity(
            self, log_m_h_lookup, log_A, log_m_c, alpha, beta)
        
        # use lookup table to create callable spline
        log_m_h_func = interp1d(log_quantity_lookup, log_m_h_lookup,
                                bounds_error = False,
                                fill_value=([-np.inf],[np.inf]))
        log_m_h = log_m_h_func(log_quantity)
        
        log_m_h = make_array(log_m_h)
        log_m_h = self._check_overflow(log_m_h) # deal with very large m_h
        return(log_m_h)
    
    def calculate_log_halo_mass_alternative(self, log_quantity, log_A, log_m_c,
                                            alpha, beta, xtol=0.1):
        '''
        Calculate halo mass from input observable quantity and model paramter.
        This is done by numerical inverting the relation for the quantity as
        function of halo mass (making this a very expensive operation).
        Absolute tolerance for inversion can be adjusted using xtol.
        This method is more accurate, but more time consuming (also has
        problems if beta is close to 1).
        '''
        log_m_h = invert_function(func=self.calculate_log_quantity,
                                  fprime=self.calculate_dlogquantity_dlogmh,
                                  fprime2=self.calculate_d2logquantity_dlogmh2,
                                  x0_func=self._initial_guess,
                                  y=log_quantity,
                                  args=(log_A, log_m_c, alpha, beta),
                                  xtol=xtol)
        return(log_m_h)

    def calculate_dlogquantity_dlogmh(self, log_m_h, log_A, log_m_c, alpha, beta):
        '''
        Calculate d/d(log m_h) log_quantity. High mass end for beta near
        one treated as special case, where value apporaches zero.
        '''
        if np.isnan(log_m_h).any():
            return(np.nan)
        log_m_h = self._check_overflow(log_m_h)
            
        sn, bh = self._variable_substitution(log_m_h, log_m_c, alpha, beta)
        first_derivative = np.empty_like(bh)
        
        # deal with bh becoming infinite sometimes
        inf_mask = np.isfinite(bh) 
        sn       = sn[inf_mask]
        bh       = bh[inf_mask]
        
        first_derivative[inf_mask] = 1 - (-alpha * sn + beta * bh) / (sn + bh)
        first_derivative[np.logical_not(inf_mask)] = 0
        return(first_derivative)

    def calculate_d2logquantity_dlogmh2(self, log_m_h, log_A, log_m_c, alpha, beta):
        '''
        Calculate d^2/d(log m_h)^2 log_quantity. High mass end for beta near
        one treated as special case, where value apporaches zero.
        '''
        if np.isnan(log_m_h).any():
            return(np.nan)
        log_m_h = self._check_overflow(log_m_h)
        
        x = 10**((alpha+beta) * (log_m_h - log_m_c))
        second_derivative = np.empty_like(x)
        
        # deal with bh becoming infinite sometimes
        inf_mask    = np.isfinite(x) 
        numerator   = -np.log(10) * (alpha+beta)**2 * x[inf_mask]
        denominator = (x[inf_mask]+1)**2 
        
        second_derivative[inf_mask] = numerator/denominator
        second_derivative[np.logical_not(inf_mask)] = 0
        return(second_derivative)

    def _variable_substitution(self, log_m_h, log_m_c, alpha, beta):
        '''
        Transform input quanitities to more easily handable quantities.
        '''
        if np.isnan(log_m_h).any():
            return(np.nan)
        ratio = log_m_h - log_m_c  # m_h/m_c in log space

        log_stellar    = - alpha * ratio     # log of sn feedback contribution
        log_black_hole =    beta * ratio     # log of bh feedback contribution
        return(10**log_stellar, 10**log_black_hole)

    def _initial_guess(self, log_quantity, log_A, log_m_c, alpha, beta):
        '''
        Calculate initial guess for halo mass (for function inversion). Do this
        by calculating the high and low mass end approximation of the relation.
        '''
        # turnover quantity, where dominating feedback changes
        log_q_turn = log_A - np.log10(2) + log_m_c

        if log_quantity < log_q_turn:
            x0 = (log_quantity - log_A + alpha * log_m_c) / (1 + alpha)
        else:
            x0 = (log_quantity - log_A - beta * log_m_c) / (1 - beta)
        return(x0)
    
    def _check_overflow(self, log_m_h):
        '''
        Control for overflows by checking if halo mass exceeds upper limit.
        '''
        if np.any(log_m_h>self._upper_m_h):
            if np.isscalar(log_m_h):
                log_m_h = np.inf
            else:
                log_m_h[log_m_h>self._upper_m_h] = np.inf
        return(log_m_h)
    
    def _make_m_h_space(self, log_m_c, alpha, beta, num, epsilon = 0.001):
        '''
        Create array of m_h points arranged so that the points are dense where
        q(m_h) changes quickly and sparse where they aren't.
        num controls number of points that are calculated.
        epsilon controls how far out points are created (epsilon is the 
        ration between the strength of the two feedback types, i.e. 
        sn/bh = epsilon).

        '''
        # check relative contribution of feedbacks using sn = epsilon*bh to 
        # get limiting mass where either contribution drops below epsilon,
        # this is log_m_lim = -1/(alpha+beta) * log(epsilon) + log_m_c
        
        # mass where bh feedback dominanting
        log_m_bh = -1/(alpha+beta) * np.log10(epsilon) + log_m_c
        # mass where sn feedback dominanting
        log_m_sn =  1/(alpha+beta) * np.log10(epsilon) + log_m_c       
        
        # create high density space
        dense_log_m_h = np.linspace(log_m_sn, log_m_bh, int(num*0.8))
        # create low density spaces
        sparse_log_m_lower = np.linspace(log_m_sn/2, log_m_sn, int(num*0.1))
        sparse_log_m_upper = np.linspace(log_m_bh, 2*log_m_sn, int(num*0.1))
        
        log_m_h_table = np.sort(np.concatenate([dense_log_m_h, 
                                                sparse_log_m_lower,
                                                sparse_log_m_upper]))
        return(np.unique(log_m_h_table))
        

class StellarFeedback(StellarBlackholeFeedback):
    def __init__(self, log_m_c, initial_guess, bounds):
        '''
        Model for stellar feedback.
        '''
        super().__init__(log_m_c, initial_guess, bounds)
        self.name = 'stellar'

    def calculate_log_quantity(self, log_m_h, log_A, log_m_c, alpha):
        return(StellarBlackholeFeedback.calculate_log_quantity(self, log_m_h, log_A,
                                                               log_m_c,
                                                               alpha,
                                                               beta=0))

    def calculate_log_halo_mass(self, log_quantity, log_A, log_m_c, alpha,
                                xtol=0.1):
        log_m_h = invert_function(func=self.calculate_log_quantity,
                                  fprime=self.calculate_dlogquantity_dlogmh,
                                  fprime2=self.calculate_d2logquantity_dlogmh2,
                                  x0_func=self._initial_guess,
                                  y=log_quantity,
                                  args=(log_A, log_m_c, alpha),
                                  xtol=xtol)
        return(log_m_h)

    def calculate_dlogquantity_dlogmh(self, log_m_h, log_A, log_m_c, alpha):
        return(StellarBlackholeFeedback.calculate_dlogquantity_dlogmh(self, log_m_h,
                                                                      log_A, log_m_c,
                                                                      alpha,
                                                                      beta=0))

    def calculate_d2logquantity_dlogmh2(self, log_m_h, log_A, log_m_c, alpha):
        return(StellarBlackholeFeedback.calculate_d2logquantity_dlogmh2(self, log_m_h,
                                                                        log_A,
                                                                        log_m_c,
                                                                        alpha,
                                                                        beta=0))

    def _initial_guess(self, log_quantity, log_A, log_m_c, alpha):
        # turnover quantity, where dominating feedback changes
        log_q_turn = log_A - np.log10(2) + log_m_c

        if log_quantity < log_q_turn:
            x0 = (log_quantity - log_A + alpha * log_m_c) / (1 + alpha)
        else:
            x0 = (log_quantity - log_A)
        return(x0)


class NoFeedback(StellarBlackholeFeedback):
    def __init__(self, log_m_c, initial_guess, bounds):
        '''
        Model without feedback.
        '''
        super().__init__(log_m_c, initial_guess, bounds)
        self.name = 'none'
        self.log2 = np.log10(2) # so it doesn't need t be calc everytime
        
    def calculate_log_quantity(self, log_m_h, log_A):
        return(log_A - self.log2 + log_m_h)

    def calculate_log_halo_mass(self, log_quantity, log_A):
        return(log_quantity - log_A + self.log2)

    def calculate_dlogquantity_dlogmh(self, log_m_h, log_A):
        return(1)

    def calculate_d2logquantity_dlogmh2(self, log_m_h, log_A):
        return(0)
