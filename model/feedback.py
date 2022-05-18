#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:34:20 2022

@author: chris
"""
import numpy as np
from scipy.interpolate import interp1d

from model.helper import make_array, invert_function

def feedback_model(feedback_name, log_m_c, initial_guess, bounds,
                   fixed_m_c=True):
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
    if feedback_name not in ['none', 'stellar', 'stellar_blackhole', 'quasar']:
       raise ValueError('feedback_name not known.')     
    
    if feedback_name == 'none':
            return(NoFeedback(log_m_c, initial_guess[:1],
                                  bounds[:, :1]))
                
    if fixed_m_c:
        if feedback_name == 'stellar':
            feedback = StellarFeedback(log_m_c, initial_guess[:-1],
                                       bounds[:, :-1])
        elif feedback_name == 'stellar_blackhole':
            feedback = StellarBlackholeFeedback(log_m_c, initial_guess,
                                                bounds)
        elif feedback_name == 'quasar':
            feedback = QuasarFeedback(log_m_c, initial_guess,
                                                bounds)
            
    else:
        # add log_m_c to initial guess and bounds
        initial_guess  = np.insert(initial_guess, 0, log_m_c)
        log_m_c_range  = 3 # +- range in which log_m_c is expected
        bounds         = np.insert(bounds, 0, 
                                   [log_m_c-log_m_c_range,
                                    log_m_c+log_m_c_range],
                                   axis=1)
        
        if feedback_name == 'stellar':
            feedback = StellarFeedback_free_m_c(log_m_c, 
                                                initial_guess[:-1],
                                                bounds[:, :-1])
        elif feedback_name == 'stellar_blackhole':
            feedback = StellarBlackholeFeedback_free_m_c(log_m_c,
                                                         initial_guess,
                                                         bounds)     
        elif feedback_name == 'quasar':
            feedback = QuasarFeedback_free_m_c(log_m_c, initial_guess,
                                               bounds)
    return(feedback)

################ MODEL WITH FREE CRITICAL MASS ################################


class StellarBlackholeFeedback_free_m_c(object):
    def __init__(self, log_m_c, initial_guess, bounds):
        '''
        Model for stellar + black hole feedback with free m_c.

        '''
        self.name = 'stellar_blackhole_free_m_c'
        self.log_m_c = log_m_c               # critical mass for feedback
        self.initial_guess = initial_guess   # initial guess for least_squares 
                                             # fit
        self.bounds = bounds                 # parameter (A, alpha, beta) bounds
        
        # max halo mass (just used to avoid overflows)
        self._upper_m_h = 50
        # latest parameter used
        self._current_parameter = None
        self.log_m_h_function   = None

    def calculate_log_quantity(self, log_m_h, log_m_c, log_A, alpha, beta):
        '''
        Calculate observable quantity from input halo mass and model parameter.
        '''
        log_m_h = make_array(log_m_h)
        log_m_h = self._check_overflow(log_m_h)
        
        sn, bh = self._variable_substitution(log_m_h, log_m_c, alpha, beta)
        log_quantity = np.empty_like(bh)
        
        # deal with bh becoming infinite sometimes
        inf_mask = np.isfinite(sn+bh) 
        sn       = sn[inf_mask]
        bh       = bh[inf_mask]
        
        log_quantity[inf_mask] = log_A + log_m_h[inf_mask] - np.log10(sn + bh)
        log_quantity[np.logical_not(inf_mask)] = np.inf
        return(log_quantity)

    def calculate_log_halo_mass(self, log_quantity, log_m_c, log_A, alpha,
                                beta, num = 500):
        '''
        Calculate halo mass from input quantity quantity and model parameter.
        Do this by calculating table of quantity for halo masses near critical
        mass and linearly interpolating inbetween (and extrapolating beyond
        range). Once interpolated function is created, it's saved and called 
        for new calculations until new parameter are passed.
        num controls number of points that halo mass is initially calculated for,
        higher values make result more accurate but also more expensive.

        '''
        if self._current_parameter == [log_m_c, log_A, alpha, beta]:
            pass
        else:
            self._current_parameter = [log_m_c, log_A, alpha, beta]
            # create lookup tables of halo mass and quantities
            log_m_h_lookup = self._make_m_h_space(log_m_c, alpha, beta, num)
            log_quantity_lookup = StellarBlackholeFeedback_free_m_c.\
                                  calculate_log_quantity(
                                  self, log_m_h_lookup, log_m_c, log_A,
                                  alpha, beta)
            
            # use lookup table to create callable spline
            self.log_m_h_function = interp1d(log_quantity_lookup, 
                                             log_m_h_lookup,
                                             bounds_error = False,
                                             fill_value=([-np.inf],[np.inf]))
        
        log_m_h = self.log_m_h_function(log_quantity)
        log_m_h = make_array(log_m_h)
        log_m_h = self._check_overflow(log_m_h) # deal with very large m_h
        return(log_m_h)
    
    def calculate_log_halo_mass_alternative(self, log_quantity, log_m_c, log_A,
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
                                  args=(log_m_c, log_A, alpha, beta),
                                  xtol=xtol)
        return(log_m_h)

    def calculate_dlogquantity_dlogmh(self, log_m_h, log_m_c, log_A, alpha, 
                                      beta):
        '''
        Calculate d/d(log m_h) log_quantity. High mass end for beta near
        one treated as special case, where value apporaches zero.
        '''
        log_m_h = make_array(log_m_h)
        log_m_h = self._check_overflow(log_m_h)
            
        sn, bh = self._variable_substitution(log_m_h, log_m_c, alpha, beta)
        first_derivative = np.empty_like(bh)
        
        # deal with bh becoming infinite sometimes
        inf_mask = np.isfinite(bh+sn) 
        sn       = sn[inf_mask]
        bh       = bh[inf_mask]
        
        first_derivative[inf_mask] = 1 - (-alpha * sn + beta * bh) / (sn + bh)
        first_derivative[np.logical_not(inf_mask)] = 0
        return(first_derivative)

    def calculate_d2logquantity_dlogmh2(self, log_m_h, log_m_c, log_A, alpha, 
                                        beta):
        '''
        Calculate d^2/d(log m_h)^2 log_quantity. High mass end for beta near
        one treated as special case, where value apporaches zero.
        '''
        log_m_h = make_array(log_m_h)
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
        log_m_h = self._check_overflow(log_m_h)
        
        ratio = log_m_h - log_m_c       # m_h/m_c in log space
        log_stellar    = - alpha * ratio     # log of sn feedback contribution
        if beta == 0:
            log_black_hole = np.full_like(log_stellar, 0)
        else:
            log_black_hole = beta * ratio     # log of bh feedback contribution
        return(10**log_stellar, 10**log_black_hole)

    def _initial_guess(self, log_quantity, log_m_c, log_A, alpha, beta):
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
        Also deal with Nans.
        '''
        if np.isnan(log_m_h).any():
            return(np.full_like(log_m_h,np.nan))
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
        
        # mass where bh feedback strongly dominating
        log_m_bh = -1/(alpha+beta) * np.log10(epsilon) + log_m_c
        # mass where sn feedback strongly dominating
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
    
    
class StellarFeedback_free_m_c(StellarBlackholeFeedback_free_m_c):
    def __init__(self, log_m_c, initial_guess, bounds):
        '''
        Model for stellar feedback with free m_c.
        '''
        super().__init__(log_m_c, initial_guess, bounds)
        self.name = 'stellar_free_m_c'

    def calculate_log_quantity(self, log_m_h, log_m_c, log_A, alpha):
        return(StellarBlackholeFeedback_free_m_c.
                   calculate_log_quantity(self, log_m_h, 
                                          log_m_c, 
                                          log_A, 
                                          alpha,
                                          beta=0))
    
    def calculate_log_halo_mass(self, log_quantity, log_m_c, log_A, alpha,
                                num = 500):
        return(StellarBlackholeFeedback_free_m_c.
                   calculate_log_halo_mass(self, log_quantity,
                                           log_m_c,
                                           log_A,
                                           alpha, 
                                           beta=0, 
                                           num=num))

    def calculate_dlogquantity_dlogmh(self, log_m_h, log_m_c, log_A, alpha):
        return(StellarBlackholeFeedback_free_m_c.
                   calculate_dlogquantity_dlogmh(self, log_m_h, 
                                                 log_m_c, 
                                                 log_A,
                                                 alpha, 
                                                 beta=0))

    def calculate_d2logquantity_dlogmh2(self, log_m_h, log_m_c, log_A, alpha):
        return(StellarBlackholeFeedback_free_m_c.
                   calculate_d2logquantity_dlogmh2(self, log_m_h, 
                                                   log_m_c, 
                                                   log_A,
                                                   alpha,
                                                   beta=0))
    
    
class QuasarFeedback_free_m_c(object):
    def __init__(self, log_m_c, initial_guess, bounds):
        '''
        Feedback model for Black Hole growth with free m_c.

        '''
        self.name = 'quasar_free_m_c'
        self.log_m_c = log_m_c               # critical mass for feedback
        self.initial_guess = initial_guess   # initial guess for least_squares 
                                             # fit
        self.bounds = bounds                 # parameter (A, gamma) bounds
        
        # max halo mass (just used to avoid overflows)
        self._upper_m_h = 50
        # latest parameter used
        self._current_parameter = None
        self.log_m_h_function   = None

    def calculate_log_quantity(self, log_m_h, log_m_c, log_A, gamma):
        '''
        Calculate observable quantity from input halo mass and model parameter.
        '''
        log_m_h = make_array(log_m_h)
        log_m_h = self._check_overflow(log_m_h)       
        
        x            = self._variable_substitution(log_m_h, log_m_c, gamma)
        log_quantity = log_A + np.log10(1 + x) 
        return(log_quantity)

    def calculate_log_halo_mass(self, log_quantity, log_m_c, log_A, gamma):
        '''
        Calculate halo mass from input observable quantity and model parameter.
        '''
        
        x = np.power(10, log_quantity - log_A) - 1
        
        # deal with infinities
        log_m_h  = np.empty_like(x)
        inf_mask = x==0
        
        log_m_h[inf_mask]                 = -np.inf
        log_m_h[np.logical_not(inf_mask)] = 1/gamma *\
                                            np.log10(x[np.logical_not(inf_mask)]) +\
                                            log_m_c
        
        log_m_h = make_array(log_m_h)
        log_m_h = self._check_overflow(log_m_h) # deal with very large m_h
        return(log_m_h)
    
    def calculate_dlogquantity_dlogmh(self, log_m_h, log_m_c, log_A, gamma):
        '''
        Calculate d/d(log m_h) log_quantity. High mass end for beta near
        one treated as special case, where value apporaches zero.
        '''
        log_m_h = make_array(log_m_h)
        log_m_h = self._check_overflow(log_m_h)
            
        x = self._variable_substitution(log_m_h, log_m_c, gamma)
        first_derivative = 1/(1+1/x) * gamma
        return(first_derivative)

    def _variable_substitution(self, log_m_h, log_m_c, gamma):
        '''
        Transform input quanitities to more easily handable quantities.
        '''
        log_m_h = self._check_overflow(log_m_h)
        
        ratio    = log_m_h - log_m_c  # m_h/m_c in log space
        log_x    = gamma * ratio      # log of sn feedback contribution
        return(10**log_x)
    
    def _check_overflow(self, log_m_h):
        '''
        Control for overflows by checking if halo mass exceeds upper limit.
        Also deal with Nans.
        '''
        if np.isnan(log_m_h).any():
            return(np.full_like(log_m_h,np.nan))
        if np.any(log_m_h>self._upper_m_h):
            if np.isscalar(log_m_h):
                log_m_h = np.inf
            else:
                log_m_h[log_m_h>self._upper_m_h] = np.inf
        return(log_m_h)
    
    
################ MODEL WITH FIXED CRITICAL MASS ###############################

class StellarBlackholeFeedback(StellarBlackholeFeedback_free_m_c):
    def __init__(self, log_m_c, initial_guess, bounds):
        '''
        Model for stellar + black hole feedback.

        '''
        super().__init__(log_m_c, initial_guess, bounds)

    def calculate_log_quantity(self, log_m_h, log_A, alpha, beta):
        return(StellarBlackholeFeedback_free_m_c.
                   calculate_log_quantity(self, log_m_h,
                                          log_m_c = self.log_m_c,
                                          log_A = log_A,
                                          alpha = alpha,
                                          beta  = beta))
    
    def calculate_log_halo_mass(self, log_quantity, log_A, alpha, beta,
                                num = 500):
        return(StellarBlackholeFeedback_free_m_c.
                   calculate_log_halo_mass(self, log_quantity, 
                                           log_m_c = self.log_m_c,
                                           log_A = log_A,
                                           alpha = alpha,
                                           beta  = beta,
                                           num=num))

    def calculate_dlogquantity_dlogmh(self, log_m_h, log_A, alpha, beta):
        return(StellarBlackholeFeedback_free_m_c.
                   calculate_dlogquantity_dlogmh(self, log_m_h,
                                                 log_m_c = self.log_m_c,
                                                 log_A = log_A,
                                                 alpha = alpha,
                                                 beta  = beta))

    def calculate_d2logquantity_dlogmh2(self, log_m_h, log_A, alpha, beta):
        return(StellarBlackholeFeedback_free_m_c.
                   calculate_d2logquantity_dlogmh2(self, log_m_h, 
                                                   log_m_c = self.log_m_c,
                                                   log_A = log_A,
                                                   alpha = alpha,
                                                   beta  = beta))
        

class StellarFeedback(StellarBlackholeFeedback):
    def __init__(self, log_m_c, initial_guess, bounds):
        '''
        Model for stellar feedback.
        '''
        super().__init__(log_m_c, initial_guess, bounds)
        self.name = 'stellar'

    def calculate_log_quantity(self, log_m_h, log_A, alpha):
        return(StellarBlackholeFeedback_free_m_c.
                   calculate_log_quantity(self, log_m_h,
                                          log_m_c = self.log_m_c,
                                          log_A = log_A,
                                          alpha = alpha,
                                          beta  = 0))
    
    def calculate_log_halo_mass(self, log_quantity, log_A, alpha, num = 500):
        return(StellarBlackholeFeedback_free_m_c.
                   calculate_log_halo_mass(self, log_quantity, 
                                           log_m_c = self.log_m_c,
                                           log_A = log_A,
                                           alpha = alpha,
                                           beta  = 0,
                                           num=num))

    def calculate_dlogquantity_dlogmh(self, log_m_h, log_A, alpha):
        return(StellarBlackholeFeedback_free_m_c.
                   calculate_dlogquantity_dlogmh(self, log_m_h,
                                                 log_m_c = self.log_m_c,
                                                 log_A = log_A,
                                                 alpha = alpha,
                                                 beta  = 0))

    def calculate_d2logquantity_dlogmh2(self, log_m_h, log_A, alpha):
        return(StellarBlackholeFeedback_free_m_c.
                   calculate_d2logquantity_dlogmh2(self, log_m_h, 
                                                   log_m_c = self.log_m_c,
                                                   log_A = log_A,
                                                   alpha = alpha,
                                                   beta  = 0))


class NoFeedback(StellarBlackholeFeedback):
    def __init__(self, log_m_c, initial_guess, bounds):
        '''
        Model without feedback.
        '''
        super().__init__(log_m_c, initial_guess, bounds)
        self.name = 'none'
        self.log2 = np.log10(2) # so it doesn't need to be calc everytime
        
    def calculate_log_quantity(self, log_m_h, log_A):
        '''
        Calculate observable quantity from input halo mass and model parameter.
        '''
        log_m_h = make_array(log_m_h)
        return(log_A - self.log2 + log_m_h)

    def calculate_log_halo_mass(self, log_quantity, log_A):
        '''
        Calculate halo mass from input observable quantity and model parameter.
        '''
        log_quantity = make_array(log_quantity)
        return(log_quantity - log_A + self.log2)

    def calculate_dlogquantity_dlogmh(self, log_m_h, log_A):
        log_m_h = make_array(log_m_h)
        return(np.full_like(log_m_h, 1))

    def calculate_d2logquantity_dlogmh2(self, log_m_h, log_A):
        log_m_h = make_array(log_m_h)
        return(np.full_like(log_m_h, 0))
    
    
class QuasarFeedback(QuasarFeedback_free_m_c):
    def __init__(self, log_m_c, initial_guess, bounds):
        '''
        Feedback model for Black Hole growth.

        '''
        super().__init__(log_m_c, initial_guess, bounds)

    def calculate_log_quantity(self, log_m_h, log_A, gamma):
        return(QuasarFeedback_free_m_c.
                   calculate_log_quantity(self, log_m_h,
                                          log_m_c = self.log_m_c,
                                          log_A   = log_A,
                                          gamma   = gamma))
    
    def calculate_log_halo_mass(self, log_quantity, log_A, gamma):
        return(QuasarFeedback_free_m_c.
                   calculate_log_halo_mass(self, log_quantity, 
                                           log_m_c = self.log_m_c,
                                           log_A = log_A,
                                           gamma = gamma))

    def calculate_dlogquantity_dlogmh(self, log_m_h, log_A, gamma):
        return(QuasarFeedback_free_m_c.
                   calculate_dlogquantity_dlogmh(self, log_m_h,
                                                 log_m_c = self.log_m_c,
                                                 log_A   = log_A,
                                                 gamma   = gamma))