#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:34:20 2022

@author: chris
"""
import numpy as np
from scipy.interpolate import interp1d

from model.eddington import ERDF
from model.helper import make_array, invert_function

def physics_model(physics_name, log_m_c, initial_guess, bounds,
                   fixed_m_c=True):
    '''
    Return physics model that relates SMF and HMF, including model function,
    model name, initial guess and physical parameter bounds for fitting,
    that related SMF and HMF.
    The model function parameters are left free and can be obtained via fitting.
    Three models implemented:
        none      : no feedback adjustment
        sn        : supernova feedback
        both      : supernova and black hole feedback
        quasar    : black hole growth model
        eddington  : eddigtion ratio distribution function model
    '''
    if physics_name not in ['none', 'stellar', 'stellar_blackhole', 'quasar',
                            'eddington']:
       raise NameError('physics_name not known.')     
    
    if physics_name == 'none':
            return(NoFeedback(log_m_c, initial_guess[:1],
                                  bounds[:, :1]))
    if physics_name == 'eddington':
        return(QuasarLuminosity(log_m_c, initial_guess, bounds))
                
    if fixed_m_c:
        if physics_name == 'stellar':
            physics = StellarFeedback(log_m_c, initial_guess[:-1],
                                       bounds[:, :-1])
        elif physics_name == 'stellar_blackhole':
            physics = StellarBlackholeFeedback(log_m_c, initial_guess,
                                                bounds)
        elif physics_name == 'quasar':
            physics = QuasarGrowth(log_m_c, initial_guess,
                                   bounds)
        elif physics_name == 'eddington':
            physics = QuasarLuminosity(log_m_c, initial_guess, bounds)
            
    else:
        # add log_m_c to initial guess and bounds
        initial_guess  = np.insert(initial_guess, 0, log_m_c)
        log_m_c_range  = 3 # +- range in which log_m_c is expected
        bounds         = np.insert(bounds, 0, 
                                   [log_m_c-log_m_c_range,
                                    log_m_c+log_m_c_range],
                                   axis=1)
        
        if physics_name == 'stellar':
            physics = StellarFeedback_free_m_c(log_m_c, 
                                                initial_guess[:-1],
                                                bounds[:, :-1])
        elif physics_name == 'stellar_blackhole':
            physics = StellarBlackholeFeedback_free_m_c(log_m_c,
                                                         initial_guess,
                                                         bounds)     
        elif physics_name == 'quasar':
            physics = QuasarGrowth_free_m_c(log_m_c, initial_guess,
                                            bounds)
    return(physics)

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
        if alpha == beta == 0:
            return(log_quantity - log_A + np.log10(2))
        
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
    
class QuasarGrowth_free_m_c(object):
    def __init__(self, log_m_c, initial_guess, bounds):
        '''
        Black hole growth model with free m_c.

        '''
        self.name = 'quasargrowth_free_m_c'
        self.log_m_c = log_m_c               # critical mass for model
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
        log_quantity = make_array(log_quantity)
        ## variable substitution
        log_x       = np.full_like(log_quantity, np.nan)
        # check when the -1 can be ignored and calculate approximation
        mask        = ((log_quantity - log_A)>10)
        log_x[mask] = log_quantity[mask] - log_A
        # if -1 cannot be ignored, calculate value properly
        inv_mask = np.logical_not(mask)
        # deal with infinities and negative values, and regime where model
        # breaks down
        _x       = np.power(10, log_quantity[inv_mask] - log_A) - 1
        log_x[_x>=0] = np.log10(_x[_x>=0])
        
        
        # calculate halo mass
        log_m_h                  = np.empty_like(log_x)
        log_m_h[np.isnan(log_x)] = -np.inf
        log_m_h[np.isfinite(log_x)] = 1/gamma * log_x[np.isfinite(log_x)] + log_m_c
        
        log_m_h = make_array(log_m_h)
        log_m_h = self._check_overflow(log_m_h) # deal with very large m_h
        return(log_m_h)
    
    def calculate_dlogquantity_dlogmh(self, log_m_h, log_m_c, log_A, gamma):
        '''
        Calculate d/d(log m_h) log_quantity.
        '''
        log_m_h = make_array(log_m_h)
        log_m_h = self._check_overflow(log_m_h)
            
        x = self._variable_substitution(log_m_h, log_m_c, gamma)
        
        # deal with values where model breaks down
        first_derivative = np.empty_like(log_m_h)
        mask             = log_m_h<=0
        first_derivative[mask] = np.inf # so that phi will be = 0
        
        #calculate first derivative
        first_derivative[np.logical_not(mask)] = 1/(1+
                                                 1/x[np.logical_not(mask)])*\
                                                    gamma
        return(first_derivative)

    def _variable_substitution(self, log_m_h, log_m_c, gamma):
        '''
        Transform input quanitities to more easily handable quantities.
        '''
        log_m_h = self._check_overflow(log_m_h)
        
        ratio    = log_m_h - log_m_c  # m_h/m_c in log space
        log_x    = gamma * ratio
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
    
    
class QuasarLuminosity(object):
    def __init__(self, log_m_c, initial_guess, bounds):
        '''
        Black hole bolometric luminosity model with free e_star.
        
        '''
        self.name = 'quasarluminosity'
        self.initial_guess = initial_guess   # initial guess for least_squares 
                                             # fit
        self.bounds = bounds                 # parameter bounds
        
        self.log_m_c = log_m_c
        
        # latest parameter used
        self._current_parameter       = None
        self.eddington_distribution   = None
        
    def calculate_log_quantity(self, log_m_h, log_eddington_ratio, log_C, 
                               gamma):
        '''
        Calculate (log of) bolometric luminosity from input halo mass, 
        model parameter and chosen log_eddington_ratio.
        '''
        log_m_h = make_array(log_m_h)
        return(log_eddington_ratio + log_C + gamma*(log_m_h - self.log_m_c))
        
    def calculate_log_halo_mass(self, log_L, log_eddington_ratio, log_C, 
                                 gamma):
        '''
        Calculate (log of) halo mass from input bolometric luminosity, 
        model parameter and chosen log_eddington_ratio.
        '''
        log_L = make_array(log_L)
        return((log_L-(log_eddington_ratio + log_C))/gamma + self.log_m_c)
    
    def calculate_dlogquantity_dlogmh(self, log_m_h, log_eddington_ratio, 
                                      log_C, gamma):
        '''
        Calculate d/d(log m_h) log_quantity.
        '''
        return(np.full_like(log_m_h, gamma))
        
    def calculate_log_erdf(self, log_eddington_ratio, log_eddington_star, 
                           rho):
        '''
        Calculate (log) value of ERDF (probability of given eddington ratio), 
        given input log_eddingtion_ratio and parameter (log_eddington_star, 
        rho).

        '''
        self._make_distribution(log_eddington_star, rho)
        log_erdf = self.eddington_distribution.log_probability(log_eddington_ratio)
        return(log_erdf)
    
    def calculate_mean_log_eddington_ratio(self, log_eddington_star, 
                                           rho):
        '''
        Calculate mean eddington ratio for the given parameter.

        '''
        self._make_distribution(log_eddington_star, rho)
        mean = self.eddington_distribution.mean()
        return(mean)
    
    def draw_eddington_ratio(self, log_eddington_star, rho, num=1):
        '''
        Draw random sample of Eddingtion ratio from distribution defined by
        parameter (log_eddington_star, rho). Can draw num samples at 
        once.

        '''
        self._make_distribution(log_eddington_star, rho)
        log_eddington_ratio = self.eddington_distribution.rvs(size=num)
        return(log_eddington_ratio)
    
    def _make_distribution(self, log_eddington_star, rho):
        '''
        Check if distribution function with the given parameter was already
        created and stored, otherwise create it.

        '''
        if self._current_parameter == [log_eddington_star, rho]:
            pass
        else:
            self._current_parameter     = [log_eddington_star, rho]
            self.eddington_distribution = ERDF(log_eddington_star, rho)
        return(self.eddington_distribution)
    
    
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
    
    
class QuasarGrowth(QuasarGrowth_free_m_c):
    def __init__(self, log_m_c, initial_guess, bounds):
        '''
        Black hole growth model.

        '''
        super().__init__(log_m_c, initial_guess, bounds)

    def calculate_log_quantity(self, log_m_h, log_A, gamma):
        return(QuasarGrowth_free_m_c.
                   calculate_log_quantity(self, log_m_h,
                                          log_m_c = self.log_m_c,
                                          log_A   = log_A,
                                          gamma   = gamma))
    
    def calculate_log_halo_mass(self, log_quantity, log_A, gamma):
        return(QuasarGrowth_free_m_c.
                   calculate_log_halo_mass(self, log_quantity, 
                                           log_m_c = self.log_m_c,
                                           log_A = log_A,
                                           gamma = gamma))

    def calculate_dlogquantity_dlogmh(self, log_m_h, log_A, gamma):
        return(QuasarGrowth_free_m_c.
                   calculate_dlogquantity_dlogmh(self, log_m_h,
                                                 log_m_c = self.log_m_c,
                                                 log_A   = log_A,
                                                 gamma   = gamma))