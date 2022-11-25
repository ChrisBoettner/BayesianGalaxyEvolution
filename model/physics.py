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
                  fixed_m_c=True, eddington_erdf_params=None):
    '''
    Return physics model that relates SMF and HMF, including model function,
    model name, initial guess and physical parameter bounds for fitting,
    that related SMF and HMF.
    The model function parameters are left free and can be obtained via fitting.
    Three models implemented:
        none                : no feedback adjustment
        sn                  : supernova feedback
        both                : supernova and black hole feedback
        quasar              : black hole growth model
        eddington_free_ERDF : bolometric luminosity model using free ERDF
        eddington           : bolometric luminosity model using fixed ERDF
                              (need erdf parameter as additional input)
    '''
    if physics_name not in ['none', 'stellar', 'stellar_blackhole', 'quasar',
                            'eddington_free_m_c_free_ERDF',
                            'eddington_free_ERDF', 'eddington']:
        raise NameError('physics_name not known.')

    if physics_name == 'none':
        return(NoFeedback(log_m_c, initial_guess[:1],
                          bounds[:, :1]))

    if physics_name == 'eddington':
        if eddington_erdf_params is None:
            raise ValueError('Eddington model with fixed ERDF needs ERDF '
                             'parameter passed via eddington_erdf_params '
                             'argument.')
        return(QuasarLuminosity(log_m_c, initial_guess[:2], bounds[:, :2],
               eddington_erdf_params))

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
        elif physics_name == 'eddington_free_ERDF':
            physics = QuasarLuminosity_free_ERDF(log_m_c, initial_guess,
                                                 bounds)
        elif physics_name == 'eddington':
            if eddington_erdf_params is None:
                raise ValueError('Eddington model with fixed ERDF needs ERDF '
                                 'parameter passed via eddington_erdf_params '
                                 'argument.')
            physics = QuasarLuminosity(log_m_c, initial_guess[:-2], 
                                    bounds[:, :-2], eddington_erdf_params)

    else:
        # add log_m_c to initial guess and bounds
        initial_guess = np.insert(initial_guess, 0, log_m_c)
        log_m_c_range = 3  # +- range in which log_m_c is expected
        bounds = np.insert(bounds, 0, [10, log_m_c+log_m_c_range], axis=1)

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
        elif physics_name == 'eddington_free_ERDF':
            physics = QuasarLuminosity_free_m_c_free_ERDF(log_m_c,
                                                          initial_guess,
                                                          bounds)
        elif physics_name in 'eddington':
            raise NotImplementedError('physics model with free m_c but fixed '
                                      'ERDF not implemented.')
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
        # parameter (A, alpha, beta) bounds
        self.bounds = bounds
        
        # parameter used compared to the unconstrained model
        self.parameter_used = [0,1,2,3]
        self.fixed_m_c_flag = False

        # max halo mass (just used to avoid overflows)
        self._upper_m_h = 50
        # latest parameter used
        self._current_parameter = None
        self.log_m_h_function = None

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
        sn = sn[inf_mask]
        bh = bh[inf_mask]

        log_quantity[inf_mask] = log_A + log_m_h[inf_mask] - np.log10(sn + bh)
        log_quantity[np.logical_not(inf_mask)] = np.inf
        return(log_quantity)

    def calculate_log_halo_mass(self, log_quantity, log_m_c, log_A, alpha,
                                beta, num=500):
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
            log_m_h_lookup = self._make_m_h_space(log_m_c, log_A, alpha, 
                                                  beta, num=num)
            log_quantity_lookup = StellarBlackholeFeedback_free_m_c.\
                calculate_log_quantity(
                    self, log_m_h_lookup, log_m_c, log_A,
                    alpha, beta)

            # use lookup table to create callable spline
            self.log_m_h_function = interp1d(log_quantity_lookup,
                                             log_m_h_lookup,
                                             bounds_error=False,
                                             fill_value=([-np.inf], [np.inf]))

        log_m_h = self.log_m_h_function(log_quantity)
        log_m_h = make_array(log_m_h)
        log_m_h = self._check_overflow(log_m_h)  # deal with very large m_h
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
        sn = sn[inf_mask]
        bh = bh[inf_mask]

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
        inf_mask = np.isfinite(x)
        numerator = -np.log(10) * (alpha+beta)**2 * x[inf_mask]
        denominator = (x[inf_mask]+1)**2

        second_derivative[inf_mask] = numerator/denominator
        second_derivative[np.logical_not(inf_mask)] = 0
        return(second_derivative)

    def _variable_substitution(self, log_m_h, log_m_c, alpha, beta):
        '''
        Transform input quanitities to more easily handable quantities.
        '''
        log_m_h = self._check_overflow(log_m_h)

        ratio = log_m_h - log_m_c         # m_h/m_c in log space
        log_stellar = - alpha * ratio     # log of sn feedback contribution
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
            return(np.full_like(log_m_h, np.nan))
        if np.any(log_m_h > self._upper_m_h):
            if np.isscalar(log_m_h):
                log_m_h = np.inf
            else:
                log_m_h[log_m_h > self._upper_m_h] = np.inf
        return(log_m_h)
    
    def _calculate_feedback_regimes(self, log_m_c, log_A, alpha, beta, 
                                   log_epsilon=-2, output='halo_mass'):
        '''
        Calculate halo mass where one of the feedbacks is strongly dominating 
        in the sense that (M_h/M_c)^-alpha > epsilon * (M_h/M_c)^beta and 
        the other way around. Returns 3 values: [log_m_c, log_m_sn, log_m_ bh].
        If output='halo_mass' (default) return transition
        halo masses, if output='quantity' return quantity values (for this
        halo mass and parameter).
        '''
        # check relative contribution of feedbacks using sn = epsilon*bh to
        # get limiting mass where either contribution drops below epsilon,
        # this is log_m_lim = -1/(alpha+beta) * log(epsilon) + log_m_c
        
        # mass where sn feedback is dominating
        log_m_h_sn = 1/(alpha+beta)  * log_epsilon + log_m_c
        # mass where bh feedback is dominating
        log_m_h_bh = -1/(alpha+beta) * log_epsilon + log_m_c
        
        if output == 'halo_mass':
            return(np.array([log_m_c, log_m_h_bh, log_m_h_sn]))
        elif output == 'quantity':
            # calculate corresponding quantity values
            log_q_c = StellarBlackholeFeedback_free_m_c.calculate_log_quantity(
                                                   self, log_m_c, log_m_c, 
                                                   log_A, alpha, beta)[0]
            log_q_sn = StellarBlackholeFeedback_free_m_c.calculate_log_quantity(
                                                   self, log_m_h_sn, log_m_c, 
                                                   log_A, alpha, beta)[0]
            log_q_bh = StellarBlackholeFeedback_free_m_c.calculate_log_quantity(
                                                   self, log_m_h_bh, log_m_c, 
                                                   log_A, alpha, beta)[0]
            return(np.array([log_q_c, log_q_sn, log_q_bh]))
        else:
            raise ValueError('output must be either \'halo_mass\' or '
                             '\'quantity\'.')

    def _make_m_h_space(self, log_m_c, log_A, alpha, beta, num, log_epsilon=-3):
        '''
        Create array of m_h points arranged so that the points are dense where
        q(m_h) changes quickly and sparse where they aren't.
        num controls number of points that are calculated.
        epsilon controls how far out points are created (epsilon is the 
        ration between the strength of the two feedback types, i.e. 
        sn/bh = epsilon).

        '''
        
        # calculate halo masses where one of the feedbacks strongly dominates
        _, log_m_bh, log_m_sn = StellarBlackholeFeedback_free_m_c.\
                                 _calculate_feedback_regimes(
                                    self, log_m_c, log_A, alpha, beta,
                                    log_epsilon=log_epsilon)

        # create high density space
        dense_log_m_h = np.linspace(log_m_sn, log_m_bh, int(num*0.8))
        # create low density spaces
        sparse_log_m_lower = np.linspace(log_m_sn/2, log_m_sn, int(num*0.1))
        sparse_log_m_upper = np.linspace(log_m_bh, 2*log_m_bh, int(num*0.1))

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
        self.parameter_used = [0,1,2]
        self.fixed_m_c_flag = False

    def calculate_log_quantity(self, log_m_h, log_m_c, log_A, alpha):
        return(StellarBlackholeFeedback_free_m_c.
               calculate_log_quantity(self, log_m_h,
                                      log_m_c,
                                      log_A,
                                      alpha,
                                      beta=0))

    def calculate_log_halo_mass(self, log_quantity, log_m_c, log_A, alpha,
                                num=500):
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
    
    def _calculate_feedback_regimes(self, log_m_c, log_A, alpha,
                                   log_epsilon=-2, output='halo_mass'):
        return(StellarBlackholeFeedback_free_m_c.
               _calculate_feedback_regimes(self, log_m_c,
                                                log_A,
                                                alpha,
                                                beta=0,
                                                log_epsilon=log_epsilon,
                                                output=output))


class QuasarGrowth_free_m_c(StellarBlackholeFeedback_free_m_c):
    def __init__(self, log_m_c, initial_guess, bounds):
        '''
        Black hole growth model with free m_c.

        '''
        super().__init__(log_m_c, initial_guess, bounds)
        self.name = 'quasargrowth_free_m_c'
        self.parameter_used = [0,1,2]
        self.fixed_m_c_flag = False

    def calculate_log_quantity(self, log_m_h, log_m_c, log_B, eta):
        '''
        Calculate (log of) bolometric luminosity from input halo mass, 
        model parameter and chosen log_eddington_ratio.
        '''
        log_m_h = make_array(log_m_h)
        return(log_B + eta*(log_m_h - log_m_c))

    def calculate_log_halo_mass(self, log_quantity, log_m_c, log_B,
                                eta):
        '''
        Calculate (log of) halo mass from input bolometric luminosity, 
        model parameter and chosen log_eddington_ratio.
        '''
        log_quantity = make_array(log_quantity)
        return((log_quantity-log_B)/eta + log_m_c)

    def calculate_dlogquantity_dlogmh(self, log_m_h, log_m_c,
                                      log_B, eta):
        '''
        Calculate d/d(log m_h) log_quantity.
        '''
        return(np.full_like(log_m_h, eta))

class QuasarLuminosity_free_m_c_free_ERDF(object):
    def __init__(self, log_m_c, initial_guess, bounds):
        '''
        Black hole bolometric luminosity model with free m_c and ERDF.

        '''
        self.name = 'eddington_free_m_c_free_ERDF'
        self.initial_guess = initial_guess   # initial guess for least_squares
        # fit
        self.bounds = bounds                 # parameter bounds

        self.log_m_c = log_m_c
        
        # parameter used compared to the unconstrained model
        self.parameter_used = [0,1,2,3,4]
        self.fixed_m_c_flag = False

        # latest parameter used
        self.parameter = None
        self.eddington_distribution = None

    def calculate_log_quantity(self, log_m_h, log_eddington_ratio, log_m_c,
                               log_C, eta):
        '''
        Calculate (log of) bolometric luminosity from input halo mass, 
        model parameter and chosen log_eddington_ratio.
        '''
        log_m_h = make_array(log_m_h)
        return(log_eddington_ratio + log_C + eta*(log_m_h - log_m_c))

    def calculate_log_halo_mass(self, log_L, log_eddington_ratio, log_m_c,
                                log_C, eta):
        '''
        Calculate (log of) halo mass from input bolometric luminosity, 
        model parameter and chosen log_eddington_ratio.
        '''
        log_L = make_array(log_L)
        return((log_L-(log_eddington_ratio + log_C))/eta + log_m_c)

    def calculate_dlogquantity_dlogmh(self, log_m_h, log_eddington_ratio,
                                      log_m_c, log_C, eta):
        '''
        Calculate d/d(log m_h) log_quantity.
        '''
        return(np.full_like(log_m_h, eta))

    def calculate_log_erdf(self, log_eddington_ratio, log_eddington_star,
                           rho):
        '''
        Calculate (log) value of ERDF (probability of given eddington ratio), 
        given input log_eddingtion_ratio and parameter (log_eddington_star, 
        rho).

        '''
        self._make_distribution(log_eddington_star, rho)
        log_erdf = self.eddington_distribution.log_probability(
            log_eddington_ratio)
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
        if self.parameter == [log_eddington_star, rho]:
            pass
        else:
            self.parameter = [log_eddington_star, rho]
            self.eddington_distribution = ERDF(log_eddington_star, rho)
        return(self.eddington_distribution)


################ MODEL WITH FIXED CRITICAL MASS ###############################

class StellarBlackholeFeedback(StellarBlackholeFeedback_free_m_c):
    def __init__(self, log_m_c, initial_guess, bounds):
        '''
        Model for stellar + black hole feedback.

        '''
        super().__init__(log_m_c, initial_guess, bounds)
        self.name = 'stellar_blackhole'
        self.parameter_used = [1,2,3]
        self.fixed_m_c_flag = True

    def calculate_log_quantity(self, log_m_h, log_A, alpha, beta):
        return(StellarBlackholeFeedback_free_m_c.
               calculate_log_quantity(self, log_m_h,
                                      log_m_c=self.log_m_c,
                                      log_A=log_A,
                                      alpha=alpha,
                                      beta=beta))

    def calculate_log_halo_mass(self, log_quantity, log_A, alpha, beta,
                                num=500):
        return(StellarBlackholeFeedback_free_m_c.
               calculate_log_halo_mass(self, log_quantity,
                                       log_m_c=self.log_m_c,
                                       log_A=log_A,
                                       alpha=alpha,
                                       beta=beta,
                                       num=num))

    def calculate_dlogquantity_dlogmh(self, log_m_h, log_A, alpha, beta):
        return(StellarBlackholeFeedback_free_m_c.
               calculate_dlogquantity_dlogmh(self, log_m_h,
                                             log_m_c=self.log_m_c,
                                             log_A=log_A,
                                             alpha=alpha,
                                             beta=beta))

    def calculate_d2logquantity_dlogmh2(self, log_m_h, log_A, alpha, beta):
        return(StellarBlackholeFeedback_free_m_c.
               calculate_d2logquantity_dlogmh2(self, log_m_h,
                                               log_m_c=self.log_m_c,
                                               log_A=log_A,
                                               alpha=alpha,
                                               beta=beta))
    
    def _calculate_feedback_regimes(self, log_A, alpha, beta, log_epsilon=-2, 
                                   output='halo_mass'):
        return(StellarBlackholeFeedback_free_m_c.
               _calculate_feedback_regimes(self, 
                                          log_m_c=self.log_m_c,
                                          log_A=log_A,
                                          alpha=alpha,
                                          beta=beta,
                                          log_epsilon=log_epsilon,
                                          output=output))


class StellarFeedback(StellarBlackholeFeedback):
    def __init__(self, log_m_c, initial_guess, bounds):
        '''
        Model for stellar feedback.
        '''
        super().__init__(log_m_c, initial_guess, bounds)
        self.name = 'stellar'
        self.parameter_used = [1,2]
        self.fixed_m_c_flag = True

    def calculate_log_quantity(self, log_m_h, log_A, alpha):
        return(StellarBlackholeFeedback_free_m_c.
               calculate_log_quantity(self, log_m_h,
                                      log_m_c=self.log_m_c,
                                      log_A=log_A,
                                      alpha=alpha,
                                      beta=0))

    def calculate_log_halo_mass(self, log_quantity, log_A, alpha, num=500):
        return(StellarBlackholeFeedback_free_m_c.
               calculate_log_halo_mass(self, log_quantity,
                                       log_m_c=self.log_m_c,
                                       log_A=log_A,
                                       alpha=alpha,
                                       beta=0,
                                       num=num))

    def calculate_dlogquantity_dlogmh(self, log_m_h, log_A, alpha):
        return(StellarBlackholeFeedback_free_m_c.
               calculate_dlogquantity_dlogmh(self, log_m_h,
                                             log_m_c=self.log_m_c,
                                             log_A=log_A,
                                             alpha=alpha,
                                             beta=0))

    def calculate_d2logquantity_dlogmh2(self, log_m_h, log_A, alpha):
        return(StellarBlackholeFeedback_free_m_c.
               calculate_d2logquantity_dlogmh2(self, log_m_h,
                                               log_m_c=self.log_m_c,
                                               log_A=log_A,
                                               alpha=alpha,
                                               beta=0))
    
    def _calculate_feedback_regimes(self, log_A, alpha, log_epsilon=-2, 
                                   output='halo_mass'):
        return(StellarBlackholeFeedback_free_m_c.
               _calculate_feedback_regimes(self, log_m_c=self.log_m_c,
                                          log_A=log_A,
                                          alpha=alpha,
                                          beta=0,
                                          log_epsilon=log_epsilon,
                                          output=output))

class NoFeedback(StellarBlackholeFeedback):
    def __init__(self, log_m_c, initial_guess, bounds):
        '''
        Model without feedback.
        '''
        super().__init__(log_m_c, initial_guess, bounds)
        self.name = 'none'
        self.log2 = np.log10(2)  # so it doesn't need to be calc everytime

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
        self.name = 'quasargrowth'
        self.parameter_used = [1,2]
        self.fixed_m_c_flag = True

    def calculate_log_quantity(self, log_m_h, log_B, eta):
        return(QuasarGrowth_free_m_c.
               calculate_log_quantity(self, log_m_h,
                                      log_m_c=self.log_m_c,
                                      log_B=log_B,
                                      eta=eta))

    def calculate_log_halo_mass(self, log_quantity, log_B, eta):
        return(QuasarGrowth_free_m_c.
               calculate_log_halo_mass(self, log_quantity,
                                       log_m_c=self.log_m_c,
                                       log_B=log_B,
                                       eta=eta))

    def calculate_dlogquantity_dlogmh(self, log_m_h, log_B, eta):
        return(QuasarGrowth_free_m_c.
               calculate_dlogquantity_dlogmh(self, log_m_h,
                                             log_m_c=self.log_m_c,
                                             log_B=log_B,
                                             eta=eta))

class QuasarLuminosity_free_ERDF(QuasarLuminosity_free_m_c_free_ERDF):
    def __init__(self, log_m_c, initial_guess, bounds):
        '''
        Black hole bolometric luminosity model with free ERDF.

        '''
        self.name = 'eddington_free_ERDF'
        self.initial_guess = initial_guess   # initial guess for least_squares
        # fit
        self.bounds = bounds                 # parameter bounds

        self.log_m_c = log_m_c
        
        # parameter used compared to the unconstrained model
        self.parameter_used = [1,2,3,4]
        self.fixed_m_c_flag = True

        # latest parameter used
        self.parameter = None
        self.eddington_distribution = None

    def calculate_log_quantity(self, log_m_h, log_eddington_ratio, 
                               log_C, eta):
        return(QuasarLuminosity_free_m_c_free_ERDF.calculate_log_quantity(
                                self,
                                log_m_h=log_m_h,
                                log_eddington_ratio=log_eddington_ratio,
                                log_m_c=self.log_m_c,
                                log_C=log_C,
                                eta=eta))

    def calculate_log_halo_mass(self, log_L, log_eddington_ratio, 
                                log_C, eta):
        return(QuasarLuminosity_free_m_c_free_ERDF.calculate_log_halo_mass(
                                self,
                                log_L=log_L,
                                log_eddington_ratio=log_eddington_ratio,
                                log_m_c=self.log_m_c,
                                log_C=log_C,
                                eta=eta))
    
    def calculate_dlogquantity_dlogmh(self, log_m_h, log_eddington_ratio,
                                      log_C, eta):
        return(QuasarLuminosity_free_m_c_free_ERDF.calculate_dlogquantity_dlogmh(
                                self,
                                log_m_h=log_m_h,
                                log_eddington_ratio=log_eddington_ratio,
                                log_m_c=self.log_m_c,
                                log_C=log_C,
                                eta=eta))
    

class QuasarLuminosity(QuasarLuminosity_free_ERDF):
    def __init__(self, log_m_c, initial_guess, bounds, eddington_erdf_params):
        '''
        Black hole bolometric luminosity model with fixed ERDF.

        '''
        self.name = 'eddington'
        self.initial_guess = initial_guess   # initial guess for least_squares
        # fit
        self.bounds = bounds                 # parameter bounds

        self.log_m_c = log_m_c
        
        self.parameter_used = [1,2]
        self.fixed_m_c_flag = True

        # parameter used
        self.parameter = eddington_erdf_params
        self.eddington_distribution = ERDF(*self.parameter)

    def calculate_log_erdf(self, log_eddington_ratio):
        log_erdf = self.eddington_distribution.log_probability(
            log_eddington_ratio)
        return(log_erdf)

    def calculate_mean_log_eddington_ratio(self):
        mean = self.eddington_distribution.mean()
        return(mean)

    def draw_eddington_ratio(self, num=1):
        log_eddington_ratio = self.eddington_distribution.rvs(size=num)
        return(log_eddington_ratio)

    def _make_distribution(self, log_eddington_star, rho):
        raise NotImplementedError('Cannot make new distribution for fixed ERDF'
                                  ' model.')
