#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:34:20 2022

@author: chris
"""
import numpy as np

from model.helper import make_list, invert_function


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
        feedback = StellarFeedback(log_m_c, initial_guess[:2],
                                   bounds[:, :2])
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
        self._upper_m_h = 40

    def calculate_log_quantity(self, log_m_h, log_A, alpha, beta):
        '''
        Calculate observable quantity from input halo mass and model parameter.
        '''
        if np.isnan(log_m_h).any() or (log_m_h > self._upper_m_h).any():
            return(np.nan)
        sn, bh = self._variable_substitution(log_m_h, alpha, beta)
        log_quantity = log_A + log_m_h - np.log10(sn + bh)
        return(log_quantity)

    def calculate_log_halo_mass(self, log_quantity, log_A, alpha, beta,
                                xtol=0.1):
        '''
        Calculate halo mass from input observable quantity and model paramter.
        This is done by numerical inverting the relation for the quantity as
        function of halo mass (making this a very expensive operation).
        Absolute tolerance for inversion can be adjusted using xtol.
        '''
        log_m_h = invert_function(func=self.calculate_log_quantity,
                                  fprime=self.calculate_dlogquantity_dlogmh,
                                  fprime2=self.calculate_d2logquantity_dlogmh2,
                                  x0_func=self._initial_guess,
                                  y=log_quantity,
                                  args=(log_A, alpha, beta),
                                  xtol=xtol)
        return(log_m_h)

    def calculate_log_halo_mass_alternative(self, log_quantity, log_A, alpha,
                                            beta,
                                            spacing=1e-3):
        '''
        Calculate halo mass from input quantity quantity and model parameter.
        Do this by creating a lookup table for quantity-halo mass relation
        for the input paramter and then look for clostest value in that table
        (faster, but less accurate than the default version).
        Increasing spacing makes result more accurate but takes longer to
        calculate.
        '''
        print('Warning: This method does currently not produce correct results.')

        # create lookup tables of halo mass and quantities
        log_m_h_lookup = np.arange(8, 17, spacing)
        log_quantity_lookup = StellarBlackholeFeedback.calculate_log_quantity(
            self, log_m_h_lookup, log_A, alpha, beta)

        log_quantity = make_list(log_quantity)
        # find closest values to log_quantity in lookup array
        idx = np.searchsorted(log_quantity_lookup, log_quantity)
        # if value larger than any value in reference array, use largest value
        idx[idx == len(log_m_h_lookup)] -= 1

        log_m_h = log_m_h_lookup[idx]

        # for consistent behaviour with other methods
        if len(log_m_h) == 1:
            log_m_h = log_m_h[0]
        return(log_m_h)

    def calculate_dlogquantity_dlogmh(self, log_m_h, log_A, alpha, beta):
        '''
        Calculate d/d(log m_h) log_quantity
        '''
        if np.isnan(log_m_h).any() or (log_m_h > self._upper_m_h).any():
            return(np.nan)
        sn, bh = self._variable_substitution(log_m_h, alpha, beta)

        first_derivative = 1 - (-alpha * sn + beta * bh) / (sn + bh)
        return(first_derivative)

    def calculate_d2logquantity_dlogmh2(self, log_m_h, log_A, alpha, beta):
        '''
        Calculate d^2/d(log m_h)^2 log_quantity
        '''

        if np.isnan(log_m_h).any() or (log_m_h > self._upper_m_h).any():
            return(np.nan)
        sn, bh = self._variable_substitution(log_m_h, alpha, beta)

        denominator = (sn + bh)
        numerator_one = (alpha**2 * sn + beta**2 * bh)
        numerator_two = (-alpha * sn + beta * bh)**2

        second_derivative = -np.log(10) * (numerator_one / denominator
                                           + numerator_two / denominator**2)
        return(second_derivative)

    def _variable_substitution(self, log_m_h, alpha, beta):
        '''
        Transform input quanitities to more easily handable quantities.
        '''
        if np.isnan(log_m_h).any():
            return(np.nan)
        ratio = log_m_h - self.log_m_c  # m_h/m_c in log space

        log_stellar = - alpha * ratio        # log of sn feedback contribution
        log_black_hole = beta * ratio        # log of bh feedback contribution
        return(10**log_stellar, 10**log_black_hole)

    def _initial_guess(self, log_quantity, log_A, alpha, beta):
        '''
        Calculate initial guess for halo mass (for function inversion). Do this
        by calculating the high and low mass end approximation of the relation.
        '''
        # turnover quantity, where dominating feedback changes
        log_q_turn = log_A - np.log10(2) + self.log_m_c

        if log_quantity < log_q_turn:
            x0 = (log_quantity - log_A + alpha * self.log_m_c) / (1 + alpha)
        else:
            x0 = (log_quantity - log_A - beta * self.log_m_c) / (1 - beta)
        return(x0)


class StellarFeedback(StellarBlackholeFeedback):
    def __init__(self, log_m_c, initial_guess, bounds):
        '''
        Model for stellar feedback.
        '''
        super().__init__(log_m_c, initial_guess, bounds)
        self.name = 'stellar'

    def calculate_log_quantity(self, log_m_h, log_A, alpha):
        return(StellarBlackholeFeedback.calculate_log_quantity(self, log_m_h, log_A,
                                                               alpha,
                                                               beta=0))

    def calculate_log_halo_mass(self, log_quantity, log_A, alpha,
                                xtol=0.1):
        log_m_h = invert_function(func=self.calculate_log_quantity,
                                  fprime=self.calculate_dlogquantity_dlogmh,
                                  fprime2=self.calculate_d2logquantity_dlogmh2,
                                  x0_func=self._initial_guess,
                                  y=log_quantity,
                                  args=(log_A, alpha),
                                  xtol=xtol)
        return(log_m_h)

    def calculate_dlogquantity_dlogmh(self, log_m_h, log_A, alpha):
        return(StellarBlackholeFeedback.calculate_dlogquantity_dlogmh(self, log_m_h,
                                                                      log_A, alpha,
                                                                      beta=0))

    def calculate_d2logquantity_dlogmh2(self, log_m_h, log_A, alpha):
        return(StellarBlackholeFeedback.calculate_d2logquantity_dlogmh2(self, log_m_h,
                                                                        log_A,
                                                                        alpha,
                                                                        beta=0))

    def _initial_guess(self, log_quantity, log_A, alpha):
        # turnover quantity, where dominating feedback changes
        log_q_turn = log_A - np.log10(2) + self.log_m_c

        if log_quantity < log_q_turn:
            x0 = (log_quantity - log_A + alpha * self.log_m_c) / (1 + alpha)
        else:
            x0 = (log_quantity - log_A)
        return(x0)


class NoFeedback(StellarBlackholeFeedback):
    def __init__(self, log_m_c, initial_guess, bounds):
        '''
        Model for without feedback.
        '''
        super().__init__(log_m_c, initial_guess, bounds)
        self.name = 'none'

    def calculate_log_quantity(self, log_m_h, log_A):
        return(StellarBlackholeFeedback.calculate_log_quantity(self, log_m_h, log_A,
                                                               alpha=0,
                                                               beta=0))

    def calculate_log_halo_mass(self, log_quantity, log_A):
        return(log_quantity - log_A + np.log10(2))

    def calculate_dlogquantity_dlogmh(self, log_m_h, log_A):
        return(StellarBlackholeFeedback.calculate_dlogquantity_dlogmh(self, log_m_h,
                                                                      log_A,
                                                                      alpha=0,
                                                                      beta=0))

    def calculate_d2logquantity_dlogmh2(self, log_m_h, log_A):
        return(StellarBlackholeFeedback.calculate_d2logquantity_dlogmh2(self,
                                                                        log_m_h,
                                                                        log_A,
                                                                        alpha=0,
                                                                        beta=0))
