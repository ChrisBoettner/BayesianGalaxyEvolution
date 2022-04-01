#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 11:49:01 2022

@author: chris
"""

import numpy as np
from scipy.optimize import root_scalar, curve_fit, dual_annealing
from scipy.spatial.distance import cdist, euclidean
from scipy.stats import gaussian_kde

from data_processing import load_mcmc_data, load_hmf_functions

## FRONT END CALLING FUNCTIONS
def calculate_schechter_parameter(model, observable, redshifts, num = int(1e+6),
                                  beta = 'zero'):
    '''
    Calculate the Schechter parameter for a given model (with given input array
    for observable and redshift). It does this by drawing random samples from
    the parameter distributions of the the model and using these to calculate
    the number density function (SMF/UVLF). A Schechter function is then fitted
    to this function. This is repeated so we get a Schechter parameter
    distribution. 
    Returns median, 16th percentile and upper error and 84th percentile,
    as well as full distribution for the parameters,
    for every redshift.
    '''
    parameter = []; lower_error = []; upper_error = []; dist = []
    # make input redshift, input compatible with integer and array
    if np.isscalar(redshifts):
        redshifts = np.array([redshifts])
    if np.isscalar(observable):
        redshifts = np.array([observable])
        
    for z in redshifts:
        schechter_params_at_z = []
        for n in range(num):
            # create model function
            A, alpha, beta       = model.get_parameter_sample(z)
            number_dens_function = model.number_density_function(observable, z, A, alpha, beta)
            if np.any(np.isnan(number_dens_function)):
                continue
            # remove tail end of model function where values are repeated bc HMF ends
            ndf = np.copy(number_dens_function[0])
            idx = np.argwhere(np.around(ndf/ndf[-1],4) == 1)[0][0] # index of first occurence of repeated value
            ndf = ndf[:idx]
            q   = observable[:idx] # cut input variable to same length for fitting
            
            # fit schechter function
            schechter_params,_   = curve_fit(log_schechter_function, np.log10(q),
                                             np.log10(ndf), p0 = [-3,11,-2],
                                             bounds = [[-np.inf,-np.inf,-np.inf],[0,np.inf,np.inf]],
                                             maxfev = 10000)
            schechter_params_at_z.append(schechter_params)
            
        schechter_params_at_z = np.array(schechter_params_at_z)
        dist.append(schechter_params_at_z)
        parameter.append(  np.percentile(schechter_params_at_z, 50, axis=0))  
        #parameter.append( geometric_median(schechter_params_at_z))
        lower_error.append(np.percentile(schechter_params_at_z, 16, axis=0))
        upper_error.append(np.percentile(schechter_params_at_z, 84, axis=0))
    return(np.array(parameter), np.array(lower_error), np.array(upper_error), dist)        


def calculate_halo_mass(model, observable, redshifts, num = int(1e+6), beta = 'zero'):
    '''
    Calculate distribution of halo_mass for every input observable (lum/m_star)
    in array observable. 
    For more details, see calculate_observable
    '''
    dist = []
    # make input redshift, input compatible with integer and array
    if np.isscalar(redshifts):
        redshifts = np.array([redshifts])
    if np.isscalar(observable):
        redshifts = np.array([observable])
    
    for z in redshifts:
        # calc values and save to array
        dist_at_z = []
        for o in observable:
            d = model.calculate_quantity_distribution(o, z, input_mode = 'observable',
                                                      num = num, 
                                                      beta_assumption = beta)            
            dist_at_z.append(d)
        dist.append(np.array(dist_at_z))
    return(dist)

def calculate_observable(model, m_halo, redshifts, num = int(1e+6), beta = 'zero'):
    '''
    Calculate distribution of observable (lum/m_star) for every halo mass in 
    array m_halo. 
    Can be used for single redshift or for array of redshifts, output will be
    list with every element consituting result for one redshift.
    
    Output arrays contains observable distribution (of size num) for every input
    halo mass.
    
    Two modes for how the beta parameter is treated for redshifts where only
    alpha could be fitted.
    'zero' : Assume beta is zero.
    'cont' : Assume beta has same distribution as in last redshift where it 
             could be estimated (z = 4 at the moment)
    '''
    dist = []
    # make input redshift, halo mass compatible with integer and array
    if np.isscalar(redshifts):
        redshifts = np.array([redshifts])
    if np.isscalar(m_halo):
        redshifts = np.array([m_halo])
    
    for z in redshifts:
        # calc values and save to array
        dist_at_z = []
        for m_h in m_halo:
            d = model.calculate_quantity_distribution(m_h, z, input_mode = 'halo_mass',
                                                      num = num, 
                                                      beta_assumption = beta)        
            dist_at_z.append(d)
        dist.append(np.array(dist_at_z))
    return(dist)

## MAIN MODEL
class model():
    '''
    Determined model that related halo mass to stellar mass and UV luminosity. 
    To select model, call function with either 'mstar' or 'lum'.
    '''
    def __init__(self, observable_name, m_c = 1e+12):
        self.observable_name = observable_name
        
        self.hmf             = dataset(load_hmf_functions())
        
        parameter, distribution = load_mcmc_data(observable_name)
        self.parameter     = dataset(parameter)
        #self.parameter      = DeprecationWarning('Use of parameter estimate not advised,\
        #                                          use distributions instead')
        self.distribution   = dataset(distribution)   
        
        self.feedback_model = feedback_model(m_c) 
        
        parameter_num       = np.array([d.shape[1] for d in distribution])
        self.parameter_num  = dataset(parameter_num)
        
    def number_density_function(self, observable, z, A, alpha, beta):
        '''
        Calculate the number density function (SMF/LF) for a given observed
        value, redshift and feedback model parameter. Can take array of feedback
        parameter samples (but not redshift).
        '''
        
        if np.isscalar(A):
            A = np.array([A]); alpha = np.array([alpha]); beta = np.array([beta])
        
        log_observable = np.log10(observable)
        
        number_density_func = []
        for i in range(len(A)):
            # calculate halo mass from observed value
            log_m_h = self.feedback_model.calculate_log_halo_mass(log_observable, A[i], alpha[i], beta[i])
            m_h     = np.power(10, log_m_h)
            
            hmf_value   = self.hmf.at_z(z)(m_h)
            model_value = self.feedback_model.calculate_dlogobservable_dlogmh(log_m_h, A[i], alpha[i], beta[i])
            
            number_density_func.append(hmf_value/model_value)
        return(number_density_func)
    
    def get_parameter_sample(self, z, num = 1, beta_assumption = 'zero'):
        '''
        Get a sample from feedback parameter distribution at given redshift.
        
        Two modes for how the beta parameter is treated for redshifts where only
        alpha could be fitted.
        'zero' : Assume beta is zero.
        'cont' : Assume beta has same distribution as in last redshift where it 
                 could be estimated (z = 4 at the moment)
        '''
        # randomly draw from parameter distribution at z 
        random_draw = np.random.choice(self.distribution.at_z(z).shape[0],
                                       size = num)
        parameter_draw = self.distribution.at_z(z)[random_draw]
        
        # seperate the values to easily call function
        if parameter_draw.shape[1] == 3: # meaning sn + bh feedback
            A       = parameter_draw[:,0]
            alpha   = parameter_draw[:,1]
            beta    = parameter_draw[:,2]
        if parameter_draw.shape[1]== 2: # just sn feedback
            A       = parameter_draw[:,0]
            alpha   = parameter_draw[:,1]
            if beta_assumption == 'zero':
                beta    = np.zeros(num)
            if beta_assumption == 'cont':
                # find last occurence where parameter number is 3, since that
                # should be last z where beta could be estimated
                ind = np.argmin(self.parameter_num.data)-1
                
                beta_ind =  np.random.choice(self.distribution.at_z(ind).shape[0],
                                               size = num)
                beta = self.distribution.at_z(ind)[:,-1][beta_ind]      
                
                
        # correct for the change in variables done in the fit originally    
        if self.observable_name == 'lum':
            A = A*1e+18
        return(A,alpha,beta)
        
        
    def calculate_quantity_distribution(self, inp, z, input_mode,
                                        num = int(1e+6), beta_assumption = 'zero'):
        '''
        input_mode = 'halo_mass' :  Calcuate observable (lum/m_star) for a given
                                    input halo mass and redshift. 
        input_mode = 'observable':  Calcuate halo mass for a given input
                                    observable (lum/m_star) and redshift. 
        To do this, random samples are drawn from the parameter distribution to
        calculate quantity distribution.
        
        Two modes for how the beta parameter is treated for redshifts where only
        alpha could be fitted.
        'zero' : Assume beta is zero.
        'cont' : Assume beta has same distribution as in last redshift where it 
                 could be estimated (z = 4 at the moment)
        '''
        
        if not np.isfinite(inp):
            return(np.array([np.nan]*num))
        
        inp = np.log10(inp)
    
        # randomly draw from parameter distribution at z 
        A, alpha, beta = self.get_parameter_sample(z, num, beta_assumption)

        # calculate quantity distribution using model function
        if input_mode == 'halo_mass':
            log_quantity_dist  = self.feedback_model.calculate_log_observable(inp, A, alpha, beta)
        if input_mode == 'observable':
            log_quantity_dist  = self.feedback_model.calculate_log_halo_mass(inp, A, alpha, beta)
        quantity_dist = np.power(10, log_quantity_dist)
        return(quantity_dist) 

## PHYSICAL MODELS 
class feedback_model():
    def __init__(self, m_c = 1e+12):
        self.m_c           = m_c
    def calculate_log_observable(self, log_m_h, A, alpha, beta):
        if np.isnan(log_m_h).any() or np.any(log_m_h<0) or np.any(log_m_h>20):
            return(np.nan)
        ratio = np.power(10, log_m_h - np.log10(self.m_c))
        sn = ratio**(-alpha)
        bh = ratio**beta
        obs = A * np.power(10, log_m_h)/(sn + bh)
        return(np.log10(obs))    
    def calculate_log_halo_mass(self, log_observable, A, alpha, beta):
        if (not np.isscalar(A)) and (not np.isscalar(log_observable)) and \
            (len(log_observable)>1) and (len(A)>1):
            raise ValueError('Either observable or parameter must be scalar values and not arrays.')
        if np.isscalar(A):
            A, alpha, beta = np.array([A]), np.array([alpha]), np.array([beta])
        
        log_m_halo = []
        for i in range(len(A)):
            log_m_h = invert_function(func    = self.calculate_log_observable,
                                      fprime  = self.calculate_dlogobservable_dlogmh,
                                      fprime2 = self.calculate_d2logobservable_dlogmh2,
                                      x0_func = self._initial_guess, 
                                      y       = log_observable,
                                      args    = (A[i], alpha[i], beta[i]))
            log_m_halo.append(log_m_h)           
        log_m_halo = np.array(log_m_halo)
        length = np.prod(log_m_halo.shape)
        log_m_halo = log_m_halo.reshape(length)
        return(log_m_halo)
    def calculate_dlogobservable_dlogmh(self, log_m_h, A, alpha, beta):
        if np.isnan(log_m_h).any() or np.any(log_m_h<0) or np.any(log_m_h>20):
            return(np.nan)
        ratio = np.power(10, log_m_h - np.log10(self.m_c))
        sn = ratio**(-alpha)
        bh = ratio**beta
        return(1 - (-alpha*sn + beta * bh)/(sn + bh))
    def calculate_d2logobservable_dlogmh2(self, log_m_h, A, alpha, beta): 
        if np.isnan(log_m_h).any() or np.any(log_m_h<0) or np.any(log_m_h>20):
            return(np.nan)
        ratio = np.power(10, log_m_h - np.log10(self.m_c))
        sn = ratio**(-alpha)
        bh = ratio**beta
        denom = (sn + bh)
        num_one = (alpha**2*sn + beta**2 * bh)
        num_two =(-alpha*sn + beta * bh)**2
        return(-np.log(10)*(num_one/denom+num_two/denom**2))
    def _initial_guess(self, log_observable, A, alpha, beta):
        # guess initial value for inverting function by using high and low mass
        # end approximation
        m_t          = A/2*self.m_c  # turnover mass where dominating feedback changes
        trans_regime = 20            # rough estimate for transient regime where both are important
        if np.power(10, log_observable) < m_t/trans_regime:
            x0 = np.power((self.m_c)**alpha/A*np.power(10, log_observable),1/(1+alpha))
        elif np.power(10, log_observable) > m_t*trans_regime:
            x0 = np.power((self.m_c)**(-beta)/A*np.power(10, log_observable),1/(1-beta))   
        else:
            x0 = np.power(10, log_observable)*2/A
        return(np.log10(x0))
    
def log_schechter_function(log_observable, log_phi_star, log_obs_star, alpha):
    '''
    Calculate the value of Schechter function log10(d(n)/dlog10(obs)) for an
    observable (in base10 log), using Schechter parameters.
    '''
    norm        = np.log10(np.log(10)) + log_phi_star
    power_law   = (alpha+1)*(log_observable-log_obs_star)
    exponential = - np.power(10,log_observable-log_obs_star)/np.log(10)
    return(norm + power_law + exponential)
        
    
def lum_to_sfr(L_nu):
    '''
    Converts between UV luminosity (in ergs s^-1 Hz^-1) and star formation rate
    (in solar masses yr^-1), using the Kennicutt relation.
    Kennicutt parameter chosen for Chabrier IMF and mass limits of 0.01 and 100 
    solar masses.
    See
    https://astronomy.stackexchange.com/questions/39806/relation-between-absolute-magnitude-of-uv-and-star-formation-rate
    '''
    C_Kennicutt = 7.8e-29
    return(C_Kennicutt*L_nu)

def lum_to_mag(L_nu):
    '''
    Converts between UV luminosity (in ergs s^-1 Hz^-1)) and absolute magnitude
    M_UV
    '''
    d    = 3.086e+19                # 10pc in cm
    flux = L_nu/(4*np.pi*d**2)
    M_uv = -2.5*np.log10(flux)-48.6 # definition in AB magnitude system
    return(M_uv)

## HELP FUNCTIONS  
class dataset():
    # easily retrieve data at certain redshift
    def __init__(self, data):
        self.data = data
    def at_z(self, redshift):
        return(self.data[redshift])

def invert_function(func, fprime, fprime2, x0_func, y, args):
    '''
    For a function y=f(x), calculate x values for an input set of y values.

    '''
    if np.isscalar(y):
        y = np.array([y])
    x = []     
    for val in y:
        def root_func(x,*args):
            return(func(x,*args)-val)
        
        x0 = x0_func(val, *args) # guess initial value
        
        root = root_scalar(root_func, fprime = fprime, fprime2=fprime2, args = args,
                            method='halley', x0 = x0, rtol=1e-6).root
        # if Halley's method doesn't work, try Newton
        if np.isnan(root):
                root = root_scalar(root_func, fprime = fprime, fprime2=fprime2, args = args,
                                    method='newton', x0 = x0, rtol=1e-6).root
        x.append(root)
    x = np.array(x)
    return(x)

def geometric_median(X, eps=1e-8):
    '''
    Geometric mean for multivariate distribution.
    From 
    https://stackoverflow.com/questions/30299267/geometric-median-of-multidimensional-points

    '''
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1
        y = y1

    
def find_mode(dist, lower_bound, upper_bound):
    '''
    Find Mode of distribution by approximating function using a Gaussian kernal
    denisity estimate and then minimizing that function using dual annealing.
    '''
    median = np.median(dist,axis=0)
    mode = []
    dist_func     = gaussian_kde(dist.T)
    dist_func_neg = lambda x: (-1)*dist_func(x) # negative, since we search for minimum
    bounds = list(zip(lower_bound,upper_bound))
    mode = dual_annealing(dist_func_neg, bounds, x0 = median).x
    return(mode)