# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 21:34:38 2021

@author: boettner
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from astropy.cosmology import Planck18

import leastsq_fitting
import mcmc_fitting
#import mcmc_fit_test as mcmc_fitting

## DATA CONTAINER
class model_container():
    '''
    Container object that contains all important information about modelled
    UVLF.
    '''
    def __init__(self, lfs, hmfs, feedback_name, fitting_method, prior_name, mode):
        parameter, modelled_lf, distribution, lf_model = fit_LF_model(lfs, hmfs, feedback_name,
                                                                      fitting_method, prior_name, mode)
        self.feedback_name = feedback_name
        
        self.parameter     = lf_data(parameter)
        self.lf            = lf_data(modelled_lf)
        self.distribution  = lf_data(distribution)
        
        self.model         = lf_data(lf_model)
        
    def plot_parameter(self, color, marker, linestyle, label):
        # style parameter for fit
        self.color     = color
        self.marker    = marker
        self.linestyle = linestyle
        self.label     = label
        return(self)
 
class lf_data():
    # easily retrieve data at certain redshift
    def __init__(self, data):
        self.data = data
    def at_z(self, redshift):
        return(self.data[redshift])

## MAIN FUNCTIONS   
def fit_LF_model(lfs, hmfs, feedback_name,
                 fitting_method, prior_name, mode):
    '''
    Perform LF fit for all redshifts.
    
    feedback_name can either be a single string (then this model is used for all
    redshifts) or a list of strings corresponding to the model used for each redshift.
    
    Choose between 3 different prior model:
        uniform  : assume uniform prior within bounds for each redshift
        marginal : use marginalized distribution for parameter from previous 
                   redshift (assuming independence of parameter)
        full     : use full distribution for parameter from previous 
                   redshift (assuming dependence of parameter)
    
    IMPORTANT : Critical luminosity preset to 
    '''
    
    parameter = []; modelled_lf = []; distribution = []; lf_models = []
    posterior_samp = None
    bounds         = None
    
    # do fits in different order (for successive prior):
    # start at z=4 since we have a lot of data there and still use sn + bh model, 
    # evolve that to redshift 10. then use distribution at 4 again and evolve backwards
    # to z=0
    for z in [4,5,6,7,8,9,10,3,2,1,0]:
        print(z)

        lf  = np.copy(lfs[z])
        hmf = np.copy(hmfs[z])
        
        lf[:,0] = lf[:,0]/1e+18 # change units for luminosities for numerical stability

        m_c = 1e+12
        # calc m_crit according to Bower et al. 2017
        #m_c = calculate_m_crit(z=z)
        
        # create model object
        # (choose feedback model based on feedback_name input)
        if isinstance(feedback_name, str):
            lf_model           = lf_model_class(lf, hmf, feedback_name, m_c, z=z)
            lf_model.directory = lf_model.feedback_model.name
        elif len(feedback_name) == len(lfs):
            lf_model           = lf_model_class(lf, hmf, feedback_name[z], m_c, z=z) 
            lf_model.directory =  'changing'
        else:
            raise ValueError('feedback_name must either be a string or a \
                              list of strings with the same length as lfs.')
                              
        lf_model.filename  = lf_model.directory + str(lf_model.z) + prior_name
        
        # create new prior from distribution of previous iteration
        if z==3: # which is posterior at z=4           
            posterior_samp  = distribution[0] # which is posterior at z=4
            bounds          = list(zip(*lf_models[0].feedback_model.bounds)) # get bounds from z=4
        
        if prior_name == 'uniform':
            prior, bounds = mcmc_fitting.uniform_prior(lf_model, posterior_samp, bounds)
        elif prior_name == 'marginal':
            prior, bounds = mcmc_fitting.dist_from_hist_1d(lf_model, posterior_samp, bounds) 
        elif prior_name == 'full':
            prior, bounds = mcmc_fitting.dist_from_hist_nd(lf_model, posterior_samp, bounds)
        
        # fit parameter
        params, mod_lf, posterior_samp = fit_model(lf_model,
                                                   fitting_method, prior, 
                                                   prior_name, mode)
        
        if z<=3: # add at beginning
            parameter.insert(0,params)  
            modelled_lf.insert(0,mod_lf)
            distribution.insert(0,posterior_samp)     
            lf_models.insert(0,lf_model)   
        else:
            parameter.append(params)  
            modelled_lf.append(mod_lf)
            distribution.append(posterior_samp)     
            lf_models.append(lf_model)
        
    print('Remember you change units for L and have to adjust A parameter accordingly')
        
    return(parameter, modelled_lf, distribution, lf_models)

def fit_model(lf_model, fitting_method, prior, prior_name, mode):
    '''
    Fit the modelled LF (modelled from HMF + feedback) to the observed LF
    Three feedback models: 'none', 'sn', 'both'
    
    IMPORTANT : Abundances (phi values) below 1e-6 are cut off because they 
                cant be measured reliably.
    
    Returns:
    params        : set of fitted parameter (A, alpha, beta)
    modelled_lf   : modelled LF obtained from scaling HMF
    cost          : distribution of parameter (for mcmc fitting)
    '''
    
    # create model and perform fit
    if fitting_method == 'least_squares':
        params, dist   = leastsq_fitting.lsq_fit(lf_model)
    elif fitting_method == 'mcmc':
        params, dist   = mcmc_fitting.mcmc_fit(lf_model, prior, prior_name, mode)
    
    # create data for modelled lf (for plotting)   
    # import pdb; pdb.set_trace()
    lum_range   = np.logspace(7,12,1000)/lf_model.unit
    modelled_lf = lf_model.function(lum_range, params)
    modelled_lf = np.array([lum_range, modelled_lf]).T
    
    #if np.isnan(modelled_lf).any():
     #   import pdb; pdb.set_trace()
 
    # return to 1 solar mass unit 
    modelled_lf[:,0] = modelled_lf[:,0]*lf_model.unit
    return(params, modelled_lf, dist)

## CREATE THE LF MODEL
class lf_model_class():
    def __init__(self, lf, hmf, feedback_name, m_c, z, base_unit =  1e+10):
        # cut unreliable values    
        lf = lf[lf[:,1]>1e-6]
        
        # Change units for numerical stability
        lf[:,0]     =  lf[:,0]/base_unit
        hmf[:,0]    =  hmf[:,0]/base_unit
        m_c         =  m_c/base_unit
        
        self.observations   = lf
        self.hmf            = hmf
        self.hmf_function   = interp1d(*hmf.T) # turn hmf data into evaluable function (using linear interpolation)
        self.feedback_model = feedback_model(feedback_name, m_c) # choose feedback model function
        self.z              = z
        self.unit           = base_unit
        
        self.directory      = None
        self.filename       = None
    def function(self, l, params):
        '''
        Create LF model function by multiplying HMF function with feedback model 
        derivative.
        IMPORTANT: If calculated halo mass m_h is bigger than the largest one 
        given in HMFs by Pratika, set to highest available value instead. (Should
        not really be a problem, since this only happens at z=2, where the value 
        is only minimally bigger)
        '''
        
        # check that parameters are sensible, otherwise invert function will
        # fail to determine halo masses
        if not leastsq_fitting.within_bounds(params, *self.feedback_model.bounds):
                return(np.inf) # return inf (or huge value) if outside of bounds
        
        log_l = np.log10(l)
        
        # calculate halo masses from stellar masses using model
        log_m_h = self.feedback_model.calculate_log_halo_mass(log_l, *params)
        m_h     = np.power(10, log_m_h)
        
        # if halo masses in HMFs is exceeded, set to this value
        m_h_max = np.amax(self.hmf[:,0])
        m_h[m_h>m_h_max] = m_h_max
        m_h_min = np.amin(self.hmf[:,0])
        m_h[m_h<m_h_min] = m_h_min
        return(self.hmf_function(m_h)/self.feedback_model.calculate_dlogobservable_dlogmh(log_m_h,*params))

## DEFINE THE FEEDBACK MODELS
def feedback_model(feedback_name, m_c):
    '''
    Return feedback model that relates LF and HMF, including model function, 
    model name, initial guess and physical parameter bounds for fitting,
    that related LF and HMF. 
    The model function parameters are left free and can be obtained via fitting.
    Three models implemented:
        none    : no feedback adjustment
        sn      : supernova feedback
        both    : supernova and black hole feedback
    '''
    if feedback_name == 'none':  
        model = no_feedback(feedback_name, m_c)
    if feedback_name == 'sn':  
        model = supernova_feedback(feedback_name, m_c)
    if feedback_name == 'both':  
        model = supernova_blackhole_feedback(feedback_name, m_c)
    return(model)

# the feedback models with all necessary parameter and functional equations
# see overleaf notes where these equations come from
class no_feedback():
    def __init__(self, feedback_name, m_c):
        self.name          = feedback_name
        self.m_c           = m_c
        self.initial_guess = [1]
        self.bounds        = [[0], [10]]
    def calculate_log_observable(self, log_m_h, A):
        return(np.log10(A/2*np.power(10, log_m_h)))
    def calculate_log_halo_mass(self, log_observable, A):
        return(np.log10(2*np.power(10, log_observable)/A))
    def calculate_dlogobservable_dlogmh(self, log_m_h, A):
        return(1)        


class supernova_feedback():
    def __init__(self, feedback_name, m_c):
        self.name          = feedback_name
        self.m_c           = m_c      
        self.initial_guess = [0.1, 1]
        self.bounds        = [[0, 0], [10, 4]]
    def calculate_log_observable(self, log_m_h, A, alpha):
        if np.isnan(log_m_h).any() or np.any((log_m_h-np.log10(self.m_c))>18):
            return(np.nan)
        ratio = np.power(10, log_m_h - np.log10(self.m_c))
        sn = ratio**(-alpha)
        obs = A * np.power(10, log_m_h)/(1 + sn)
        return(np.log10(obs))    
    def calculate_log_halo_mass(self, log_observable, A, alpha):
        log_m_h = invert_function(func    = self.calculate_log_observable,
                                  fprime  = self.calculate_dlogobservable_dlogmh,
                                  fprime2 = self.calculate_d2logobservable_dlogmh2,
                                  x0_func = self._initial_guess, 
                                  y       = log_observable,
                                  args    = (A, alpha))
        return(log_m_h)
    def calculate_dlogobservable_dlogmh(self, log_m_h, A, alpha):
        if np.isnan(log_m_h).any() or np.any((log_m_h-np.log10(self.m_c))>18):
            return(np.nan)
        ratio = np.power(10, log_m_h - np.log10(self.m_c))
        sn = ratio**(-alpha)
        return(1 - (-alpha*sn)/(1 + sn))
    def calculate_d2logobservable_dlogmh2(self, log_m_h, A, alpha): 
        if np.isnan(log_m_h).any() or np.any((log_m_h-np.log10(self.m_c))>18):
            return(np.nan)
        ratio = np.power(10, log_m_h - np.log10(self.m_c))
        sn = ratio**(-alpha)
        denom = (1 + sn)
        num_one = alpha**2*sn
        num_two =(-alpha*sn)**2
        return(-np.log(10)*(num_one/denom+num_two/denom**2))
    def _initial_guess(self, log_observable, A, alpha):
        # guess initial value for inverting function by using high and low mass
        # end approximation
        m_t          = A/2*self.m_c  # turnover mass where dominating feedback changes
        trans_regime = 20            # rough estimate for transient regime where both are important
        if np.power(10, log_observable) < m_t/trans_regime:
            x0 = np.power((self.m_c)**alpha/A*np.power(10, log_observable),1/(1+alpha))
        elif np.power(10, log_observable) > m_t*trans_regime:
            x0 = A*self.m_c  
        else:
            x0 = np.power(10, log_observable)*2/A
        return(np.log10(x0))

class supernova_blackhole_feedback():
    def __init__(self, feedback_name, m_c):
        self.name          = feedback_name
        self.m_c           = m_c
        self.initial_guess = [0.1, 1, 0.1]       
        self.bounds        = [[0, 0, 0], [10, 4, 0.8]]
    def calculate_log_observable(self, log_m_h, A, alpha, beta):
        if np.isnan(log_m_h).any() or np.any((log_m_h-np.log10(self.m_c))>18):
            return(np.nan)
        ratio = np.power(10, log_m_h - np.log10(self.m_c))
        sn = ratio**(-alpha)
        bh = ratio**beta
        obs = A * np.power(10, log_m_h)/(sn + bh)
        return(np.log10(obs))    
    def calculate_log_halo_mass(self, log_observable, A, alpha, beta):
        log_m_h = invert_function(func    = self.calculate_log_observable,
                                  fprime  = self.calculate_dlogobservable_dlogmh,
                                  fprime2 = self.calculate_d2logobservable_dlogmh2,
                                  x0_func = self._initial_guess, 
                                  y       = log_observable,
                                  args    = (A, alpha, beta))
        return(log_m_h)
    def calculate_dlogobservable_dlogmh(self, log_m_h, A, alpha, beta):
        if np.isnan(log_m_h).any() or np.any((log_m_h-np.log10(self.m_c))>18):
            return(np.nan)
        ratio = np.power(10, log_m_h - np.log10(self.m_c))
        sn = ratio**(-alpha)
        bh = ratio**beta
        return(1 - (-alpha*sn + beta * bh)/(sn + bh))
    def calculate_d2logobservable_dlogmh2(self, log_m_h, A, alpha, beta): 
        if np.isnan(log_m_h).any() or np.any((log_m_h-np.log10(self.m_c))>18):
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
        
## HELP FUNCTIONS
def calculate_m_crit(z):
    '''
    Calculate critical mass at a given redshift in solar masses, following the
    model by Bower et al. 2017.
    https://doi.org/10.1093/mnras/stw2735
    '''
    omega_m = Planck18.Om(z=z)
    omega_l = Planck18.Ode(z=z)
    delta_z = np.power(omega_m*(1+z)**3+omega_l, 1/3)
    return(np.power(delta_z,-3/8)*1e+12)

def invert_function(func, fprime, fprime2, x0_func, y, args):
    '''
    For a function y=f(x), calculate x values for an input set of y values.

    '''
    x = []      
    for val in y:
        def root_func(m,*args):
            return(func(m,*args)-val)
        
        x0_in = x0_func(val, *args) # guess initial value
        
        root = root_scalar(root_func, fprime = fprime, fprime2=fprime2, args = args,
                            method='halley', x0 = x0_in, rtol=1e-6).root
        # if Halley's method doesn't work, try Newton
        if np.isnan(root):
                root = root_scalar(root_func, fprime = fprime, fprime2=fprime2, args = args,
                                    method='newton', x0 = x0_in, rtol=1e-6).root
        x.append(root)
    x = np.array(x)
    return(x)
