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
    SMF.
    '''
    def __init__(self, smfs, hmfs, feedback_name, fitting_method, prior_name, mode):
        parameter, modelled_smf, distribution, smf_model = fit_SMF_model(smfs, hmfs, feedback_name,
                                                                         fitting_method, prior_name, mode)
        self.feedback_name = feedback_name
        
        self.parameter     = smf_data(parameter)
        self.smf           = smf_data(modelled_smf)
        self.distribution  = smf_data(distribution)
        
        self.model         = smf_data(smf_model)
        
    def plot_parameter(self, color, marker, linestyle, label):
        # style parameter for fit
        self.color     = color
        self.marker    = marker
        self.linestyle = linestyle
        self.label     = label
        return(self)
 
class smf_data():
    # easily retrieve data at certain redshift
    def __init__(self, data):
        self.data = data
    def at_z(self, redshift):
        if redshift == 0:
            raise ValueError('Redshift 0 not in data')
        return(self.data[redshift-1])

## MAIN FUNCTIONS   
def fit_SMF_model(smfs, hmfs, feedback_name,
                  fitting_method, prior_name, mode):
    '''
    Perform SMF fit for all redshifts.
    
    feedback_name can either be a single string (then this model is used for all
    redshifts) or a list of strings corresponding to the model used for each redshift.
    
    Choose between 3 different prior model:
        uniform  : assume uniform prior within bounds for each redshift
        marginal : use marginalized distribution for parameter from previous 
                   redshift (assuming independence of parameter)
        full     : use full distribution for parameter from previous 
                   redshift (assuming dependence of parameter)
    
    IMPORTANT : Critical mass calculated according to model by Bower et al.
    '''
    
    parameter = []; modelled_smf = []; distribution = []; smf_models = []
    posterior_samp = None
    bounds         = None
    for i in range(len(smfs)):
        smf = np.copy(smfs[i])
        hmf = np.copy(hmfs[i+1]) # list starts at z=0 not 1, like smf
        
        # calc m_crit according to Bower et al.
        m_c = calculate_m_crit(z=i+1)
        
        # create model object
        # (choose feedback model based on feedback_name input)
        if isinstance(feedback_name, str):
            smf_model           = smf_model_class(smf, hmf, feedback_name, m_c, z=i+1)
            smf_model.directory =  smf_model.feedback_model.name
        elif len(feedback_name) == len(smfs):
            smf_model           = smf_model_class(smf, hmf, feedback_name[i], m_c, z=i+1) 
            smf_model.directory =  'changing'
        else:
            raise ValueError('feedback_name must either be a string or a \
                              list of strings with the same length as smfs.')
                              
        smf_model.filename  = smf_model.directory + str(smf_model.z) + prior_name
        
        # create new prior from distribution of previous iteration
        if prior_name == 'uniform':
            prior, b = mcmc_fitting.uniform_prior(smf_model, posterior_samp, bounds) 
        elif prior_name == 'marginal':
            prior, b = mcmc_fitting.dist_from_hist_1d(smf_model, posterior_samp, bounds) 
        elif prior_name == 'full':
            prior, b = mcmc_fitting.dist_from_hist_nd(smf_model, posterior_samp, bounds) 
        bounds = b
        
        # fit parameter
        params, mod_smf, posterior_samp = fit_model(smf_model,
                                                    fitting_method, prior, 
                                                    prior_name, mode)
        parameter.append(params)  
        modelled_smf.append(mod_smf)
        distribution.append(posterior_samp)     
        smf_models.append(smf_model)
        
    return(parameter, modelled_smf, distribution, smf_models)

def fit_model(smf_model, fitting_method, prior, prior_name, mode):
    '''
    Fit the modelled SMF (modelled from HMF + feedback) to the observed SMF
    Three feedback models: 'none', 'sn', 'both'
    
    IMPORTANT : Abundances (phi values) below 1e-6 are cut off because they 
                cant be measured reliably.
    
    Returns:
    params        : set of fitted parameter (A, alpha, beta)
    modelled_smf  : modelled SMF obtained from scaling HMF
    cost          : distribution of parameter (for mcmc fitting)
    '''
    # choose fitting method
    if fitting_method == 'least_squares':
        fit = leastsq_fitting.lsq_fit
    elif fitting_method == 'mcmc':
        def fit(smf_model): # choose saving/loading mode and prior
            return(mcmc_fitting.mcmc_fit(smf_model, prior, prior_name, mode))
    
    # create model and perform fit
    params, dist = fit(smf_model)
    
    # create data for modelled smf (for plotting)   
    modelled_smf = np.copy(smf_model.hmf)
    modelled_smf = modelled_smf[modelled_smf[:,0]<(1e+13/smf_model.unit)]
    modelled_smf[:,1] = smf_model.function(modelled_smf[:,0], params)
    
    # return to 1 solar mass unit 
    modelled_smf[:,0] = modelled_smf[:,0]*smf_model.unit
    return(params, modelled_smf, dist)

## CREATE THE SMF MODEL
class smf_model_class():
    def __init__(self, smf, hmf, feedback_name, m_c, z, base_unit =  1e+10):
        # cut unreliable values    
        smf = smf[smf[:,1]>1e-6]
        
        # Change units from 1 solar mass to 10^10 solar masses for numerical stability
        smf[:,0]    = smf[:,0]/base_unit
        hmf[:,0]    = hmf[:,0]/base_unit 
        m_c = m_c/base_unit
    
        self.observations   = smf
        self.hmf            = hmf
        self.hmf_function   = interp1d(*hmf.T) # turn hmf data into evaluable function (using linear interpolation)
        self.feedback_model = feedback_model(feedback_name, m_c) # choose feedback model function
        self.z              = z
        self.unit           = base_unit
        
        self.directory      = None
        self.filename       = None
    def function(self, m_star, params):
        '''
        Create SMF model function by multiplying HMF function with feedback model 
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
        
        # calculate halo masses from stellar masses using model
        m_h = self.feedback_model.calculate_m_h(m_star, *params)
        
        # if halo masses in HMFs is exceeded, set to this value
        m_h_max = np.amax(self.hmf[:,0])
        m_h[m_h>m_h_max] = m_h_max
        m_h_min = np.amin(self.hmf[:,0])
        m_h[m_h<m_h_min] = m_h_min
        return(self.hmf_function(m_h) / self.feedback_model.calculate_dlogmstar_dlogmh(m_h,*params))

## DEFINE THE FEEDBACK MODELS
def feedback_model(feedback_name, m_c):
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
        self.initial_guess = [0.01]
        self.bounds        = [[0], [2]]
    def calculate_m_star(self, m_h, A):
        return(A/2*m_h)
    def calculate_m_h(self, m_star, A):
        return(2*m_star/A)
    def calculate_dlogmstar_dlogmh(self, m_h, A):
        return(1)        

class supernova_feedback():
    def __init__(self, feedback_name, m_c):
        self.name          = feedback_name
        self.m_c           = m_c      
        self.initial_guess = [0.01, 1]
        self.bounds        = [[0, 0], [2, 3]]
    def calculate_m_star(self, m_h, A, alpha):
        if np.isnan(m_h).any() or np.any(m_h<0):
            return(np.nan)
        sn = (m_h/self.m_c)**(-alpha)
        return(A * m_h/(1 + sn))   
    def calculate_m_h(self, m_star, A, alpha):        
        m_h = invert_function(func    = self.calculate_m_star,
                              fprime  = self._calculate_dmstar_dmh,
                              fprime2 = self._calculate_d2mstar_dmh2,
                              x0_func = self._guess_initial_m_h, 
                              y       = m_star,
                              args    = (A, alpha)) 
        return(m_h)
    def calculate_dlogmstar_dlogmh(self, m_h, A, alpha):
        sn = (m_h/self.m_c)**(-alpha)
        return(1 + alpha*sn/(1 + sn))
    def _calculate_dmstar_dmh(self, m_h, A, alpha): # first derivative, just used to calc inverse
        if m_h<0:
            return(np.nan)
        sn = (m_h/self.m_c)**(-alpha)
        return(A * ( 1 + (1+alpha)*sn)/(1+sn)**2)
    def _calculate_d2mstar_dmh2(self, m_h, A, alpha): # second derivative, just used to calc inverse
        if m_h<0:
            return(np.nan)
        x = m_h/self.m_c; a = alpha
        denom = 1 + x**(-a)
        first_num  = (1+a)
        second_num = -2*(1+(1+a)*x**(-a))
        return(A/self.m_c*(-a*x**(-a-1)) * (first_num/denom**2 + second_num/denom**3))
    def _guess_initial_m_h(self, m_star, A, alpha):
        # guess initial value for inverting function by using high and low mass
        # end approximation
        m_t          = A/2*self.m_c # turnover mass where dominating feedback changes
        trans_regime = 20           # rough estimate for transient regime where both are important
        if m_star < m_t/trans_regime:
            x0 = np.power((self.m_c)**alpha/A*m_star,1/(1+alpha))
        elif m_star > m_t*trans_regime:
            x0 = A*self.m_c
        else:
            x0 =  m_star*2/A
        return(x0)

class supernova_blackhole_feedback():
    def __init__(self, feedback_name, m_c):
        self.name          = feedback_name
        self.m_c           = m_c
        self.initial_guess = [0.01, 1, 0.3]       
        self.bounds        = [[0, 0, 0], [2, 3, 0.8]]
    def calculate_m_star(self, m_h, A, alpha, beta):
        if np.isnan(m_h).any() or np.any(m_h<0):
            return(np.nan)
        sn = (m_h/self.m_c)**(-alpha)
        bh = (m_h/self.m_c)**beta
        return(A * m_h/(sn + bh))    
    def calculate_m_h(self, m_star, A, alpha, beta):
        m_h = invert_function(func    = self.calculate_m_star,
                              fprime  = self._calculate_dmstar_dmh,
                              fprime2 = self._calculate_d2mstar_dmh2,
                              x0_func = self._guess_initial_m_h, 
                              y       = m_star,
                              args    = (A, alpha, beta)) 
        return(m_h)
    def calculate_dlogmstar_dlogmh(self, m_h, A, alpha, beta):
        #print(A, alpha, beta)
        sn = (m_h/self.m_c)**(-alpha)
        bh = (m_h/self.m_c)**beta
        return(1 - (-alpha*sn + beta * bh)/(sn + bh))
    def _calculate_dmstar_dmh(self, m_h, A, alpha, beta): 
        # first derivative, just used to calc inverse
        if m_h<0:
            return(np.nan)
        sn = (m_h/self.m_c)**(-alpha)
        bh = (m_h/self.m_c)**beta
        return(A * ((1+alpha)*sn+(1-beta)*bh)/(sn + bh)**2)
    def _calculate_d2mstar_dmh2(self, m_h, A, alpha, beta): 
        # second derivative, just used to calc inverse
        if m_h<0:
            return(np.nan)
        x = m_h/self.m_c; a = alpha; b = beta
        denom = x**(-a) + x**b
        first_num  = (1+a)*(-a)*x**(-a-1)+(1-b)*b*x**(b-1)
        second_num = -2*((1+a)* x**(-a)+(1+b)*x**b)*(-a*x**(-a-1)+b*x**(b-1))
        return(A/self.m_c * (first_num/denom**2 + second_num/denom**3))
    def _guess_initial_m_h(self, m_star, A, alpha, beta):
        # guess initial value for inverting function by using high and low mass
        # end approximation
        #if m_star == 0.0001:
        #    import pdb; pdb.set_trace()
        m_t          = A/2*self.m_c  # turnover mass where dominating feedback changes
        trans_regime = 20            # rough estimate for transient regime where both are important
        if m_star < m_t/trans_regime:
            x0 = np.power((self.m_c)**alpha/A*m_star,1/(1+alpha))
        elif m_star > m_t*trans_regime:
            x0 = np.power((self.m_c)**(-beta)/A*m_star,1/(1-beta))   
        else:
            x0 = m_star*2/A
        return(x0)
        
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
                                    method='halley', x0 = x0_in, rtol=1e-6).root
        x.append(root)
    x = np.array(x)
    return(x)
