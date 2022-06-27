#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 17:46:44 2022

@author: chris
"""
import warnings

import numpy as np
from scipy.integrate import trapezoid

from progressbar import ProgressBar, FormatLabel, NullBar

from model.helper import mag_to_lum, within_bounds, make_array, system_path
from model.calibration import mcmc_fitting, leastsq_fitting
from model.quantity_options import get_quantity_specifics, get_bounds
from model.physics import physics_model
from model.calibration.parameter import load_parameter

################ MAIN CLASSES #################################################
def choose_model(quantity_name):
    '''
    Choose appropriate model for quantity.
    
    '''
    if quantity_name in ['mstar', 'Muv', 'mbh']:
        return(ModelResult)
    elif quantity_name == 'Lbol':
        return(ModelResult_QLF)
    else:
        raise NameError('quantity_name not known.')

class ModelResult():
    '''
    Central model object.
    '''

    def __init__(self, redshifts, log_ndfs, log_hmf_functions,
                 quantity_name, physics_name, prior_name,
                 fitting_method, saving_mode, name_addon=None,
                 groups=None, calibrate=True, paramter_calc = True,
                 progress=True, **kwargs):
        '''
        Main model object. Calibrate the model by fitting parameter to
        observational data.

        Parameters
        ----------
        redshifts : int or array
            List of redshifts that are fitted.
        log_ndfs : dict
            Dictonary of number density function observations (UVLFs/SMFs) at
            different redshifts of form {z: ndf}. 
            Input should be logarithmic values (quantity, phi). For luminosity
            function, the quantity should be given in magnitudes.
        log_hmf_functions : dict
            Dictonary of halo mass function at every redshift, which should be 
            callable functions of form {z: hmf}.
        quantity_name : str
            Name of the quantity modelled. Must be 'Muv' or 'mstar'.
        physics_name : str
            Name of physics model. Must be in implemented models in 
            quantity_options. Model 'changing' uses 'stellar_blackhole'
            for z<4 and 'stellar' for z>4.
        prior_name : str
            Name of prior model. Must be 'uniform' or 'successive'.
        fitting_method : str
            Name of fitting procedure. Must be 'least_squares' or 'mcmc'.
        saving_mode : str
            Name of saving procedure. Must be 'saving', 'loading' or 'temp',
            where 'temp' means no saving.
        name_addon : str, optional
            Optional extension to filename for saving.
        groups : list, optional
            List of group objects that contain the observational data and 
            plotting information.
        calibrate : bool, optional
            Choose if model is supposed to be calibrated or not. 
            The default is True.
        parameter_calc : bool, optional
            Choose if best fit parameter are supposed to be calculated 
            (or laoded). The default is True.
        progress : bool, optional
            Choose if progress bar is supposed to be shown. The default is 
            True.
        **kwargs : dict
            Additional arguments that can be passed to the mcmc function.

        Returns
        -------
        None.

        '''

        self.redshift = make_array(redshifts)
        self.log_ndfs = Redshift_dict(log_ndfs)
        self.log_hmfs = Redshift_dict(log_hmf_functions)
        self.groups = groups

        self.hmf_slope = -0.9 # approximate low mass slope of HMFs
        
        self.quantity_name = quantity_name
        # load options related to the quantity 
        self.quantity_options = get_quantity_specifics(self.quantity_name)
        self.log_m_c = self.quantity_options['log_m_c']
        
        self.physics_name = physics_name
        self.prior_name = prior_name

        self.fitting_method = fitting_method
        self.saving_mode = saving_mode
        self.name_addon = name_addon
        
        # location where files will be saved or loaded from
        self.directory = None

        self.physics_model = Redshift_dict({})
        self.filename = Redshift_dict({})
        self.parameter = Redshift_dict({})
        self.distribution = Redshift_dict({})
        
        self.progress = progress
        
        if calibrate:
            self.fit_model(self.redshift, **kwargs)
        else:
            for z in self.redshift:
                self._add_physics_model(z)
                

        if (saving_mode == 'loading'):
            try:
                parameter = load_parameter(self, name_addon)
                parameter = {int(z): p for z, p in parameter.items()}
                self.parameter = Redshift_dict(parameter)
            except FileNotFoundError:
                warnings.warn('Couldn\'t load best fit parameter')

        # default plot parameter per physics_model
        if self.physics_name == 'none':
            self._plot_parameter('black', 'o', '-', 'No Feedback')
        elif self.physics_name == 'stellar':
            self._plot_parameter('C1', 's', '--', 'Stellar Feedback')
        elif self.physics_name == 'stellar_blackhole':
            self._plot_parameter(
                'C2', 'v', '-.', 'Stellar + Black Hole Feedback')
        elif self.physics_name == 'changing':
            feedback_change_z = self.quantity_options['feedback_change_z']
            max_z             = 10
            self._plot_parameter(
                ['C2'] * feedback_change_z + ['C1'] * (max_z+1-feedback_change_z),
                'o', '-',
                ['Stellar + Black Hole Feedback'] * feedback_change_z \
                + ['Stellar Feedback'] * (max_z+1-feedback_change_z))
        elif self.physics_name == 'quasar':
            self._plot_parameter(
                'C3', '^', ':', 'BH Growth Model')
        elif self.physics_name == 'eddington':
            self._plot_parameter(
                'C4', '^', ':', 'Bolometric Luminosity Model')
        else:
            warnings.warn('Plot parameter not defined')

    def fit_model(self, redshifts, **kwargs):
        '''
        Calibrate model by fitting to data (or loading previous fit).

        Parameters
        ----------
        redshifts : int or list
            Choose redshifts that are supposed to be fitted.
        **kwargs : dict
            See main description.

        Returns
        -------
        None.

        '''
        # define progress bar (mcmc uses custom one)
        custom_bar_flag =  ((self.progress and 
                             self.fitting_method=='mcmc' and 
                             not (self.saving_mode=='loading')) 
                            or (not self.progress))
        if custom_bar_flag:
            PBar = NullBar()
        else:
            PBar = ProgressBar(widgets=[FormatLabel('')])
        
        redshifts = make_array(redshifts)
        
        # run fits
        parameters, distributions = {}, {}
        posterior_samp, bounds = None, None
        for z in PBar(redshifts):
            # progress tracking
            if not custom_bar_flag:
                model_details = self.quantity_options['ndf_name'] + ' - ' +\
                               'physics('+self.physics_name + ') - ' +\
                                self.prior_name + ' prior: '
                PBar.widgets[0] = model_details + f'z = {z}'
            self._z = z # temporary storage for current redshift

            # add saving paths and file name
            self.directory = system_path() + self.quantity_name + '/' \
                            + self.physics_name + '/'
            filename = self.prior_name + '_z' + str(z)
            # if manual modification of saving path is wanted
            if self.name_addon:
                filename = filename + ''.join(self.name_addon)
            self.filename.add_entry(z, filename)
      
            # add physics model
            self._add_physics_model(z)

            # create new prior from distribution of previous iteration
            if self.fitting_method != 'loading':
                if self.prior_name == 'uniform':
                    prior, bounds = mcmc_fitting.uniform_prior(
                        self, posterior_samp, bounds)
                elif self.prior_name == 'marginal':
                    raise DeprecationWarning(
                        'Marginal prior not really sensible anymore, I think.')
                    prior, bounds = mcmc_fitting.dist_from_hist_1d(
                        self, posterior_samp, bounds)
                elif self.prior_name == 'successive':
                    prior, bounds = mcmc_fitting.dist_from_hist_nd(
                        self, posterior_samp, bounds)
                else:
                    raise NameError('Prior model not known.')

            # fit parameter/sample distribution
            if self.fitting_method == 'least_squares':
                parameter, posterior_samp = leastsq_fitting.lsq_fit(self)
            elif self.fitting_method == 'mcmc':
                parameter, posterior_samp = mcmc_fitting.mcmc_fit(
                    self, prior, saving_mode=self.saving_mode, 
                    progress = self.progress, **kwargs)
            else:
                raise NameError('fitting_method not known.')
            
            parameters[z]    = parameter
            distributions[z] = posterior_samp
            
            if (not custom_bar_flag) and (z==redshifts[-1]):
                PBar.widgets[0] = model_details + 'DONE'
            
        # add distributions to model object after fitting is done, because
        # otherwise, large amount of data in model slows (parallel) mcmc run
        self.parameter    = Redshift_dict(parameters)
        self.distribution = Redshift_dict(distributions)
        
        return

    def calculate_log_abundance(self, log_quantity, z, parameter):
        '''
        Calculate (log of) value (phi) of modelled number density function by 
        multiplying HMF function with physics model derivative for a given
        redshift.
        
        IMPORTANT: Input units must be log m_star in units of solar masses for
        the SMF and UV luminosity in absolute mag for UVLF.
        
        Parameters
        ----------
        log_quantity : float64
            Input (log of) observable quantity. For mass function must be in
            stellar masses, for luminosities must be in absolute magnitudes.
        z : int
            Redshift at which value is calculated.
        parameter : list 
            Model parameter used for calculation.

        Returns
        -------
        log_phi : float
            Log value of ndf at the input value and redshift.

        '''
        log_quantity = make_array(log_quantity)
        if self.quantity_name == 'Muv':
            # convert magnitude to luminosity
            log_quantity = np.log10(mag_to_lum(log_quantity))

        # check that parameters are within bounds
        assert within_bounds(parameter, *self.physics_model.at_z(z).bounds),\
        'Parameter out of bounds.'

        # calculate halo masses from stellar masses using model
        log_m_h = self.physics_model.at_z(z).calculate_log_halo_mass(
            log_quantity, *parameter)
        
        ## calculate value of bh phi (hmf + quasar growth model)  
        
        # calculate value of halo mass function
        log_hmf   = self.calculate_log_hmf(log_m_h, z)
        
        # calculate physics/feedback effect (and deal with zero values)
        fb_factor = self.physics_model.at_z(z).calculate_dlogquantity_dlogmh(
                         log_m_h, *parameter)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',category=RuntimeWarning)
            # log of 0 is -inf, suppress corresponding numpy warning 
            log_fb_factor             = np.log10(fb_factor)
            # calculate modelled phi value (ignore inf-inf warning)
            log_phi                   = log_hmf - log_fb_factor
            
        # deal with infinite masses 
        log_phi[np.isinf(log_m_h)]    =  - np.inf
        return(log_phi)

    def draw_parameter_sample(self, z, num=1):
        '''
        Get a sample from physics model parameter distribution at given redshift.

        Parameters
        ----------
        z : int
            Redshift at which value is calculated.
        num : int, optional
            Number of samples to be drawn. The default is 1.

        Returns
        -------
        parameter_sample : list
            List of parameter samples.
        '''
        if self.distribution.is_None():
            raise AttributeError('distribution dictonary is empty. Probably'\
                                 ' wasn\'t calculated.')
        
        # randomly draw from parameter distribution at z
        random_draw = np.random.choice(self.distribution.at_z(z).shape[0],
                                       size=num)
        parameter_sample = self.distribution.at_z(z)[random_draw]
        return(parameter_sample)

    def calculate_quantity_distribution(self, log_halo_mass, z, num=int(1e+5)):
        '''
        At a given redshift, calculate distribution of observable quantity
        (mstar/Muv) for a given halo mass by drawing parameter sample and
        calculating value for each one.

        Parameters
        ----------
        log_halo_mass : float64 or list
            Input (log) halo masses for which quantity distribution is 
            caluclated.
        z : int
            Redshift at which value is calculated.
        num : int, optional
            Number of samples drawn. The default is int(1e+5).

        Returns
        -------
        log_quantity_dist : array
            Calculated distribution.

        '''
        
        parameter_sample = self.draw_parameter_sample(z, num=num)

        log_quantity_dist = []
        for p in parameter_sample:
            log_quantity_dist.append(
                self.physics_model.at_z(z).calculate_log_quantity(
                    log_halo_mass, *p))
        return(np.array(log_quantity_dist))

    def calculate_halo_mass_distribution(self, log_quantity, z, num=int(1e+5)):
        '''
        At a given redshift, calculate distribution of halo mass for a given
        observable quantity (mstar/Muv) by drawing parameter sample and
        calculating value for each one (number of draws adaptable).

        Parameters
        ----------
        log_quantity : float64
            Input (log of) observable quantity. For mass function must be in
            stellar masses, for luminosities must be in absolute magnitudes.
        z : int
            Redshift at which value is calculated.
        num : int, optional
            Number of samples drawn. The default is int(1e+5).

        Returns
        -------
        log_halo_mass_dist : array
            Calculated distribution.

        '''
        parameter_sample = self.draw_parameter_sample(z, num=num)

        log_halo_mass_dist = []
        for p in parameter_sample:
            log_halo_mass_dist.append(
                self.physics_model.at_z(z).calculate_log_halo_mass(
                    log_quantity, *p))
        return(np.array(log_halo_mass_dist))
    
    def calculate_abundance_distribution(self, log_quantity, z, num=int(1e+5)):
        '''
        At a given redshift, calculate distribution of phi for a given
        observable quantity (mstar/Muv) value by drawing parameter sample and
        calculating value for each one.

        Parameters
        ----------
        log_quantity : float64
            Input (log of) observable quantity. For mass function must be in
            stellar masses, for luminosities must be in absolute magnitudes.
        z : int
            Redshift at which value is calculated.
        num : int, optional
            Number of samples drawn. The default is int(1e+5).

        Returns
        -------
        log_abundance_dist : array
            Calculated distribution.

        '''
        
        parameter_sample = self.draw_parameter_sample(z, num=num)

        log_abundance_dist = []
        for p in parameter_sample:
            log_abundance_dist.append(
                self.calculate_log_abundance(log_quantity, z, p))
        return(np.array(log_abundance_dist))

    def calculate_ndf(self, z, parameter, quantity_range=None):
        '''
        Calculate a model number density function over a representative range
        at redshift z and using input parameter.  

        Parameters
        ----------
        z : int
            Redshift at which value is calculated.
        parameter : list
            Input model parameter.
        quantity_range : list, optional
            Range over which values are supposed to be calculated. If None
            use default from options dictonary.

        Returns
        -------
        ndf : array
            Calculated number density functions.

        '''
        if quantity_range is None:
            quantity_range = self.quantity_options['quantity_range']

        ndf = self.calculate_log_abundance(quantity_range, z, parameter)
        return([quantity_range, ndf])
    
    def calculate_log_hmf(self, log_halo_mass, z):
        '''
        Calculate value of Halo Mass Function for a given halo mass and
        redshift. If any of the values are nan, it's assumed they're 
        were supposed to be extrapolated at the high mass end, which should
        result in a value close to -inf.

        Parameters
        ----------
        log_quantity : float64
            Input (log of) observable quantity. For mass function must be in
            stellar masses, for luminosities must be in absolute magnitudes.
        z : int
            Redshift at which value is calculated.

        Returns
        -------
        log_hmf : array
            Calculated values of HMF.

        '''
        log_halo_mass = make_array(log_halo_mass)
        # ignore RuntimeWarning for high end extrapolation, deal with those
        # after by setting log_hmf to -inf
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',category=RuntimeWarning)
            log_hmf = self.log_hmfs.at_z(z)(log_halo_mass)
        log_hmf[np.isnan(log_hmf)] = -np.inf
        return(log_hmf)
        

    def get_ndf_sample(self, z, num=100, quantity_range=None):
        '''
        Get a sample of ndf curves (as a list) with parameters randomly drawn
        from the distribution.

        Parameters
        ----------
        z : int
            Redshift at which value is calculated.
        num : int, optional
            Number of sample ndfs calculated. The default is 100.
        quantity_range : list, optional
            Range over which values are supposed to be calculated..

        Returns
        -------
        ndf_sample : list
            List of calculated number density functions.

        '''
        parameter_sample = self.draw_parameter_sample(z, num=num)
        ndf_sample = []
        for n in range(num):
            ndf = self.calculate_ndf(z, parameter_sample[n],
                                     quantity_range=quantity_range)
            ndf_sample.append(np.array(ndf).T)
        return(ndf_sample)
    
    def _add_physics_model(self, z):
        '''
        Add physics model to general model according to physics_name.

        '''
        ## create physics model
        if self.physics_name in ['none', 'stellar', 'stellar_blackhole',
                                  'quasar', 'eddington']:
            fb_name = self.physics_name
        elif self.physics_name == 'changing':  # standard changing feedback
            feedback_change_z = self.quantity_options['feedback_change_z']
            if z < feedback_change_z:
                fb_name = 'stellar_blackhole'
            elif z >= feedback_change_z:
                fb_name = 'stellar'
        else:
            raise NameError('physics_name not known.')
        # get model parameter bounds
        bounds_at_z = get_bounds(z, self)
        # add model
        self.physics_model.add_entry(z, physics_model(
            fb_name,
            self.log_m_c,
            initial_guess=self.quantity_options['model_p0'],
            bounds=bounds_at_z))
        return
        
    def _plot_parameter(self, color, marker, linestyle, label):
        '''
        Add style parameter for plot

        Parameters
        ----------
        color : str or list
            Color passed to matplotlib.
        marker : str or list
            Marker style passed to matplotlib.
        linestyle : str or list
            Linestyle passed to matplotlib.
        label : str or list
            Labels passed to matplotlib.

        Returns
        -------
        None.
        
        '''
        self.color     = color
        self.marker    = marker
        self.linestyle = linestyle
        self.label     = label
        return

    
class ModelResult_QLF(ModelResult):
    '''
    An adapated Model class used for calculating the Quasar Luminosity Function
    (QLF). For this quantity, we need an adapted methodology that takes
    input from the Active Black Hole Mass Function, so that things need to be
    adapted.
    IMPORTANT: For compatibility with the parent class, the 'physics_model'
               will be overwritten with the name of the Eddington rate model.
               Please call self.physics_model for and self.physics_name for
               information on the Eddington model, and 
               self.bh_model.physics_model and self.bh_model.physics_name for
               the feedback/black hole growth model.
    '''
    def __init__(self, redshifts, log_ndfs, log_hmf_functions,
                 quantity_name, physics_name, prior_name,
                 fitting_method, saving_mode, name_addon=None,
                 groups=None, calibrate=True, paramter_calc = True,
                 progress=True, **kwargs):    
       
        # used for _make_space, so it doesn't have to be recreated every time
        self._initial_eddington_space = np.linspace(-50,50,200)
        
        # initalize model itself
        super().__init__(redshifts, log_ndfs, log_hmf_functions,
                         quantity_name, physics_name, prior_name,
                         fitting_method, saving_mode, name_addon,
                         groups, calibrate, paramter_calc,
                         progress, **kwargs)

              
    def calculate_log_abundance(self, log_L, z, parameter, num=100):
        #breakpoint()
        log_L = make_array(log_L)

        # check that parameters are within bounds
        assert within_bounds(parameter, *self.physics_model.at_z(z).bounds),\
        'Parameter out of bounds.' 
        
        # phi = []
        
        phi = []
        for L in log_L:
            eddington_ratio_space = self._make_log_eddington_ratio_space(L, z, 
                                                                         parameter)
            log_qlf_contribution  = self.calculate_log_QLF_contribution(eddington_ratio_space,
                                                                        L,
                                                                        z,
                                                                        parameter)
            
            phi.append(trapezoid(np.power(10, log_qlf_contribution), 
                       eddington_ratio_space)) 
        
        # for L in log_L:
        #     print(L)
            
        #     def func(log_eddington_ratio):
        #         return(self.calculate_QLF_contribution(log_eddington_ratio,
        #                                                L, z, parameter))
            
        #     phi.append(calculate_integral(func, [-100, 100]))
            
            
            
        # phi = []
        # for L in log_L:
        #     #print(L)
        #     with warnings.catch_warnings():
        #         warnings.simplefilter('ignore', category=IntegrationWarning)
        #         #try:
        #         integral = quad(self.calculate_QLF_contribution, 
        #                           -np.inf,
        #                             np.inf,
        #                         args = (L, z, parameter), limit=500)[0]
        #         #except IntegrationWarning:
        #         #    print('whoops')
        #         #    integral = np.nan
        #     phi.append(integral)
        # phi = np.array(phi)
        
        # eddingtion_ratio_sample = np.sort(self.physics_model.at_z(z).draw_eddington_ratio(*parameter[1:], 10000))
        # delta_edd = eddingtion_ratio_sample[1:]-eddingtion_ratio_sample[:-1]
        
        
        # phi_contribution = []
        # for i in range(len(eddingtion_ratio_sample)-1):
        #     phi_contribution.append(self.calculate_QLF_contribution(eddingtion_ratio_sample[i]+delta_edd[i]/2,
        #                                                             log_L,
        #                                                             z,
        #                                                             parameter))
            
        # # sum all contributions for different eddington ratios
        # phi_contribution = np.array(phi_contribution)
        # phi              = np.sum(delta_edd[:,np.newaxis]*phi_contribution, axis=0)
        
        # calculate log of calculated BH luminosity function, taking care of
        # zero values
        phi     = np.array(phi)
        log_phi = np.empty_like(phi)
        log_phi[phi==0] = -np.inf
        log_phi[phi!=0] = np.log10(phi[phi!=0])
        return(log_phi)

    def calculate_quantity_distribution(self, log_black_hole_mass, 
                                        z, num=int(1e+5)):
        return()

    def calculate_halo_mass_distribution(self, log_L, z, num=int(1e+5)):
        return()
    
    def calculate_log_QLF_contribution(self, log_eddington_ratio, log_L, z, 
                                       parameter):
        '''
        WRITE UP
        '''
        
        if (parameter[3] <= -self.hmf_slope/parameter[1]):
            raise ValueError('Slope of ERDF smaller than (slope of HMF/'\
                              'eta). QLF integral will not converge.')

        log_phi = self._calculate_phi_contribution(log_eddington_ratio, log_L, z, 
                                                   parameter)
        
        # calculate probability for eddington_ratio
        log_erdf        = self.physics_model.at_z(z).\
                               calculate_log_erdf(log_eddington_ratio,
                                                  *parameter[2:])
                               
        # put it all together
        log_qlf_contribution = log_phi + log_erdf
        return(log_qlf_contribution)
    
    
    def _calculate_phi_contribution(self, log_eddington_ratio, log_L, z, 
                                    parameter):
        '''
        Calculate value of (HMF+feedback) function that will contribute to
        QLF.
        '''
        
        # calculate halo masses from stellar masses using model
        log_m_h = self.physics_model.at_z(z).calculate_log_halo_mass(
                       log_L, log_eddington_ratio, *parameter[:2])
        
        ## calculate value of bh phi (hmf + quasar growth model)  
        
        # calculate value of halo mass function
        log_hmf   = self.calculate_log_hmf(log_m_h, z)
        
        # calculate physics/feedback effect
        fb_factor = self.physics_model.at_z(z).calculate_dlogquantity_dlogmh(
                         log_m_h,log_eddington_ratio, *parameter[:2])
       
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',category=RuntimeWarning)
            # log of 0 is -inf, suppress corresponding numpy warning 
            log_fb_factor             = np.log10(fb_factor)
            # calculate modelled phi value (ignore inf-inf warning)
            log_phi                   = log_hmf - log_fb_factor
            
        # deal with infinite masses 
        log_phi[np.isinf(log_m_h)]    =  - np.inf
        return(log_phi)
        
    
    def _make_log_eddington_ratio_space(self, log_L, z, parameter, 
                                        log_cut=4, num=200):
        '''
        '''      
        # calculate some initial points to locate approximate location of 
        # maximum of QLF contribution
        initial_eddington_space = np.copy(self._initial_eddington_space)
        initial_qlf_points = self.calculate_log_QLF_contribution(initial_eddington_space,
                                                                 log_L,
                                                                 z,
                                                                 parameter)
        
        # find maximum of function
        max_idx              = np.argmax(initial_qlf_points)
        qlf_contribution_max = initial_qlf_points[max_idx]
        
        # find places of negligable contribution
        relative_diff         = np.abs(1 - initial_qlf_points/qlf_contribution_max)
        relevant_contribution = np.logical_not(relative_diff>log_cut)
        
        # calculate first estimate of relevant space
        relevant_eddington_space = initial_eddington_space[relevant_contribution]
        lower_limit              = relevant_eddington_space[0]
        upper_limit              = relevant_eddington_space[-1]
    
        # check if relevant part of integral is within searched space
        if (relevant_contribution[0]==False 
            and relevant_contribution[-1]==False):
            # pass if relevant space is within boundries
            pass
        else:
            # estimate slope for large Eddington ratios
            slope_inverse = ((initial_eddington_space[-1]
                              -initial_eddington_space[-2])/
                             (initial_qlf_points[-1] 
                              -initial_qlf_points[-2]))
            if (slope_inverse>0):
                raise ValueError('Slope estimate of large eddington_ratio end ' 
                                 'of QLF contribution is positive, integral '
                                 'will not converge or relevant QLF '
                                 'contribution is outside of initial Eddington '
                                 'space.')
        
            if relevant_contribution[-1]==True:
                rel_diff = np.inf
                for i in range(101):
                    upper_limit = upper_limit-slope_inverse*(log_cut+i)
                    qlf_contribution      = self.calculate_log_QLF_contribution(
                                              upper_limit,
                                              log_L, z, parameter)
                    rel_diff = np.abs(1 - qlf_contribution/qlf_contribution_max)
                    if rel_diff>log_cut:
                        break
                    
            if relevant_contribution[0]==True:
                rel_diff = np.inf
                for i in range(101):
                    upper_limit = upper_limit+slope_inverse*(log_cut+i)
                    qlf_contribution      = self.calculate_log_QLF_contribution(
                                              lower_limit,
                                              log_L, z, parameter)
                    rel_diff = np.abs(1 - qlf_contribution/qlf_contribution_max)
                    if rel_diff>log_cut:
                        break
            
            # if loop completes, raise error
            if i == 100:
                raise StopIteration('Bounds for relevant eddington_ratios could'
                                    'not be found. QLF contribution might '
                                    'converge very slowly.')
        
        
        # creat new space that contributes mainly to integral
        eddington_space         = np.linspace(lower_limit, 
                                              upper_limit,
                                              num)
        return(eddington_space)
        

class Redshift_dict():
    def __init__(self, input_dict):
        '''
        Convience Class to easily retrieve data at certain redshift.

        Parameters
        ----------
        input_dict : dict
            Input dictonary of quantity, of form {z: value}.

        Returns
        -------
        None.
        
        '''
        self.dict = input_dict
        self.list = None
        self.update_data()

    def add_entry(self, z, value):
        '''Add new entry at z to dictonary.'''
        if np.isscalar(z):
            self.dict[z] = value
        else:
            for i in range(len(z)):
                self.dict[z[i]] = value[i]
        self.update_data()
        return

    def at_z(self, z):
        ''' Retrieve data at z.'''
        if z not in list(self.dict.keys()):
            raise NameError('Redshift not in data.')
        else:
            return(self.dict[z])

    def is_None(self):
        ''' Check if dictonary is empty. '''
        return(list(self.dict.values())[0] is None)
    
    def update_data(self):
        self.data = list(self.dict.values())
        return
