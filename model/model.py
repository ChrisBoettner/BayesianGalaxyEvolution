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

from model.helper import mag_to_lum, lum_to_mag, within_bounds, make_array, \
                         system_path
from model.data.load import load_hmf_functions
from model.calibration import mcmc_fitting, leastsq_fitting
from model.quantity_options import get_quantity_specifics
from model.physics import physics_model
from model.scatter import Joint_distribution
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

    def __init__(self, redshifts, log_ndfs, quantity_name, physics_name,
                 prior_name, fitting_method, saving_mode, ndf_fudge_factor=None,
                 name_addon=None,groups=None, calibrate=True, 
                 paramter_calc=True, progress=True, **kwargs):
        '''
        Main model object. Calibrate the model by fitting parameter to
        observational data.

        Parameters
        ----------
        redshifts : int or array
            List of redshifts that are fitted.
        log_ndfs : dict
            Dictonary of number density function observations (UVLFs/SMFs
            /BHMFs) at different redshifts of form {z: ndf}. 
            Input should be logarithmic values (quantity, phi). For luminosity
            function, the quantity should be given in magnitudes.
        quantity_name : str
            Name of the quantity modelled. Must be 'Muv', 'mstar' or 'mbh'.
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
        ndf_fudge_factor : float, optional
            Fudge factor that the ndfs are multiplied by  if wanted. The 
            default is None.
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
        # put in data
        self.redshift = make_array(redshifts)
        self.log_ndfs = Redshift_dict(log_ndfs)
        self.groups = groups
        
        # load hmf related stuff
        hmf_funcs, turnover_halo_mass, halo_number = load_hmf_functions()
        self.log_hmfs               = Redshift_dict(hmf_funcs) # callable
                                                               # functions
        self.log_turnover_halo_mass = Redshift_dict(turnover_halo_mass)
        self.total_halo_number      = Redshift_dict(halo_number)
        self.hmf_slope              = 0.9  # approximate low mass slope of 
                                           # HMFs (absolute value)
        self.log_min_halo_mass      = 3    # minimum halo mass
        self.log_max_halo_mass      = 21   # maximum halo mass
        
        if ndf_fudge_factor:
            print('Careful: Number densities adjusted by a factor of '
                  f'{ndf_fudge_factor}.')
            for key in self.log_ndfs.dict.keys():
                self.log_ndfs.dict[key][:,1] = (self.log_ndfs.dict[key][:,1] +
                                                np.log10(ndf_fudge_factor))
        
        # put in model parameter
        self.quantity_name = quantity_name
        self.physics_name = physics_name
        self.prior_name = prior_name

        self.fitting_method = fitting_method
        self.saving_mode = saving_mode
        self.name_addon = name_addon
        self.progress = progress
        
        # load options related to the quantity
        self.quantity_options = get_quantity_specifics(self.quantity_name)
        self.log_m_c = self.quantity_options['log_m_c']
        if physics_name not in self.quantity_options['physics_models']:
            raise NotImplementedError('Physics model not implemented for this '
                                      'quantity.')

        # location where files will be saved or loaded from
        self.directory =  system_path() + self.quantity_name + '/' \
                          + self.physics_name + '/'

        # empty dicts to be filled
        self.physics_model = Redshift_dict({})
        self.filename = Redshift_dict({})
        self.parameter = Redshift_dict({})
        self.distribution = Redshift_dict({})
        
        # load parameter
        if (saving_mode == 'loading'):
            try:
                parameter = load_parameter(self, name_addon)
                parameter = {int(z): p for z, p in parameter.items()}
                self.parameter = Redshift_dict(parameter)
            except FileNotFoundError:
                warnings.warn('Could not load best fit parameter')
                
        # choose if fitting should be done or not
        self.calibrate = calibrate
        if self.calibrate:
            self.fit_model(self.redshift, **kwargs)
        else:
            for z in self.redshift:
                self._add_physics_model(z)


        # default plot parameter per physics_model
        if self.physics_name == 'none':
            self._plot_parameter('black', 'o', '-', 'No Feedback')
        elif self.physics_name == 'stellar':
            self._plot_parameter('lightgrey', 's', '-', 'Stellar Feedback')
        elif self.physics_name == 'stellar_blackhole':
            self._plot_parameter(
                'C3', 'v', '-', 'Stellar + Black Hole Feedback')
        elif self.physics_name == 'changing':
            feedback_change_z = self.quantity_options['feedback_change_z']
            max_z = 10
            self._plot_parameter(
                ['C3'] * feedback_change_z + ['lightgrey'] *
                (max_z+1-feedback_change_z),
                'o', '-',
                ['Stellar + Black Hole Feedback'] * feedback_change_z
                + ['Stellar Feedback'] * (max_z+1-feedback_change_z))
        elif self.physics_name == 'quasar':
            self._plot_parameter(
                'C3', '^', '-', 'BH Growth Model')
        elif self.physics_name == 'eddington':
            self._plot_parameter(
                'C3', 'v', '-', 'Bolometric Luminosity Model')
        elif self.physics_name == 'eddington_free_ERDF':
            self._plot_parameter(
                'grey', '^', '-', 'Bolometric Luminosity Model (free ERDF)')
        elif self.physics_name == 'eddington_changing':
            feedback_change_z = self.quantity_options['feedback_change_z']
            max_z = 7
            self._plot_parameter(
                ['C3'] * feedback_change_z + ['grey'] *
                (max_z+1-feedback_change_z),
                'o', '-',
                ['eddington_free_ERDF'] * feedback_change_z
                + ['eddington'] * (max_z+1-feedback_change_z))
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
        custom_bar_flag = ((self.progress and
                            self.fitting_method == 'mcmc' and
                            not (self.saving_mode == 'loading'))
                           or (not self.progress))
        if custom_bar_flag:
            PBar = NullBar()
        else:
            PBar = ProgressBar(widgets=[FormatLabel('')])

        redshifts = make_array(redshifts)

        # run fits
        distributions = {}
        posterior_samp, bounds = None, None
        for z in PBar(redshifts):
            # progress tracking
            if not custom_bar_flag:
                model_details = self.quantity_options['ndf_name'] + ' - ' +\
                    'physics('+self.physics_name + ') - ' +\
                    self.prior_name + ' prior: '
                PBar.widgets[0] = model_details + f'z = {z}'
            self._z = z  # temporary storage for current redshift

            # add file name
            filename = self.prior_name + '_z' + str(z)
            # if manual modification of saving path is wanted
            if self.name_addon:
                filename = filename + ''.join(self.name_addon)
            self.filename.add_entry(z, filename)

            # add physics model
            self._add_physics_model(z)

            # create new prior from distribution of previous iteration
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
                    progress=self.progress, **kwargs)
            else:
                raise NameError('fitting_method not known.')
                
            if not self.saving_mode == 'loading':    
                # add parameter to model, but only if not loaded from external
                # file
                self.parameter.add_entry(z, parameter)
            
            # add distributions to model
            self.distribution.add_entry(z, posterior_samp)
                
            distributions[z] = posterior_samp

            if (not custom_bar_flag) and (z == redshifts[-1]):
                PBar.widgets[0] = model_details + 'DONE'
        return

    def calculate_log_abundance(self, log_quantity, z, parameter,
                                hmf_z=None, scatter_name='delta',
                                scatter_parameter=None, **kwargs):
        '''
        Calculate (log of) value (phi) of modelled number density function by 
        multiplying HMF function with physics model derivative for a given
        redshift.

        IMPORTANT: Input units must be log m in units of solar masses for
        the mass functions and UV luminosity in absolute mag for UVLF.
        
        EXPERIMENTAL: Can include scatter in the calculation, not tested or
        implemented thoroughly though.

        Parameters
        ----------
        log_quantity : float or array
            Input (log of) observable quantity. For mass function must be in
            stellar masses, for luminosities must be in absolute magnitudes.
        z : int
            Redshift at which value is calculated.
        parameter : list 
            Model parameter used for calculation.
        hmf_z : int, optional
            If hmf_z is given, use that reshift for calculating the values of
            the halo mass function. Useful for disentangeling baryonic physics
            and HMF evolution. The default is None.
        scatter_name: str, optional
            Name of distribution that describes scatter in quantity-
            halo mass relation. The default is 'delta', which means no scatter.
        scatter_parameter: float, optional
            Value of scatter distribution that describes spread. scale 
            parameter in scipy implementation of location-scale distributions.
            The default is 'none', must be set if scatter model other than 
            'delta' is used.
        **kwargs: dict, optional
            Additional parameter passed to 
            _experimental_log_abundance_with_scatter method.

        Returns
        -------
        log_phi : float
            Log value of ndf at the input value and redshift.

        '''
        log_quantity = make_array(log_quantity)
        
        # check that parameters are within bounds
        if not within_bounds(parameter, *self.physics_model.at_z(z).bounds):
            raise ValueError('Parameter out of bounds.')

        # set hmf redshift
        if hmf_z is None:
            hmf_z = z

        if scatter_name == 'delta':
            # conversion between magnitude and luminosity if needed
            # (for _experimental_log_abundance_with_scatter this is done in
            #  Joint_distribution class)
            log_quantity = self.unit_conversion(log_quantity, 'mag_to_lum')
            
            # calculate halo masses from stellar masses using model
            log_m_h = self.physics_model.at_z(z).calculate_log_halo_mass(
                log_quantity, *parameter)
            # calculate value of halo mass function
            log_hmf = self.calculate_log_hmf(log_m_h, hmf_z)
            # calculate physics/feedback effect (and deal with zero values)
            ph_factor = self.physics_model.at_z(z).\
                    calculate_dlogquantity_dlogmh(log_m_h, *parameter)
            with warnings.catch_warnings():
                # log of 0 is -inf, suppress corresponding numpy warning
                warnings.simplefilter('ignore', category=RuntimeWarning)
                log_ph_factor = np.log10(ph_factor)
                log_phi = log_hmf - log_ph_factor # calculate modelled phi value
            # deal with infinite masses
            log_phi[np.isinf(log_m_h)] = -np.inf
            
        else:
            if scatter_parameter is None:
                raise ValueError('scatter_parameter must be specific if '
                                 'scatter model other than delta is used.')
            log_phi = self._experimental_log_abundance_with_scatter(
                        log_quantity, z, parameter, hmf_z,
                        scatter_name, scatter_parameter, **kwargs)
            
        return(log_phi)

    def draw_parameter_sample(self, z, num=1):
        '''
        Get a sample from physics model parameter distribution at given 
        redshift.

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
            raise AttributeError('distribution dictonary is empty. Probably'
                                 ' wasn\'t calculated.')

        # randomly draw from parameter distribution at z
        random_draw = np.random.choice(self.distribution.at_z(z).shape[0],
                                       size=num)
        parameter_sample = self.distribution.at_z(z)[random_draw]
        return(parameter_sample)

    def calculate_quantity_distribution(self, log_halo_mass, z, num=int(1e+5)):
        '''
        At a given redshift, calculate distribution of observable quantity
        (mstar/Muv/mbh) for a given halo mass by drawing parameter sample and
        calculating value for each one.

        Parameters
        ----------
        log_halo_mass : float or list
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
            
        # conversion between magnitude and luminosity if needed
        log_quantity_dist = self.unit_conversion(log_quantity_dist,
                                                 'lum_to_mag')
        return(np.array(log_quantity_dist))

    def calculate_halo_mass_distribution(self, log_quantity, z, num=int(1e+5)):
        '''
        At a given redshift, calculate distribution of halo mass for a given
        observable quantity (mstar/Muv/mbh) by drawing parameter sample and
        calculating value for each one (number of draws adaptable).

        Parameters
        ----------
        log_quantity : float
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
        # conversion between magnitude and luminosity if needed
        log_quantity = self.unit_conversion(log_quantity, 'mag_to_lum')
        
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
        log_quantity : float
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
        # conversion between magnitude and luminosity if needed
        log_quantity = self.unit_conversion(log_quantity, 'mag_to_lum')

        parameter_sample = self.draw_parameter_sample(z, num=num)

        log_abundance_dist = []
        for p in parameter_sample:
            log_abundance_dist.append(
                self.calculate_log_abundance(log_quantity, z, p))
        return(np.array(log_abundance_dist))

    def calculate_ndf(self, z, parameter, quantity_range=None, hmf_z=None):
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
        hmf_z : int, optional
            If hmf_z is given, use that reshift for calculating the values of
            the halo mass function. Useful for disentangeling baryonic physics
            and HMF evolution. The default is None.

        Returns
        -------
        ndf : array
            Calculated number density functions.

        '''
        if quantity_range is None:
            quantity_range = self.quantity_options['quantity_range']

        ndf = self.calculate_log_abundance(quantity_range, z, parameter,
                                           hmf_z)
        return([quantity_range, ndf])

    def calculate_log_hmf(self, log_halo_mass, z):
        '''
        Calculate value of Halo Mass Function for a given halo mass and
        redshift. If any of the values are nan, it's assumed they're 
        were supposed to be extrapolated at the high mass end, which should
        result in a value close to -inf.

        Parameters
        ----------
        log_quantity : float
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
            warnings.simplefilter('ignore', category=RuntimeWarning)
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
    
    def calculate_feedback_regimes(self, z, parameter, log_epsilon=-1, 
                                   output='quantity'):
        '''
        Calculate (log of) quantity values at which feedback regime changes.
        For stellar and AGN feedback: Calculate values where one of the 
        feedbacks is strongly dominating in the sense that 
        (M_h/M_c)^-alpha > epsilon * (M_h/M_c)^beta and 
        the other way around. 
        Returns 3 values: [log_q(log_m_c), log_q(log_m_sn), log_q(log_m_ bh)].
        If output='halo_mass', return halo masses for these values instead.

        Parameters
        ----------
        z : int
            Redshift at which value is calculated.
        parameter : list
            Input model parameter.
        log_epsilon : float, optional
            Threshold for regime change. The default is -1.
        output : str, optional
            Choose if 'quantity' or 'halo_mass' should be 
            returned. The default is 'quantity'.

        Raises
        ------
        NotImplementedError
            So far, feedback regimes are only implemented for stellar + AGN
            feedback models.

        Returns
        -------
        regime_values: array
            Array of regime change values in order
            [log_q(log_m_c), log_q(log_m_sn), log_q(log_m_ bh)] (for stellar
             + AGN feedback model).

        '''
        if self.quantity_name not in ['mstar', 'Muv']:
            raise NotImplementedError('calculate_feedback_regimes not yet '
                                      'implemented for this physics model')
        # calculate transition values    
        regime_values = self.physics_model.at_z(z)._calculate_feedback_regimes(
                            *parameter, log_epsilon=-1, output=output)
        
        # conversion between magnitude and luminosity if needed
        regime_values = self.unit_conversion(regime_values, 'lum_to_mag')
        return(regime_values)
                                
    
    def unit_conversion(self, log_quantity, mode): 
        '''
        Perform necessary unit conversions. For Muv model, convert between 
        (input) magnitudes and (internal) luminosities.

        Parameters
        ----------
        log_quantity : float
            Input (log of) observable quantity.

        Returns
        -------
        log_quantity : float
            Converted quantity (magnitude or luminosity).

        '''
        if self.quantity_name == 'Muv':
            # convert magnitude to luminosity
            if mode == 'mag_to_lum' and np.all(log_quantity<=0):
                log_quantity = np.log10(mag_to_lum(log_quantity))
            elif mode == 'lum_to_mag' and np.all(log_quantity>=0):
                log_quantity = lum_to_mag(np.power(10,log_quantity))
            else:
                raise NameError('Something went wrong in unit conversion.')
        return(log_quantity)


    def _add_physics_model(self, z):
        '''
        Add physics model to general model according to physics_name.

        Parameters
        ----------
        z : int
            Redshift for which physics model is added.

        Returns
        -------
        None.

        '''
        # create physics model
        if self.physics_name in ['none', 'stellar', 'stellar_blackhole',
                                 'quasar']:
            ph_name = self.physics_name
        elif self.physics_name == 'changing':  # standard changing feedback
            feedback_change_z = self.quantity_options['feedback_change_z']
            if z < feedback_change_z:
                ph_name = 'stellar_blackhole'
            elif z >= feedback_change_z:
                ph_name = 'stellar'
        else:
            raise NameError('physics_name not known.')
        # add model
        self.physics_model.add_entry(z, physics_model(
            ph_name,
            self.log_m_c,
            initial_guess=self.quantity_options['model_p0'],
            bounds=self.quantity_options['model_bounds']))
        return
    
    def _experimental_log_abundance_with_scatter(
            self, log_quantity, z, parameter, hmf_z,
            scatter_name, scatter_parameter, **kwargs):  
        '''
        Experimental implementation of including scatter in the quantity-
        halo mass relation. Should be used with caution.

        Parameters
        ----------
        log_quantity : float or array
            Input (log of) observable quantity. For mass function must be in
            stellar masses, for luminosities must be in absolute magnitudes.
        z : int
            Redshift at which value is calculated.
        parameter : list 
            Model parameter used for calculation.
        hmf_z : int, optional
            If hmf_z is given, use that reshift for calculating the values of
            the halo mass function. Useful for disentangeling baryonic physics
            and HMF evolution. The default is None.
        scatter_name: str, optional
            Name of distribution that describes scatter in quantity-
            halo mass relation. The default is 'delta', which means no scatter.
        scatter_value: float, optional
            Value of scatter distribution that describes spread. scale 
            parameter in scipy implementation of location-scale distributions.
            The default is 'none', must be set if scatter model other than 
            'delta' is used.
        num : int, optional
            Number of samples created for integral evaluation. The default 
            is int(1e+5).
        **kwargs: dict, optional
            kwargs passed to calculate_quantity_marginal_density method of
            Joint_distribution.

        Returns
        -------
        None.

        '''
        if hmf_z != z:
            raise NotImplementedError('hmf_z not implemented for calculation '
                                      'with scatter.') 
        log_quantity = make_array(log_quantity)    
        
        distribution = Joint_distribution(self, scatter_name,
                                          scatter_parameter)
        phi = distribution.quantity_marginal_density(log_quantity, z,
                                                     parameter, 
                                                     **kwargs)
        return(np.log10(phi))

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
        self.color = color
        self.marker = marker
        self.linestyle = linestyle
        self.label = label
        return


class ModelResult_QLF(ModelResult):
    '''
    An adapated Model class used for calculating the Quasar Luminosity Function
    (QLF). For this quantity, we need a changed methodology that takes, so that 
    things need to be adapted.
    Two physics models implemented. 'eddington_free_ERDF' fits the Eddington
    Rate Distribution Function at every redshift, while 'eddington' only fits
    the ERDF parameter at the first redshift and then reuses the parameter for 
    higher redshifts.
    '''

    def __init__(self, redshifts, log_ndfs,
                 quantity_name, physics_name, prior_name,
                 fitting_method, saving_mode, ndf_fudge_factor=None,
                 name_addon=None, groups=None, calibrate=True, 
                 paramter_calc=True, progress=True, **kwargs):
        '''
        Main model object for QLF. Calibrate the model by fitting parameter to
        observational data.

        Parameters
        ----------
        See ModelResult parent object for information on most arguments.

        quantity_name : str
            Name of the quantity modelled. Must be 'Lbol'.
        physics_name : str
            Name of physics model. Must be in implemented models in 
            quantity_options. 'eddington' fits ERDF at first redshift and then
            uses best fit parameter, 'eddington_free_ERDF' fits ERDF at every
            redshift. 'eddington_changing' uses free ERDF up to z=2 and then fixes it
            for higher redshifts.

        Returns
        -------
        None.

        '''
        # used for make_space, so it doesn't have to be recreated every time
        self._initial_eddington_space = np.linspace(-50, 50, 100)
        self._initial_erdf = None

        # log of constant for Eddington luminosity, where
        # L_Edd = eddington_constant * m_bh (in erg/s)
        self.log_eddington_constant = 38.1

        # initalize model itself
        super().__init__(redshifts, log_ndfs,
                         quantity_name, physics_name, prior_name,
                         fitting_method, saving_mode, ndf_fudge_factor, 
                         name_addon, groups, calibrate, paramter_calc,
                         progress, **kwargs)

    def calculate_log_abundance(self, log_L, z, parameter, hmf_z=None,
                                num=100):
        '''
        Calculate (log of) value (phi) of modelled number density function by 
        integrating (HMF+feedback)*ERDF over eddington_ratios for a given
        redshift.

        Parameters
        ----------
        log_L : float or array
            Input (log of) bolometric lunionsity in ergs/s.
        z : int
            Redshift at which value is calculated.
        parameter : list 
            Model parameter used for calculation.
        num : int
            Number of points evaluating for integral.

        Returns
        -------
        log_phi : float
            Log value of ndf at the input value and redshift.

        '''
        log_L = make_array(log_L)

        # check that parameters are within bounds
        if not within_bounds(parameter, *self.physics_model.at_z(z).bounds):
            raise ValueError('Parameter out of bounds.')
        
        # set hmf redshift
        if hmf_z is not None:
            raise NotImplementedError('Fixing HMF z is not yet implemented for '
                                      'Lbol model. To do so: Add it to '
                                      'calculate_log_abundance, '
                                      'calculate_log_QLF_contribution, '
                                      'calculate_phi_contribution, so that '
                                      'hmf_z is called in calculate_log_hmf.')
        
        phi = []
        for L in log_L:
            # estimate relevant Eddington ratios that contribute
            eddington_ratio_space = self.make_log_eddington_ratio_space(L, z,
                                                                        parameter,
                                                                        num=num)
            # calculate QLF contribution at these redhifts
            log_qlf_contribution = self.\
                calculate_log_QLF_contribution(eddington_ratio_space,
                                               L,
                                               z,
                                               parameter)
            # integrate over the contributions and append to list
            phi.append(trapezoid(np.power(10, log_qlf_contribution),
                       eddington_ratio_space))

        # calculate log and deal with zero values
        phi = np.array(phi)
        log_phi = np.empty_like(phi)
        log_phi[phi == 0] = -np.inf
        log_phi[phi != 0] = np.log10(phi[phi != 0])
        return(log_phi)

    def calculate_log_QLF_contribution(self, log_eddington_ratio, log_L, z,
                                       parameter):
        '''
        Calculate contribution to QLF for a given eddington ratio.

        Parameters
        ----------
        log_eddington_ratio : float or array
            Input (log of) eddington ratio.
        log_L : float
            Input (log of) bolometric lunionsity in ergs/s.
        z : int
            Redshift at which value is calculated.
        parameter : list 
            Model parameter used for calculation.

        Raises
        ------
        ValueError
            Integral only converges if ERDF smaller than (slope of HMF/'\'eta),
            raise error if this is not the case.

        Returns
        -------
        log_qlf_contribution: float or array
            Contribution for given log_eddington_ratio.

        '''

        # calculate contribution from HMF+feedback
        log_phi = self.calculate_phi_contribution(log_eddington_ratio,
                                                   log_L, z,
                                                   parameter)
        # calculate contribution from ERDF
        log_erdf = self.calculate_ERDF_contribution(log_eddington_ratio,
                                                    z, parameter)
        # put it together
        log_qlf_contribution = log_phi + log_erdf

        # check if parameter are sensible
        if (self.physics_model.at_z(z).parameter[1]
            <= (1+self.hmf_slope/parameter[1])):
            raise ValueError('Slope of ERDF smaller than 1+(slope of HMF/'
                             'eta). QLF integral will not converge.')
        return(log_qlf_contribution)
    
    def calculate_quantity_distribution(self, log_halo_mass, z, 
                                        log_eddington_ratio=None, 
                                        num=int(2e+3)):
        '''
        At a given redshift, calculate distribution of observable quantity
        (lbol) for a given halo mass by drawing parameter sample and
        calculating value for each one. 
        If log_eddington_ratio is None, draw from values from ERDF and combine
        them with random parameter picks. If log_eddington_ratio is given, use
        this fixed value and only sample the remaining parameter (log_C and 
        eta).

        Parameters
        ----------
        log_halo_mass : float or list
            Input (log) halo masses for which quantity distribution is 
            caluclated.
        z : int
            Redshift at which value is calculated.
        log_eddington_ratio : float
            Fix (log) Eddington ratio. If value is None, draw sample 
            randomly from ERDF.
        num : int, optional
            Number of samples drawn. The default is int(1e+5).

        Returns
        -------
        log_quantity_dist : array
            Calculated distribution.

        '''
        
        # draw parameter sample and calculate Eddington ratios,
        # if Eddington ratio is given: repeat that one
        # if model has fixed Edd distribution: draw from that one
        # if model has varying Edd distribution: draw one Edd ratio from each
        # distribution per parameter
        
        parameter_sample = self.draw_parameter_sample(z, num=num)
        if log_eddington_ratio is not None:
            log_edd_ratio = np.repeat(log_eddington_ratio, num)
            parameter     = parameter_sample[:,:2] # for calculate_log_quantity
                                                   # only first two parameter 
                                                   # are used
        else:
            if self.physics_model.at_z(z).name == 'eddington':
                log_edd_ratio = np.repeat(self.physics_model.at_z(z).\
                                          draw_eddington_ratio(), num)
                parameter     = parameter_sample # should only contain the two
                                                 # parameter
            elif self.physics_model.at_z(z).name == 'eddington_free_ERDF':
                log_edd_ratio = np.array([])
                for p in parameter_sample[:,2:]:
                    log_edd_ratio = np.append(log_edd_ratio, 
                                              self.physics_model.at_z(z).\
                                                   draw_eddington_ratio(*p))
                parameter     = parameter_sample[:,:2]
            else:
                raise NameError('physics_name not known.')
        
        # calculate quantity for each parameter - eddington ratio pair
        log_quantity_dist = []        
        for i in range(len(parameter_sample)):
                log_quantity_dist.append(
                    self.physics_model.at_z(z).calculate_log_quantity(
                        log_halo_mass, log_edd_ratio[i], *parameter[i]))   
        return(np.array(log_quantity_dist))
    
    def calculate_halo_mass_distribution(self, log_L, z,
                                         log_eddington_ratio=None, 
                                         num=int(2e+3)):
        '''
        At a given redshift, calculate distribution of halo mass for a given
        observable quantity (lbol) by drawing parameter sample and
        calculating value for each one (number of draws adaptable).
        If log_eddington_ratio is None, draw from values from ERDF and combine
        them with random parameter picks. If log_eddington_ratio is given, use
        this fixed value and only sample the remaining parameter (log_C and 
        eta).
        Parameters
        ----------
        log_quantity : float
            Input (log of) observable quantity. For mass function must be in
            stellar masses, for luminosities must be in absolute magnitudes.
        z : int
            Redshift at which value is calculated.
        log_eddington_ratio : float
            Fix (log) Eddington ratio. If value is None, draw sample 
            randomly from ERDF.
        num : int, optional
            Number of samples drawn. The default is int(1e+5).

        Returns
        -------
        log_halo_mass_dist : array
            Calculated distribution.

        '''
        # draw parameter sample and calculate Eddington ratios,
        # if Eddington ratio is given: repeat that one
        # if model has fixed Edd distribution: draw from that one
        # if model has varying Edd distribution: draw one Edd ratio from each
        # distribution per parameter
        parameter_sample = self.draw_parameter_sample(z, num=num)
        if log_eddington_ratio is not None:
            log_edd_ratio = np.repeat(log_eddington_ratio, num)
            parameter     = parameter_sample[:,:2] # for calculate_log_quantity
                                                   # only first two parameter 
                                                   # are used
        else:
            if self.physics_model.at_z(z).name == 'eddington':
                log_edd_ratio = np.repeat(self.physics_model.at_z(z).\
                                          draw_eddington_ratio(), num)
                parameter     = parameter_sample # should only contain the two
                                                 # parameter
            elif self.physics_model.at_z(z).name == 'eddington_free_ERDF':
                log_edd_ratio = np.array([])
                for p in parameter_sample[:,2:]:
                    log_edd_ratio = np.append(log_edd_ratio, 
                                              self.physics_model.at_z(z).\
                                                   draw_eddington_ratio(*p))
                parameter     = parameter_sample[:,:2]
            else:
                raise NameError('physics_name not known.')
        
        # calculate quantity for each parameter - eddington ratio pair
        log_quantity_dist = []        
        for i in range(len(parameter_sample)):
                log_quantity_dist.append(
                    self.physics_model.at_z(z).calculate_log_halo_mass(
                        log_L, log_edd_ratio[i], *parameter[i]))   
        return(np.array(log_quantity_dist))

    def calculate_phi_contribution(self, log_eddington_ratio, log_L, z,
                                    parameter):
        '''
        Calculate value of (HMF+feedback) function that will contribute to
        QLF.

        Parameters
        ----------
        log_eddington_ratio : float or array
            Input (log of) eddington ratio.
        log_L : float
            Input (log of) bolometric lunionsity in ergs/s.
        z : int
            Redshift at which value is calculated.
        parameter : list 
            Model parameter used for calculation.

        Returns
        -------
        log_phi: float or array
            Contribution of (HMF+feedback). (Different log_phi from final 
            result.)

        '''
        # calculate halo masses from stellar masses using model
        log_m_h = self.physics_model.at_z(z).calculate_log_halo_mass(
            log_L, log_eddington_ratio, *parameter[:2])

        # calculate value of halo mass function
        log_hmf = self.calculate_log_hmf(log_m_h, z)

        # calculate physics/feedback effect
        ph_factor = self.physics_model.at_z(z).calculate_dlogquantity_dlogmh(
            log_m_h, log_eddington_ratio, *parameter[:2])

        # calculate final result, deal with raised warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)

            # log of 0 is -inf, suppress corresponding numpy warning
            log_ph_factor = np.log10(ph_factor)
            # calculate modelled phi value (ignore inf-inf warning)
            log_phi = log_hmf - log_ph_factor
            # deal with infinite masses
            log_phi[np.isinf(log_m_h)] = -np.inf
        return(log_phi)

    def calculate_ERDF_contribution(self, log_eddington_ratio, z,
                                     parameter):
        '''
        Calculate ERDF contribution to QLF.

        Parameters
        ----------
        log_eddington_ratio : float or array
            Input (log of) eddington ratio.
        log_L : float
            Input (log of) bolometric lunionsity in ergs/s.
        z : int
            Redshift at which value is calculated.
        parameter : list 
            Model parameter used for calculation.

        Raises
        ------
        NameError
            Error in case the physics model is not known. Should not occur
            since name is already checked when model is initialized.

        Returns
        -------
        log_erdf: float or array
            Contribution of ERDF.

        '''
        # if physics model is 'eddington', the ERDF is fixed to a specific
        # value. In this case, we can first check if the value is calculated
        # for the _initial_eddington_space usd in make_space. If so, use the
        # stored result instead of recalculating every time. If not, do the
        # calculation using fixed ERDF.
        if self.physics_model.at_z(z).name == 'eddington':
            if np.array_equal(log_eddington_ratio,
                              self._initial_eddington_space):
                return(self._initial_erdf)
            else:
                log_erdf = self.physics_model.at_z(z).\
                    calculate_log_erdf(log_eddington_ratio)

        # if the model is 'eddington_free_ERDF', the ERDF parameter are part
        # of the model parameter (last two parameter). In that case, call
        # physics function with these parameter.
        elif self.physics_model.at_z(z).name == 'eddington_free_ERDF':
            log_erdf = self.physics_model.at_z(z).calculate_log_erdf(
                log_eddington_ratio,
                *parameter[2:])
        else:
            raise NameError('physics_name not known.')
        return(log_erdf)
    
    def calculate_conditional_ERDF(self, log_L, z, parameter,
                                   eddington_ratio_space = None,
                                   black_hole_mass_distribution=False):
        '''
        Calculates the conditional ERDF for the given (log) luminosity at the
        given redshift using the given parameter. Returns
        dict of form log_L:[log eddington ratio, probability]. 
        Eddington ratio space can be an given as optional argument, if None 
        it's created using make_log_eddington_ratio_space.
        If black_hole_mass_distribution is True, calculate black hole masses 
        associated with Eddington ratios and return black hole mass 
        distribution instead, i.e. dict of form 
        log_L:[log black hole mass, probability].
        
        Parameters
        ----------
        log_L : float
            Input (log of) bolometric lunionsity in ergs/s.
        z : int
            Redshift at which value is calculated.
        parameter : list 
            Model parameter used for calculation.
        eddington_ratio_space : float or array, optional
            Eddington ratio space over which QLF contribution is supposed to be
            calculated. The default is None.
        black_hole_mass_distribution : bool, optional
            If True, transform Eddington ratios to black hole masses and 
            return probability distribution of black hole masses instead of
            ERDF.

        Returns
        -------
        distribution: dict
            Dictonary of calculated conditional ERDFs (or black hole mass
            distributions if black_hole_mass_distribution=True) (log_L are 
            keys). 

        '''
        log_L = make_array(log_L)
        distribution = {}
        for l in log_L:
            if eddington_ratio_space is None:
                eddington_ratio_space = self.make_log_eddington_ratio_space(l, z, 
                                                                parameter,
                                                                num=1000)
            qlf_con          = np.power(10, self.calculate_log_QLF_contribution(
                                                                   eddington_ratio_space,
                                                                   l, z,
                                                                   parameter))
            # the contribution has to be normalised
            normalisation     = trapezoid(qlf_con, eddington_ratio_space)
            conditional_erdf  = np.array([eddington_ratio_space,
                                            qlf_con/normalisation]).T
            
            if black_hole_mass_distribution:
                # calculate (log of) black hole masses associated with 
                # Eddington ratio and overwrite Eddingtion ratio column with
                # black hole masses
                log_m_bhs = (l - conditional_erdf[:,0] 
                             - self.log_eddington_constant)
                conditional_erdf[:,0] = log_m_bhs
                # sort according to black hole masses
                conditional_erdf = conditional_erdf[log_m_bhs.argsort()]
            
            distribution[l] = conditional_erdf
        return(distribution)

    def calculate_expected_log_black_hole_mass_from_ERDF(self, log_L, z, 
                                                         parameter):
        '''
        Calculates the expected (log of) black hole mass for given
        (log) luminosity, redshift and parameter mean value is calculated
        by first calculating the expected black hole mass distribution for the 
        given luminosity (and redshift+parameter) and then calculating the 
        expectation value of that distribution.
        
        Parameters
        ----------
        log_L : float or array
            Input (log of) bolometric lunionsity in ergs/s.
        z : int
            Redshift at which value is calculated.
        parameter : list 
            Model parameter used for calculation.
            
        Returns
        -------
        mean_log_black_hole_mass: float or array
            Array of (log of) expected black hole masses.

        '''
        
        
        # calculate black hole mass distributions
        black_hole_mass_distributions = self.calculate_conditional_ERDF(
                                            log_L, z, parameter,
                                            black_hole_mass_distribution=True)
        
        # calculate mean value from distribution
        mean_log_black_hole_mass = []
        for l in black_hole_mass_distributions.keys():
            log_mbh, prob = black_hole_mass_distributions[l].T
            mean_mbh      = trapezoid(np.power(10,log_mbh)*prob, log_mbh)
            mean_log_black_hole_mass.append(np.log10(mean_mbh))
        mean_log_black_hole_mass = np.array(mean_log_black_hole_mass)
            
        # if input scalar, return scalar
        if np.isscalar(log_L):
            mean_log_black_hole_mass = mean_log_black_hole_mass[0]
        return(mean_log_black_hole_mass)
        

    def make_log_eddington_ratio_space(self, log_L, z, parameter,
                                       log_cut=6, num=100):
        '''
        Only a small range of Eddington ratios contributes meaninfully to the
        QLF integral. This function estimates that ranges and creates a 
        linspace over this to calculate the integral

        Parameters
        ----------
        log_L : float
            Input (log of) bolometric lunionsity in ergs/s.
        z : int
            Redshift at which value is calculated.
        parameter : list 
            Model parameter used for calculation.
        log_cut : float
            The minimum relative difference between the QLF contribution 
            maximum and the value at the outer edges of the relevant eddington
            ratio space (as log value).
        num : int
            Number of points created in the linspace, which is used for 
            evaluating the integral

        Raises
        ------
        ValueError
            Error in case the estimated slope is postive. This either means
            the maximum of the contribution lies outside of the initially 
            search area or model parameter are chosen in such a way that the
            slope does not converge (second situation should not happen since
            we check for this beforehand.)
        StopIteration
            Error raised in case one of the loops completes when searching
            for the bounds of the relevant eddington ratios. Should in 
            principle not occur if everything is well behaved.

        Returns
        -------
        log_eddington_space: array
            Linspace over relevant (log) eddington ratios.

        '''
        # calculate some initial points to locate approximate location of
        # maximum of QLF contribution
        initial_eddington_space = np.copy(self._initial_eddington_space)
        initial_qlf_points = self.calculate_log_QLF_contribution(
            initial_eddington_space,
            log_L,
            z,
            parameter)

        # find maximum of function
        max_idx = np.argmax(initial_qlf_points)
        qlf_contribution_max = initial_qlf_points[max_idx]

        # find places of negligable and relevant contribution
        relative_diff = np.abs(1-initial_qlf_points/qlf_contribution_max)
        relevant_contribution = np.logical_not(relative_diff > log_cut)

        # calculate first estimate of relevant space
        relevant_eddington_space = initial_eddington_space[relevant_contribution]
        lower_limit = relevant_eddington_space[0]
        upper_limit = relevant_eddington_space[-1]

        # check if relevant part of integral is within searched space
        if (relevant_contribution[0] == False
                and relevant_contribution[-1] == False):
            # pass if relevant space is within boundaries
            pass
        # if not, further refine space
        else:
            # estimate slope for large Eddington ratios
            delta_qlf_cont = (initial_qlf_points[-1]-initial_qlf_points[-2])
            delta_edd = (initial_eddington_space[-1]
                         - initial_eddington_space[-2])
            slope_inverse = delta_edd/delta_qlf_cont

            if (slope_inverse > 0):
                raise ValueError('Slope estimate of large eddington_ratio end '
                                 'of QLF contribution is positive, integral '
                                 'will not converge or relevant QLF '
                                 'contribution is outside of initial '
                                 'Eddington space.')

            # if upper end of Eddington space still contributes more than
            # cutoff, successively increase upper bound using the slope
            # estimate (and approximation of power law at high eddington
            # ratios) until upper limit to relevant contribution is found
            i,j = 0, 0
            if relevant_contribution[-1] == True:
                rel_diff = np.inf
                for i in range(101):
                    upper_limit = upper_limit-slope_inverse*(log_cut+i)
                    qlf_contribution = self.calculate_log_QLF_contribution(
                        upper_limit,
                        log_L, z, parameter)
                    rel_diff = np.abs(1 - qlf_contribution /
                                      qlf_contribution_max)
                    if rel_diff > log_cut:
                        break
            # if lower end still contributes, do the same (This should happen
            # much more rarely, if at all, since on this side it drops
            # exponentially. If it happens, it might be a sign that the
            # maximum of the distribution is at lower eddington ratios than
            # initially searched for)
            if relevant_contribution[0] == True:
                rel_diff = np.inf
                for j in range(101):
                    upper_limit = upper_limit+slope_inverse*(log_cut+j)
                    qlf_contribution = self.calculate_log_QLF_contribution(
                        lower_limit,
                        log_L, z, parameter)
                    rel_diff = np.abs(1 - qlf_contribution /
                                      qlf_contribution_max)
                    if rel_diff > log_cut:
                        break

            # if either of the loops complete, raise error
            if (j == 100) or (i == 100):
                raise StopIteration('Bounds for relevant eddington_ratios '
                                    'could not be found. QLF contribution '
                                    'might converge very slowly.')

        # creat new space that contributes mainly to integral
        log_eddington_space = np.linspace(lower_limit,
                                          upper_limit,
                                          num)
        return(log_eddington_space)

    def _add_physics_model(self, z):
        '''
        Add physics model to general model according to physics_name. If 
        physics model is eddington_free_ERDF, fit ERDF at every redshift.
        If physics model is eddington, fit ERDF at first redshift and then
        reuse these parameters.

        Parameters
        ----------
        z : int
            Redshift for which physics model is added.

        Returns
        -------
        None.

        '''
        # create physics model
        if self.physics_name == 'none':
            ph_name = self.physics_name
            eddington_erdf_params = None
        
        elif self.physics_name == 'eddington_free_ERDF':
            ph_name = self.physics_name
            eddington_erdf_params = None

        elif self.physics_name == 'eddington':
            # for 'eddington' use free ERDF at initial redshift and fixed
            # ERDF afterwards
            if z == self.redshift[0]:
                ph_name = 'eddington_free_ERDF'
                eddington_erdf_params = None
            else:
                ph_name = 'eddington'
                if self.calibrate:
                    # use parameter at first redshift
                    try:
                        eddington_erdf_params = self.parameter.at_z(
                                                        self.redshift[0])[2:]
                    except:
                        raise NameError('Cannot look up MAP estimate for ERDF '
                                        'parameter because parameter have not '
                                        'been loaded.')             
                else:
                    eddington_erdf_params = None
                    
        elif self.physics_name == 'eddington_changing':
            # for 'eddingto_free_ERDF' fit ERDF up to z=2 
            feedback_change_z = self.quantity_options['feedback_change_z']
            if z < feedback_change_z:
                ph_name = 'eddington_free_ERDF'
                eddington_erdf_params = None
            else:
                ph_name = 'eddington'
                if self.calibrate:
                    # use parameter last free ERDF redshift
                    eddington_erdf_params = self.parameter.at_z(
                                                    feedback_change_z-1)[2:]
                else:
                    eddington_erdf_params = None
        else:
            raise NameError('physics_name not known.')
                    
        # add model
        self.physics_model.add_entry(z, physics_model(
            ph_name,
            self.log_m_c,
            initial_guess=self.quantity_options['model_p0'],
            bounds=self.quantity_options['model_bounds'],
            eddington_erdf_params=eddington_erdf_params))

        # calculate initial erdf (which is reused for fixed ERDF model)
        if ((ph_name == 'eddington') and self.calibrate
                and (self._initial_erdf is None)):
            self._initial_erdf = self.physics_model.at_z(z).\
                calculate_log_erdf(
                self._initial_eddington_space)
        return


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
        self.list = list(self.dict.values())
        self.data = None
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
