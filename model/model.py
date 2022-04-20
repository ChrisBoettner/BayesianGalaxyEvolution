#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 17:46:44 2022

@author: chris
"""
import warnings

import numpy as np

from model.helper import mag_to_lum, within_bounds, make_list, system_path

from model.calibration import mcmc_fitting, leastsq_fitting

from model.quantity_options import get_quantity_specifics
from model.feedback import feedback_model

from model.calibration.parameter import load_parameter

################ MAIN CLASSES #################################################


class ModelResult():
    '''
    Central model object.
    '''

    def __init__(self, redshifts, log_ndfs, log_hmf_functions,
                 quantity_name, feedback_name, prior_name,
                 fitting_method, saving_mode, name_addon=None,
                 groups=None, calibrate=True, **kwargs):
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
        feedback_name : str
            Name of feedback model. Must be 'none', 'stellar',
            'stellar_feedback' or 'changing'. 'changing' uses 'stellar_blackhole'
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
        **kwargs : dict
            Additional arguments that can be passed to the mcmc algorithm.

        Returns
        -------
        None.

        '''

        self.redshift = make_list(redshifts)
        self.log_ndfs = Redshift_dict(log_ndfs)
        self.log_hmfs = Redshift_dict(log_hmf_functions)
        self.groups = groups
        
        #pre-set critical mass for feedback
        self.log_m_c = 12.45

        self.quantity_name = quantity_name
        # load options related to the quantity 
        self.quantity_options = get_quantity_specifics(self.quantity_name)
        
        self.feedback_name = feedback_name
        self.prior_name = prior_name

        self.fitting_method = fitting_method
        self.saving_mode = saving_mode
        self.name_addon = name_addon
        
        # location where files will be saved or loaded from
        self.directory = None

        self.feedback_model = Redshift_dict({})
        self.filename = Redshift_dict({})
        self.parameter = Redshift_dict({})
        self.distribution = Redshift_dict({})
        if calibrate:
            self.fit_model(self.redshift, **kwargs)

        if saving_mode == 'loading':
            try:
                parameter = load_parameter(self, name_addon)
                parameter = {int(z): p for z, p in parameter.items()}
                self.parameter = Redshift_dict(parameter)
            except FileNotFoundError:
                warnings.warn('Couldn\'t load best fit parameter')

        # default plot parameter per feedback_model
        if self.feedback_name == 'none':
            self._plot_parameter('black', 'o', '-', 'No Feedback')
        elif self.feedback_name == 'stellar':
            self._plot_parameter('C1', 's', '--', 'Stellar Feedback')
        elif self.feedback_name == 'stellar_blackhole':
            self._plot_parameter(
                'C2', 'v', '-.', 'Stellar + Black Hole Feedback')
        elif self.feedback_name == 'changing':
            self._plot_parameter(
                ['C2'] * 5 + ['C1'] * 6, 'o', '-',
                ['Stellar + Black Hole Feedback'] * 5 + ['Stellar Feedback'] * 6)
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
        redshifts = make_list(redshifts)

        posterior_samp = None
        bounds = None
        for z in redshifts:
            print('z=' + str(z))
            self._z = z # temporary storage for current redshift

            # add saving paths and file name
            self.directory = system_path() + self.quantity_name + '/' \
                            + self.feedback_name + '/'
            filename = self.prior_name + '_z' + str(z)
            # if manual modification of saving path is wanted
            if self.name_addon:
                filename = filename + ''.join(self.name_addon)
            self.filename.add_entry(z, filename)
      
            # create feedback model
            if self.feedback_name in ['none', 'stellar', 'stellar_blackhole']:
                fb_name = self.feedback_name
            elif self.feedback_name == 'changing':  # standard changing feedback
                feedback_change_z = 4
                if z <= feedback_change_z:
                    fb_name = 'stellar_blackhole'
                elif z > feedback_change_z:
                    fb_name = 'stellar'
            else:
                raise ValueError('feedback_name not known.')

            self.feedback_model.add_entry(z, feedback_model(
                fb_name,
                self.log_m_c,
                initial_guess=self.quantity_options['model_p0'],
                bounds=self.quantity_options['model_bounds']))

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
                raise ValueError('Prior model not known.')

            # fit parameter/sample distribution
            if self.fitting_method == 'least_squares':
                parameter, posterior_samp = leastsq_fitting.lsq_fit(self)
            elif self.fitting_method == 'mcmc':
                parameter, posterior_samp = mcmc_fitting.mcmc_fit(
                    self, prior, saving_mode=self.saving_mode, **kwargs)
            else:
                raise ValueError('fitting_method not known.')
            
            self.parameter.add_entry(z, parameter)
            self.distribution.add_entry(z, posterior_samp)
        return

    def calculate_log_abundance(self, log_quantity, z, parameter):
        '''
        Calculate (log of) value (phi) of modelled number density function by 
        multiplying HMF function with feedback model derivative for a given
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
        if self.quantity_name == 'mstar':
            pass
        elif self.quantity_name == 'Muv':
            # convert magnitude to luminosity
            log_quantity = np.log10(mag_to_lum(log_quantity))
        else:
            raise ValueError('quantity_name not known.')

        # check that parameters are within bounds
        if not within_bounds(parameter, *self.feedback_model.at_z(z).bounds):
            return(1e+30)  # return inf (or huge value) if outside of bounds

        # calculate halo masses from stellar masses using model
        log_m_h = self.feedback_model.at_z(z).calculate_log_halo_mass(
            log_quantity, *parameter)
        # calculate modelled number density function
        log_hmf = self.log_hmfs.at_z(z)(log_m_h)
        log_fb_factor = np.log10(
            self.feedback_model.at_z(z).calculate_dlogquantity_dlogmh(
                log_m_h, *parameter))
        # calculate modelled phi value
        log_phi = log_hmf - log_fb_factor
        return(log_phi)

    def draw_parameter_sample(self, z, num=1):
        '''
        Get a sample from feedback parameter distribution at given redshift.

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
            raise AttributeError('distribution dictonary is empty. Probably' +\
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
        log_quantity_dist : list
            Calculated distribution.

        '''

        parameter_sample = self.draw_parameter_sample(z, num=num)

        log_quantity_dist = []
        for p in parameter_sample:
            log_quantity_dist.append(
                self.feedback_model.at_z(z).calculate_log_quantity(
                    log_halo_mass, *p))
        return(np.array(log_quantity_dist))

    def calculate_halo_mass_distribution(self, log_quantity, z, num=int(1e+5)):
        '''
        At a given redshift, calculate distribution of halo mass for a given
        observable quantity (mstar/Muv) by drawing parameter sample and
        calculating value for each one (number of draws adaptable.)

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
        log_halo_mass_dist : list
            Calculated distribution.

        '''
        parameter_sample = self.draw_parameter_sample(z, num=num)

        log_halo_mass_dist = []
        for p in parameter_sample:
            log_halo_mass_dist.append(
                self.feedback_model.at_z(z).calculate_log_halo_mass(
                    log_quantity, *p))
        return(np.array(log_halo_mass_dist))

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
        ndf_sample : array
            Calculated number density functions.

        '''
        if quantity_range is None:
            quantity_range = self.quantity_options['quantity_range']

        ndf = self.calculate_log_abundance(quantity_range, z, parameter)
        return([quantity_range, ndf])

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
        self.data = input_dict

    def add_entry(self, z, value):
        '''Add new entry at z to dictonary.'''
        self.data[z] = value
        return

    def at_z(self, z):
        ''' Retrieve data at z.'''
        if z not in list(self.data.keys()):
            raise ValueError('Redshift not in data.')
        else:
            return(self.data[z])

    def is_None(self):
        ''' Check if dictonary is empty. '''
        return(list(self.data.values())[0] is None)
