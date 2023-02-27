#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 18:15:34 2022

@author: chris
"""
from abc import ABC, abstractmethod

from model.interface import run_model
from model.data.load import load_data_points
from model.analysis.reference_parametrization import get_reference_function_sample, \
                                                     calculate_best_fit_reference_parameter,\
                                                     calculate_reference_parameter_from_data    
from model.analysis.calculations import calculate_qhmr, calculate_best_fit_ndf,\
                                        calculate_q1_q2_relation,\
                                        calculate_ndf_percentiles,\
                                        calculate_conditional_ERDF_distribution,\
                                        calculate_quantity_density,\
                                        calculate_stellar_mass_density
from model.plotting.convience_functions import  plot_group_data, plot_best_fit_ndf,\
                                                plot_data_with_confidence_intervals,\
                                                add_redshift_text, add_legend,\
                                                add_separated_legend,\
                                                turn_off_axes,\
                                                get_distribution_limits,\
                                                plot_data_points, CurvedText,\
                                                plot_linear_relationship,\
                                                plot_q1_q2_additional,\
                                                plot_feedback_regimes,\
                                                blend_color,\
                                                plot_ndf_data_simple,\
                                                plot_JWST_data
from model.helper import make_list, pick_from_list, sort_by_density, t_to_z,\
                         make_array, log_L_uv_to_log_sfr
from model.scatter import Joint_distribution, calculate_q1_q2_conditional_pdf

from pathlib import Path
import warnings
import numpy as np
from scipy.integrate import trapezoid
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

mpl.style.use('model/plotting/settings.rc')

################ MAIN FUNCTIONS AND CLASSES ###################################

def get_list_of_plots():
    '''
    Return overview of all possible plots by listing subclasses of Plot.
    '''
    return(Plot.__subclasses__())

class Plot(ABC):
    def __init__(self, ModelResult, columns='double', color='C3', **kwargs):
        ''' General Configurations '''
        mpl.style.use('model/plotting/settings.rc') # return to original 
                                                    # plot style parameter
        
        self.adjust_style_parameter(columns) # adjust plot style parameter
        self.color = color

        self.make_plot(ModelResult, columns, **kwargs)
        self.default_filename = None
        
        mpl.style.use('model/plotting/settings.rc') # return to original 
                                                    # plot style parameter
    
    def make_plot(self, ModelResult, columns, **kwargs):
        ModelResult = make_list(ModelResult)

        self.quantity_name = ModelResult[0].quantity_name
        self.prior_name = ModelResult[0].prior_name

        if len(ModelResult) == 1:
            ModelResult = ModelResult[0]

        fig, axes = self._plot(ModelResult, **kwargs)

        self.fig = fig
        self.axes = axes
        return

    @abstractmethod
    def _plot(self, ModelResult):
        return
    
    def adjust_style_parameter(self, columns):
        '''
        Depending on how plot is supposed to be displayed in paper format 
        ('single' or 'double' column), adjust some parameter to make parts of 
        plot readable.
        '''
        if columns == 'single':
            mpl.rcParams['axes.labelsize']  *= 1.4
            mpl.rcParams['xtick.labelsize'] *= 1.8
            mpl.rcParams['ytick.labelsize'] *= 1.65
            mpl.rcParams['font.size']       *= 2.2
            self.plot_limits = {'top': 0.965, 'bottom': 0.175,
                                'left': 0.135, 'right': 0.995,
                                'hspace': 0.0, 'wspace': 0.0}
        elif columns == 'double':
            mpl.rcParams['lines.markersize']   /= 1.2
            self.plot_limits = {'top': 0.984, 'bottom': 0.130,
                                'left': 0.078, 'right': 0.994,
                                'hspace': 0.0, 'wspace': 0.0}
        else:
            raise ValueError('columns must be either \'single\' or \'double\'')
            
        return   
    
    def to_print_mode(self, labelsize_scaling = 1, leftbottom_ticks=True,
                      labels_off = False):
        '''
        Make plot ready for printing/post-processing. Turns off axis labels, 
        changes tick label size. leftbottom_ticks can be used to turn off ticks
        on top and right side.
        '''
        if self.fig is None:
            raise AttributeError('call make_plot first to create figure.')
        
        new_labelsize = labelsize_scaling*mpl.rcParams['xtick.labelsize'] 
        
        self.quantity_name = 'presentations' # save in seperate folder
         
        if labels_off:
            self.fig.supxlabel('')
            self.fig.supylabel('')
            axes = make_list(self.axes)
            for ax in axes:
                # Turn off axis label
                ax.set_xlabel('')
                ax.set_ylabel('')
                
                # Change tick label size
                ax.tick_params(axis='both', labelsize=new_labelsize)
                
                # Turn off ticks on top/right
                if leftbottom_ticks:
                    ax.tick_params(top = False, right = False, labeltop = False,
                                   labelright = False)
        return

    def save(self, file_format='pdf', file_name=None, path=None, 
             print_mode=False):
        '''
        Save figure to file. Can change file format (e.g. \'pdf\' or \'png\'),
        file name (but default is set) and save path (but default is set).
        Can enable print_mode, which turns off axis labels, makes background
        transparent.
        '''
        if self.fig is None:
            raise AttributeError('call make_plot first to create figure.')
        if file_name is None:
            file_name = self.default_filename
        if path is None:
            path = 'plots/' + self.quantity_name + '/'
        
        transparent = False
        if print_mode:
            self.to_print_mode()
            file_name   = file_name + '_print'
            path        = 'plots/' + self.quantity_name + '/print/'
            transparent = True

        Path(path).mkdir(parents=True, exist_ok=True)
        file = file_name + '.' + file_format

        self.fig.savefig(path + file, format=file_format, 
                         transparent = transparent)
        return(self)

################ PLOTS ########################################################
class Plot_best_fit_ndf(Plot):
    def __init__(self, ModelResults, **kwargs):
        '''
        Plot modelled number density functions and data for comparison. Input
        can be a single model object or a list of objects.
        You can turn off the plotting of the data points using 'datapoints' 
        argument. Choose redshift using 'redshift' argument.
        '''
        super().__init__(ModelResults, **kwargs)
        self.default_filename = (self.quantity_name + '_ndf_best_fit')

    def _plot(self, ModelResults, datapoints=True, redshift=0):
        # make list if input is scalar
        ModelResults = make_list(ModelResults)

        if ModelResults[0].parameter.is_None():
            raise AttributeError(
                'best fit parameter have not been calculated.')

        # general plotting configuration
        fig, ax = plt.subplots(1, 1)
        fig.subplots_adjust(**self.plot_limits)
        
        # quantity specific settings
        xlabel, ylabel = ModelResults[0].quantity_options['ndf_xlabel'],\
                         ModelResults[0].quantity_options['ndf_ylabel']

        # add axes labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # add minor ticks and set number of ticks
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        #ax.minorticks_on()

        # plot group data points
        if datapoints:
            plot_group_data(ax, ModelResults[0], redshift=redshift)

        # plot modelled number density functions
        for model in ModelResults:
            plot_best_fit_ndf(ax, model, redshift=redshift)

        # add axes limits
        quantity_range = ModelResults[0].quantity_options['quantity_range']
        ax.set_ylim(ModelResults[0].quantity_options['ndf_y_axis_limit'])
        ax.set_xlim([quantity_range[0], quantity_range[-1]])
        return(fig, ax)


class Plot_best_fit_ndfs_all_z(Plot):
    def __init__(self, ModelResults, **kwargs):
        '''
        Plot modelled number density functions and data across all 
        redshifts. Input can be a single model object or a list of objects.
        You can turn off the plotting of the data points using 'datapoints' 
        argument.
        '''
        super().__init__(ModelResults, **kwargs)
        self.default_filename = (self.quantity_name + '_ndfs_best_fit')

    def _plot(self, ModelResults, datapoints=True):
        # make list if input is scalar
        ModelResults = make_list(ModelResults)

        if ModelResults[0].parameter.is_None():
            raise AttributeError(
                'best fit parameter have not been calculated.')

        # general plotting configuration
        subplot_grid = ModelResults[0].quantity_options['subplot_grid']
        fig, axes = plt.subplots(*subplot_grid, sharey=True)
        axes = axes.flatten()
        fig.subplots_adjust(**self.plot_limits)

        # quantity specific settings
        xlabel, ylabel, ncol  = ModelResults[0].quantity_options['ndf_xlabel'],\
                                ModelResults[0].quantity_options['ndf_ylabel'],\
                                ModelResults[0].quantity_options['legend_columns']
        legend_loc            = ModelResults[0].quantity_options['ndf_legend_pos']

        # add axes labels
        fig.supxlabel(xlabel)
        fig.supylabel(ylabel, x=0.01)
        fig.align_ylabels(axes)

        # add minor ticks and set number of ticks
        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.minorticks_on()

        # plot group data points
        if datapoints:
            plot_group_data(axes, ModelResults[0])

        # plot modelled number density functions
        for model in ModelResults:
            plot_best_fit_ndf(axes, model)

        # add redshift as text to subplots
        add_redshift_text(axes, ModelResults[0].redshift)

        # add axes limits
        quantity_range = ModelResults[0].quantity_options['quantity_range']
        for ax in axes:
            ax.set_ylim(ModelResults[0].quantity_options['ndf_y_axis_limit'])
            ax.set_xlim([quantity_range[0],
                         quantity_range[-1]])

        # add legend
        if (len(ModelResults)==1) and \
           (ModelResults[0].physics_name == 'changing'):
            separation_point = 2
        else:
            separation_point = len(ModelResults)
        add_separated_legend(axes, separation_point=separation_point,
                             ncol=ncol, loc = legend_loc)
        
        # turn off unused axes
        turn_off_axes(axes)
        return(fig, axes)


class Plot_marginal_pdfs(Plot):
    def __init__(self, ModelResults, **kwargs):
        '''
        Plot marginal probability distributions for model parameter. Input
        can be a single model object or a list of objects.
        '''
        super().__init__(ModelResults, **kwargs)
        self.default_filename = self.quantity_name + '_pdf'

    def _plot(self, ModelResults):
        # make list if input is scalar
        ModelResults = make_list(ModelResults)

        if ModelResults[0].distribution.is_None():
            raise AttributeError('distributions have not been calculated.')

        # general plotting configuration
        fig, axes = plt.subplots(ModelResults[0].quantity_options['model_param_num'],
                                 len(ModelResults[0].redshift),
                                 sharex='row', sharey='row')
        fig.subplots_adjust(**self.plot_limits)
        fig.subplots_adjust(top=0.92, left=0.110, hspace=0.2)

        # set plot limits
        limits = get_distribution_limits(ModelResults)
        for i, limit in enumerate(limits):
            axes[i, 0]. set_xlim(*limit)

        # add axes labels
        fig.supxlabel('Parameter Value')
        fig.supylabel('(Marginal) Probability Density', x=0.01)
        param_labels =  ModelResults[0].quantity_options['param_y_labels']
        for i, label in enumerate(param_labels):
            axes[i, 0].set_ylabel(label, multialignment='center')
        fig.align_ylabels(axes)

        # plot marginal probability distributions
        for model in ModelResults:
            for z in model.redshift:
                for i, param_dist in enumerate(model.distribution.at_z(z).T):
                    axes[i, z].hist(param_dist, bins=100, density=True,
                                    color=pick_from_list(model.color, z),
                                    label=pick_from_list(model.label, z),
                                    alpha=0.3)
                    
        # turn of y ticks, add minor ticks and set number for x ticks
        for ax in axes.flatten():
            ax.get_yaxis().set_ticks([])
            ax.xaxis.set_major_locator(MaxNLocator(2))
            ax.minorticks_on()

        # add redshift as text
        add_redshift_text(axes[0], ModelResults[0].redshift)

        # add legend
        add_legend(axes, (0, -1), sort=True,
                   loc='upper right', bbox_to_anchor=(1.0, 1.4),
                   ncol=3, fancybox=True)

        # turn off unused axes
        turn_off_axes(axes)
        return(fig, axes)


class Plot_parameter_sample(Plot):
    def __init__(self, ModelResult, **kwargs):
        '''
        Plot a sample of the model parameter distribution at every redshift.
        Choose number of samples using num argument. In principle you can
        add a second axis showing the lookback time using the time_axis
        argument, but the code is rather experimental.
        Can choose to only plot some columns (and marginalise over the rest),
        use marginalise argument that has to be of the form (marg_z, columns),
        where marg_z is the redshift at which marginalisation is supposed to 
        start and columns is a list of the columns that are meant to be kept. 
        If rasterized=False, create vector graphic, otherwise fixed resolution
        graphic.
        '''
        super().__init__(ModelResult, **kwargs)
        self.default_filename = self.quantity_name + '_parameter'

    def _plot(self, ModelResult, num=int(1e+4), time_axis=False,
              marginalise=None, rasterized=True):
        
        # adjust fig size
        figsize    = mpl.rcParams['figure.figsize']
        figsize[1] *= 1.5
        
        # general plotting configuration
        param_num = ModelResult.distribution.at_z(ModelResult.
                                                  redshift[0]).shape[1]  
        fig, axes = plt.subplots(param_num, 1, sharex=True, figsize=figsize)
        fig.subplots_adjust(**self.plot_limits)

        if ModelResult.distribution.is_None():
            raise AttributeError('distributions have not been calculated.')

        # add axes labels
        param_labels =  ModelResult.quantity_options['param_y_labels']
        if param_num > ModelResult.quantity_options['model_param_num']:
            m_c_label = (r'$\log M_\mathrm{c}^'
                        + ModelResult.quantity_options[
                                        'quantity_subscript'] + r'$')
            param_labels = param_labels + [m_c_label]
        for i, label in enumerate(param_labels):
            axes[i].set_ylabel(label, multialignment='center')
        axes[-1].set_xlabel(r'Redshift $z$')
        fig.align_ylabels(axes)
        
        # parameter for marginalisation
        marg_z  = marginalise[0] if marginalise else ModelResult.redshift[-1]+1
        columns = marginalise[1] if marginalise else None
        
        # create custom color map
        washed_out_color = blend_color(self.color, 0.2)
        cm = LinearSegmentedColormap.from_list("Custom", 
                                               [washed_out_color,self.color],
                                               N=num)
        
        # check if log_m_c is kept free at any redshift
        free_m_c = False in [ModelResult.physics_model.at_z(z).
                             fixed_m_c_flag for z in ModelResult.redshift]
        
        # draw and plot parameter samples
        for z in ModelResult.redshift:
            # draw parameter sample
            parameter_sample = ModelResult.draw_parameter_sample(z, num)
            # estimate density using Gaussian KDE and use to assign color,
            # marginalise first if wanted
            cols = columns if z>=marg_z else None
            parameter_sample, color = sort_by_density(parameter_sample, 
                                                      keep=cols)

            # create xvalues and add scatter for easier visibility
            x = np.repeat(z, num) + np.random.normal(loc=0,
                                                     scale=0.03, size=num)
            # plot parameter sample 
            # if some parameter are marginalised, only plot remaining ones
            if z>=marg_z:
                for i,j in enumerate(columns):
                    if free_m_c:
                        # put log_m_c on last axis
                        k = j-1 if j>0 else parameter_sample.shape[-1]-1
                    else:
                        k = j
                    axes[k].scatter(x, parameter_sample[:, i],
                                    c=color, s=0.1, cmap=cm,
                                    rasterized=rasterized)
            else:
                # otherwise plot all, but make destinction where depending
                # of if theres a log_m_c column or not
                for i,j in enumerate(ModelResult.physics_model.at_z(z).
                                     parameter_used):
                    if free_m_c:
                        # put log_m_c on last axis
                        k = j-1 if j>0 else parameter_sample.shape[-1]-1
                        axes[k].scatter(x, parameter_sample[:, i],
                                        c=color, s=0.1, cmap=cm,
                                        rasterized=rasterized)
                    else:
                        axes[i].scatter(x, parameter_sample[:, i],
                                        c=color, s=0.1, cmap=cm,
                                        rasterized=rasterized)

        # add tick for every redshift
        axes[-1].set_xticks(ModelResult.redshift)
        axes[-1].set_xticklabels(ModelResult.redshift)

        # set ticks and set ylim
        for i, ax in enumerate(axes.flatten()):
            ax.tick_params(axis='y', which='minor')
            ax.yaxis.set_major_locator(MaxNLocator(2))
            
            if free_m_c and ax==(axes.flatten()[-1]):
                continue
            
            curr_lim  = ax.get_ylim()
            upper_lim = (1.15*curr_lim[1] if curr_lim[1]>0 
                         else 0.85*curr_lim[1])
            ax.set_ylim([curr_lim[0], upper_lim])
            
            if (ModelResult.quantity_name in ['mstar','Muv']):
                if (i==2 and free_m_c) or (i==3 and (not free_m_c)):
                    ax.set_ylim([-0.03, 1.1])

        # second axis for time, very experimental
        if len(ModelResult.redshift)==11 and time_axis:
            warnings.warn('additional time axis deprecated, use with caution')
            ax_z = axes[0].twiny()
            axes = np.append(axes, ax_z)
            ax_z.set_xlim(axes[0].get_xlim())
    
            t_ticks = np.arange(1, 14, 1).astype(int)
            t_ticks = np.append(t_ticks, 13.3)
            t_label = np.append(
                t_ticks[:-1].astype(int).astype(str), t_ticks[-1].astype(str))
            t_loc = t_to_z(t_ticks)
    
            ax_z.set_xticks(t_loc)
            ax_z.set_xticklabels(t_label)
            ax_z.set_xlabel('Lookback time [Gyr]')
            
        return(fig, axes)


class Plot_qhmr(Plot):
    def __init__(self, ModelResult, **kwargs):
        '''
        Plot relation between observable quantity and halo mass. Choose if
        relation should be shown directly or as a ratio using ratio argument.
        Select redshift to be shown using redshift and number of sigma
        equivalents using sigma. If median=True, plot median of distribution.
        If only_data is true, show only in range where observational data is 
        available.
        Adapt range over which quantities is calculated using m_halo_range
        and change y limits using y_lims. Add legend using legend argument.
        '''
        super().__init__(ModelResult, **kwargs)
        self.default_filename = self.quantity_name + '_qhmr'

    def _plot(self, ModelResult, ratio=True, redshift=None, sigma=1, 
              median=True, only_data=False, 
              m_halo_range=np.linspace(10.7, 14.23, 1000), y_lims=None,
              legend=True):
        if not np.isscalar(sigma):
            raise ValueError('Sigma must be scalar.')

        # calculate qhmr
        if redshift is None:
            redshift = ModelResult.redshift
        qhmr = {}
        for z in redshift:
            qhmr[z] = calculate_qhmr(ModelResult, z, sigma=sigma, ratio=ratio,
                                     log_m_halos=m_halo_range)
        
        # general plotting configuration
        fig, ax = plt.subplots(1, 1)
        fig.subplots_adjust(**self.plot_limits)

        # add axes labels
        ax.set_xlabel(r'log $M_\mathrm{h}$ [$M_\odot$]')
        y_label = 'log '+ModelResult.quantity_options['quantity_name_tex']
        if ratio:
            y_label = y_label + r'/$M_\mathrm{h}$'
        ax.set_ylabel(y_label, x=0.01)

        # create custom color map
        #washed_out_color = blend_color(self.color, 0.5)
        cm = LinearSegmentedColormap.from_list("Custom", 
                                               ['lightgrey', self.color],
                                               N=len(redshift))
        for i, z in enumerate(redshift):
            # calculate masks where values are outside of observations
            if only_data:
                qhmr_qs = np.copy(list(qhmr[z].values())[0][:,1])
                if  ratio: #'add halo masses back'
                    qhmr_qs += list(qhmr[z].values())[0][:,0]
                obs_qs  = ModelResult.log_ndfs.at_z(z)[:,0]
                mask_beginning = qhmr_qs<np.amin(obs_qs)
                mask_trail     = np.amax(obs_qs)<qhmr_qs
                data_mask      = np.logical_not(mask_beginning+mask_trail)
                data_masks     = [data_mask, mask_beginning, mask_trail] 
            else:
                data_masks     = None
            # plot confidence intervals
            plot_data_with_confidence_intervals(ax, qhmr[z], 
                                                cm(i/len(redshift)), 
                                                median=median,
                                                data_masks = data_masks,
                                                only_data=only_data,
                                                label = r'$z \sim$ ' + str(z))
            
        # add axis limits
        ax.set_xlim((m_halo_range[0], m_halo_range[-1]))
        if y_lims:
            ax.set_ylim(y_lims)

        # add legend and minor ticks
        ax.minorticks_on()
        # add legend
        if legend:
            add_legend(ax, 0, fontsize=32, loc='lower right')
        return(fig, ax)


class Plot_ndf_intervals(Plot):
    def __init__(self, ModelResult, **kwargs):
        '''
        Plot sample of number density functions by randomly drawing from parameter
        distribution and calculating ndfs, then calculating percentiles of
        these samples. You can adjust the number sigma equivalents drawn using
        sigma argument and number of samples drawn for calculation using 
        num argument.
        You can turn off the plotting of the data points using datapoints 
        argument, the best fit line using best_fit and the feedback regimes
        using feedback_regimes. You can add the best fit lines of other models
        by passing the (list of) model objects to additional_models. 
        '''
        super().__init__(ModelResult,  **kwargs)
        self.default_filename = self.quantity_name + '_ndf_intervals'

    def _plot(self, ModelResult, sigma=1, num=10000, best_fit=False, 
              datapoints=True, feedback_regimes=True, 
              additional_color ='#ccc979', additional_models = None):
        if ModelResult.distribution.is_None():
            raise AttributeError('distributions have not been calculated.')

        # get ndf sample
        ndfs = {}
        for z in ModelResult.redshift:
            ndfs[z] = calculate_ndf_percentiles(ModelResult, z, sigma=sigma,
                                                num=num)

        # change aspect ratio
        mpl.rcParams['figure.figsize'][1]  *= 1.35
        # general plotting configuration
        subplot_grid = ModelResult.quantity_options['subplot_grid']
        fig, axes = plt.subplots(*subplot_grid, sharey=True)
        axes = axes.flatten()
        fig.subplots_adjust(**self.plot_limits)

        # define plot parameter
        color     = self.color

        # quantity specific settings
        xlabel, ylabel, ncol  = ModelResult.quantity_options['ndf_xlabel'],\
                                ModelResult.quantity_options['ndf_ylabel'],\
                                ModelResult.quantity_options['legend_columns']
        legend_loc            = ModelResult.quantity_options['ndf_legend_pos']

        # add axes labels
        fig.supxlabel(xlabel)
        fig.supylabel(ylabel, x=0.001)
        fig.align_ylabels(axes)

        # add minor ticks and set number of ticks
        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.minorticks_on()

        # plot number density functions
        for z in ModelResult.redshift:
            plot_data_with_confidence_intervals(axes[z], ndfs[z], color,
                                                median=False, alpha=1)
        
        # plot best fit
        if best_fit:
            plot_best_fit_ndf(axes, ModelResult)

        # plot group data points
        if datapoints:
            plot_group_data(axes, ModelResult)
        
        # add feedback regimes
        if feedback_regimes:
            plot_feedback_regimes(axes, ModelResult, color=additional_color,
                                  alpha=0.25)
            
        # add best fits for other models as comparison
        if additional_models:
            models = make_list(additional_models)
            for model in models:
                plot_best_fit_ndf(axes, model, linewidth=4, alpha=0.6)

        # add redshift as text to subplots
        add_redshift_text(axes, ModelResult.redshift)

        # add axes limits
        quantity_range = ModelResult.quantity_options['quantity_range']
        for ax in axes:
            ax.set_ylim(ModelResult.quantity_options['ndf_y_axis_limit'])
            ax.set_xlim([quantity_range[0],
                         quantity_range[-1]])

        # add legend
        add_legend(axes, -1, ncol=ncol, loc=legend_loc)

        # turn off unused axes
        turn_off_axes(axes)
        return(fig, axes)
    
class Plot_ndf_predictions(Plot):
    def __init__(self, ModelResult, **kwargs):
        '''

        '''
        super().__init__(ModelResult,  **kwargs)
        self.default_filename = self.quantity_name + '_ndf_predictions'

    def _plot(self, ModelResult, upper_redshift=None, sigma=1, num=2000,
              quantity_range = None, additional_color='#7dcc79', 
              datapoints=True, y_lim=None):
        if ModelResult.distribution.is_None():
            raise AttributeError('distributions have not been calculated.')
        
        if upper_redshift is None:
            upper_redshift = ModelResult.quantity_options['extrapolation_end']
        # make redshift space
        redshift = range(ModelResult.quantity_options['extrapolation_z'][-1],
                         upper_redshift+1)
        # get ndf sample
        ndfs_extrapolate = {} # extrapolated parameter distribution
        ndfs_fixed       = {} # fixed parameter distribution, evolving HMF
        for z in redshift:
            if z>redshift[0]:
                ndfs_fixed[z]       = calculate_ndf_percentiles(ModelResult, 
                                                redshift[0], sigma=sigma,
                                                num=num, hmf_z=z,
                                                quantity_range=quantity_range,
                                                extrapolate_if_possible=True)
            ndfs_extrapolate[z] = calculate_ndf_percentiles(ModelResult, 
                                                z, sigma=sigma, num=num,
                                                quantity_range=quantity_range,
                                                extrapolate_if_possible=True)

        # general plotting configuration
        subplot_grid = list(ModelResult.quantity_options['subplot_grid'])
        subplot_grid[0] = np.ceil(len(redshift)/subplot_grid[1]).astype(int)
        fig, axes = plt.subplots(*subplot_grid, sharey=True)
        axes = axes.flatten()
        fig.subplots_adjust(**self.plot_limits)

        # quantity specific settings
        xlabel, ylabel, ncol  = ModelResult.quantity_options['ndf_xlabel'],\
                                ModelResult.quantity_options['ndf_ylabel'],\
                                ModelResult.quantity_options['legend_columns']
        legend_loc            = ModelResult.quantity_options['ndf_legend_pos']

        # add axes labels
        fig.supxlabel(xlabel)
        fig.supylabel(ylabel, x=0.001)
        fig.align_ylabels(axes)

        # add minor ticks and set number of ticks
        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.minorticks_on()

        # plot number density functions
        for i, z in enumerate(redshift):
            if z==redshift[0]:
                color = self.color
            else:
                color = additional_color
            if z>redshift[0]:
                plot_data_with_confidence_intervals(axes[i], ndfs_fixed[z],
                                                    'lightgrey',
                                                    median=False, alpha=0.7)         
            plot_data_with_confidence_intervals(axes[i], 
                                                ndfs_extrapolate[z],
                                                color,
                                                median=False, alpha=0.7)

        # plot group data points
        if datapoints:
            plot_ndf_data_simple(axes, ModelResult, redshift)

        # plot JWST data
        plot_JWST_data(axes, ModelResult, redshift[0])

        # add redshift as text to subplots
        add_redshift_text(axes, redshift, ind=range(len(redshift)))

        # add axes limits
        if quantity_range is None:
            quantity_range = ModelResult.quantity_options['quantity_range']
        for ax in axes:
            ax.set_xlim([quantity_range[0], quantity_range[-1]])
        if y_lim is None:
            y_lim = ModelResult.quantity_options['ndf_y_axis_limit']
        ax.set_ylim(y_lim)

        # add legend
        add_legend(axes, -1, ncol=ncol, loc=legend_loc)

        # turn off unused axes
        turn_off_axes(axes)
        return(fig, axes)


class Plot_reference_function_sample(Plot):
    def __init__(self, ModelResult, **kwargs):
        '''
        Plot sample of reference functions fitted to number density functions
        by randomly drawing from parameter distribution, calculating ndfs and
        then fitting reference functions.
        You can turn off the plotting of the data points using 'datapoints' 
        argument.
        '''
        super().__init__(ModelResult, **kwargs)
        self.default_filename = self.quantity_name + '_reference_sample'

    def _plot(self, ModelResult, datapoints=True):
        if ModelResult.distribution.is_None():
            raise AttributeError('distributions have not been calculated.')

        # get reference sample
        reference_functions = get_reference_function_sample(ModelResult,
                                                            ModelResult.redshift)

        # general plotting configuration
        subplot_grid = ModelResult.quantity_options['subplot_grid']
        fig, axes = plt.subplots(*subplot_grid, sharey='row', sharex=True)
        axes = axes.flatten()
        fig.subplots_adjust(**self.plot_limits)

        # define plot parameter
        plot_parameter = {'linewidth':0.2,
                          'alpha':0.2,
                          'color':'grey'}

        # quantity specific settings
        xlabel, ylabel, ncol  = ModelResult.quantity_options['ndf_xlabel'],\
                                ModelResult.quantity_options['ndf_ylabel'],\
                                ModelResult.quantity_options['legend_columns']
        legend_loc            = ModelResult.quantity_options['ndf_legend_pos']

        # add axes labels
        fig.supxlabel(xlabel)
        fig.supylabel(ylabel, x=0.01)
        fig.align_ylabels(axes)

        # add minor ticks and set number of ticks
        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.minorticks_on()

        # plot number density functions
        for z in ModelResult.redshift:
            for functions in reference_functions[z]:
                axes[z].plot(functions[:, 0], functions[:, 1], 
                             **plot_parameter)

        # plot group data points
        if datapoints:
            plot_group_data(axes, ModelResult)

        # add redshift as text to subplots
        add_redshift_text(axes, ModelResult.redshift)

        # add axes limits
        quantity_range = ModelResult.quantity_options['quantity_range']
        for ax in axes:
            ax.set_ylim(ModelResult.quantity_options['ndf_y_axis_limit'])
            ax.set_xlim([quantity_range[0],
                         quantity_range[-1]])

        # add legend
        add_separated_legend(axes, separation_point=0, ncol=ncol,
                             loc = legend_loc)

        # turn off unused axes
        turn_off_axes(axes)
        return(fig, axes)

class Plot_reference_comparison(Plot):
    def __init__(self, ModelResults, **kwargs):
        '''
        Plot sample of reference functions fitted to number density functions
        by randomly drawing from parameter distribution, calculating ndfs and
        then fitting reference functions.
        You can turn off the plotting of the data points using 'datapoints' 
        argument.
        '''
        super().__init__(ModelResults, **kwargs)
        self.default_filename = self.quantity_name + '_reference_comparison'
        
    def _plot(self, ModelResults, datapoints=True):
        # make list if input is scalar
        ModelResults = make_list(ModelResults)

        if ModelResults[0].parameter.is_None():
            raise AttributeError(
                'best fit parameter have not been calculated.')
        if ModelResults[0].distribution.is_None():
            raise AttributeError(
                'distributions have not been calculated.')
            
        # calculate reference parameter for data 
        reference_data_params = calculate_reference_parameter_from_data(ModelResults[0],
                                                                 ModelResults[0].redshift)
        # calculate reference parameter for models
        reference_models = []
        for Model in ModelResults:
            reference_models.append(calculate_best_fit_reference_parameter(
                                    Model, Model.redshift))
            
        # calculate reference samples (only for single ModelResult)
        if len(ModelResults) == 1:
            ndf_sample = {}
            for z in ModelResults[0].redshift:
                ndf_sample[z] = ModelResults[0].get_ndf_sample(z)
                  
        # general plotting configuration
        subplot_grid = ModelResults[0].quantity_options['subplot_grid']
        fig, axes = plt.subplots(*subplot_grid, sharey=True)
        axes = axes.flatten()
        fig.subplots_adjust(**self.plot_limits)
        
        # further plotting parameter:
        plot_parameter_model     = {'linewidth': 3,
                                    'alpha':1,
                                    'color':'grey'}
        plot_parameter_reference = {'linewidth':1.75,
                                    'alpha':1,
                                     'color':'C2'}
        plot_parameter_ndf_sample = {'linewidth':0.2,
                                     'alpha':0.4,
                                     'color':'grey'}
        linestyle = ['-','--',':']

        # quantity specific settings
        xlabel, ylabel, ncol  = ModelResults[0].quantity_options['ndf_xlabel'],\
                                ModelResults[0].quantity_options['ndf_ylabel'],\
                                ModelResults[0].quantity_options['legend_columns']
        legend_loc            = ModelResults[0].quantity_options['ndf_legend_pos']
        
        # def quantity range and reference function
        reference_function = ModelResults[0].quantity_options['reference_function']
        quantity_range     = ModelResults[0].quantity_options['quantity_range']
        
        # plot data group points
        if datapoints:
            plot_group_data(axes, ModelResults[0]) 
        
        # plot model ndfs and reference fits to ndfs
        for i, Model in enumerate(ModelResults):  
            ndfs             = calculate_best_fit_ndf(Model,
                                                      Model.redshift)
            reference_params = calculate_best_fit_reference_parameter(
                                   Model, Model.redshift)
            for z in Model.redshift:
                axes[z].plot(ndfs[z][:,0], ndfs[z][:,1],
                             label = 'Model (' + Model.prior_name + ' prior)',
                             linestyle = linestyle[i+1],
                             **plot_parameter_model)
                
                # add sample of reference functions
                if len(ModelResults) == 1:
                    for ndf in ndf_sample[z]:
                        axes[z].plot(ndf[:, 0], ndf[:, 1], 
                                     linestyle = linestyle[i+1], 
                                     **plot_parameter_ndf_sample)
                
                label = Model.quantity_options['reference_function_name'] +\
                        ' Function (' + Model.prior_name + ' prior)'
                axes[z].plot(quantity_range, 
                             reference_function(quantity_range, *reference_params[z]),
                             label = label,
                             linestyle = linestyle[i+1],
                             **plot_parameter_reference)
        
        # plot reference fits to data
        for z in ModelResults[0].redshift:
                axes[z].plot(quantity_range, 
                             reference_function(quantity_range,
                                                *reference_data_params[z]),
                             label = Model.quantity_options['reference_function_name'] + 
                                     ' Function (data)',
                             linestyle = linestyle[0],
                             **plot_parameter_reference) 

        # add axes labels
        fig.supxlabel(xlabel)
        fig.supylabel(ylabel, x=0.01)
        fig.align_ylabels(axes)

        # add minor ticks and set number of ticks
        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.minorticks_on()

        # add redshift as text to subplots
        add_redshift_text(axes, ModelResults[0].redshift)

        # add axes limits
        for ax in axes:
            ax.set_ylim(ModelResults[0].quantity_options['ndf_y_axis_limit'])
            ax.set_xlim([quantity_range[0],
                         quantity_range[-1]])

        # add legend
        separation_point = 2*len(ModelResults) + 1
        add_separated_legend(axes, separation_point=separation_point,
                             ncol=ncol, loc=legend_loc)
        
        # turn off unused axes
        turn_off_axes(axes)
        return(fig, axes)
    
class Plot_q1_q2_relation(Plot):
    def __init__(self, ModelResult1, ModelResult2, columns='double',
                 color = 'C3', additional_color='C3', **kwargs):
        '''
        Plot relation between two observable quantities according to Model.
        Works by using ModelResult1 to calculate halo mass distribution and
        then ModelResult2 to calculate quantity distribution for these 
        halo masses. Choose if relation should be shown directly or as a ratio 
        using ratio argument.
        You can plot additional relations with manipulated number densities 
        functions (see model for definition of fudge factor) by rerunning the 
        model with the adjusted values. The scaled_ndf parameter needs to 
        be of the form (model, scale factor), where the scale factor can be 
        scalar of an array. Different colors for the fudge factors can 
        be included using the scaled_ndf_color parameter.
        You can choose the redshift using z, the number of sigma equivalents
        shown using sigma, the datapoints using datapoints and the
        q1 range using quantity_range. Also can adjust number of samples drawn
        for calculation using num argument. Can fix y limits using y_lims.
        You can also add lines for linear relationships manually, just pass log
        of slopes via log_slopes argument. Can take array of values. You can 
        also add labels for these lines via log_slope_labels argument.
        masks argument controls if plot should highlight which parts of the
        model is constrained by data. legend=True adds legend.
        main color can be set with color argument, sometimes additional_color
        is used for other parts of the plot.
        '''
        self.adjust_style_parameter(columns) # adjust plot style parameter
        self.color     = color
        self.additional_color = additional_color
        
        self.make_plot(ModelResult1, ModelResult2, **kwargs)
        self.default_filename = (self.quantity_name + '_relation')
        
        mpl.style.use('model/plotting/settings.rc') # return to original 
                                                    # plot style parameter

    def make_plot(self, ModelResult1, ModelResult2, **kwargs):
        # adapted for two model results
        self.quantity_name = (ModelResult1.quantity_name 
                              + '_' + ModelResult2.quantity_name)
        self.prior_name = ModelResult1.prior_name
        fig, axes = self._plot(ModelResult1, ModelResult2, **kwargs)
        self.fig = fig
        self.axes = axes
        return
    
    def _plot(self, ModelResult1, ModelResult2, z=0, ratio = False, sigma=1, 
              num = 100, datapoints=False, scaled_ndf=None, 
              scaled_ndf_color = None, quantity_range=None, log_slopes=None, 
              log_slope_labels=None, masks=False, y_lims=None, legend=False,
              linewidth=5):       
        # sort sigma in reverse order
        sigma = np.sort(make_array(sigma))[::-1]
        
        # if no quantity_range is given, use default
        if quantity_range is None:
            log_q1 = ModelResult1.quantity_options['quantity_range']
            log_q1 = np.linspace(log_q1[0], log_q1[-1], 100)
        else:
            log_q1 = quantity_range
        
        if scaled_ndf_color:
            scaled_ndf_color = make_list(scaled_ndf_color)
            if len(scaled_ndf_color) != len(make_list(scaled_ndf[1])):
                raise ValueError("Length of scaled_ndf_alpha values must "
                                 "match length of scaled_ndf values.")
        elif (scaled_ndf is not None) and (scaled_ndf_color is None):
            scaled_ndf_color = ["grey"] * len(make_list(scaled_ndf[1]))
        
        # calculate relation and sigmas for all given sigmas, save as array
        q1_q2_relation = calculate_q1_q2_relation(ModelResult1,
                                                  ModelResult2,
                                                  z, log_q1, sigma=sigma,
                                                  num=num, ratio=ratio)

        # calculate additional relations with ndf fudge factor
        if scaled_ndf:
            if scaled_ndf[0] not in [ModelResult1, ModelResult2]:
                raise ValueError('Model for alternative ndf must be same as '
                                 'one of the quantity models')
            
            ndf_fudge_factors = make_list(scaled_ndf[1])
            alt_relations = {}
            for fudge_factor in ndf_fudge_factors:
                # calculate model with fudge factor
                AltModel = run_model(scaled_ndf[0].quantity_name, 
                                     scaled_ndf[0].physics_name,
                                     fitting_method='mcmc',
                                     redshift=z, num_walker=10,
                                     min_chain_length=0,
                                     ndf_fudge_factor=fudge_factor)

                # calculate q1_q2 relation for alternative model
                if AltModel.quantity_name == ModelResult1.quantity_name:
                    alt_relations[fudge_factor] = calculate_q1_q2_relation(
                                                                AltModel,
                                                                ModelResult2,
                                                                z, log_q1)[1]
                elif AltModel.quantity_name == ModelResult2.quantity_name:
                    alt_relations[fudge_factor] = calculate_q1_q2_relation(
                                                                ModelResult1,
                                                                AltModel,
                                                                z, log_q1)[1]
                else: 
                    raise ValueError('AltModel quantity name does not match '
                                     'either of the input ModelResults.')
                    
        ## create mask where model is not constrained by data
        # get minimum observed quantities
        q1_min_obs = np.amin(np.concatenate(ModelResult1.\
                                            log_ndfs.data)[:,0])  
        q2_min_obs = np.amin(np.concatenate(ModelResult2.\
                                            log_ndfs.data)[:,0]) 
        # select points where we have data, use whatever quantity is
        # more constraining
        idx_1      = q1_q2_relation[sigma[0]][:,0]>q1_min_obs
        idx_2      = q1_q2_relation[sigma[0]][:,1]>q2_min_obs
        data_mask  = (idx_1*idx_2)

        # split data mask into beginning and trailing true blocks
        # so that you can plot these seperately
        first_true                      = np.argmax(data_mask)
        mask_beginning                  = np.copy(np.logical_not(data_mask))
        mask_trail                      = np.copy(mask_beginning)
        mask_beginning[(first_true+1):] = False
        mask_trail[:(first_true+1)]     = False
        
        # turn masks on or off
        if masks:
            data_masks = [data_mask, mask_beginning, mask_trail]
        else:
            data_masks = None

        # general plotting configuration
        fig, ax = plt.subplots(1, 1)
        fig.subplots_adjust(**self.plot_limits)
        
        # plot values and confidence interval
        plot_data_with_confidence_intervals(ax, q1_q2_relation, self.color,
                                            data_masks=data_masks,
                                            linewidth=linewidth)
             
        # plot additional relations with fudge factor
        if scaled_ndf:
            for i, fudge_factor in enumerate(ndf_fudge_factors):
                ax.plot(alt_relations[fudge_factor][:,0],
                        alt_relations[fudge_factor][:,1],
                        color = scaled_ndf_color[i],
                        linewidth=linewidth)
                              
                # # add (curved) text
                text = ( f'{fudge_factor}x ' 
                        + AltModel.quantity_options['ndf_name'])
                CurvedText(x = alt_relations[fudge_factor][1:,0],
                           y = alt_relations[fudge_factor][1:,1],
                           text=text,
                           va = 'bottom',
                           axes = ax,
                           fontsize=0.65*mpl.rcParams['font.size'])  
                

        # add axis limits
        ax.set_xlim((log_q1[1], log_q1[-2]))
        if y_lims:
            ax.set_ylim(y_lims)

        # add axis labels
        #ax.set_xlabel(ModelResult1.quantity_options['ndf_xlabel'])
        #ax.set_ylabel(ModelResult2.quantity_options['ndf_xlabel'])
        
        # add axes labels
        ax.set_xlabel(ModelResult1.quantity_options['ndf_xlabel'])
        if ModelResult2.quantity_name=='Muv' and ModelResult2.sfr:
            print('label ugly implemented, also sfr or psi?')
            y_label = 'log SFR'
        else:
            y_label = 'log '+ ModelResult2.quantity_options['quantity_name_tex']
        if ratio:
            y_label = (y_label + '/' 
                       + ModelResult1.quantity_options['quantity_name_tex'])
        ax.set_ylabel(y_label, x=0.01)
        
        ## add stuff to plot
        # add measured datapoints
        if datapoints:
            plot_data_points(ax, ModelResult1, ModelResult2, z=z)
        # add lines for linear relation
        if log_slopes is not None:
            log_slopes = make_list(log_slopes)
            plot_linear_relationship(ax, log_q1, log_slopes, log_slope_labels)  
        # add other quantity-related things to plot
        plot_q1_q2_additional(ax, ModelResult1, ModelResult2, z, log_q1,
                              sigma=sigma, color=self.additional_color)
        
        # add legend
        if legend:
            add_legend(ax, 0, fontsize=32, loc='upper left')
        return(fig, ax)

class Plot_q1_q2_relation_evolution(Plot_q1_q2_relation):
    def __init__(self, ModelResult1, ModelResult2, columns='double',
                 color = 'C3', additional_color='C3', **kwargs):
        '''
        Similar to Plot_q1_q2_relation for multiple redshifts. Does not include
        the many minor addons the other function has.      
        redshift must be a list, you can choose the number of sigma equivalents
        shown using sigma and the q1 range using quantity_range. Also can 
        adjust number of samples drawn
        for calculation using num argument. Can fix y limits using y_lims.
        legend=True adds legend. Main color can be set with color argument.
        '''
        super().__init__(ModelResult1, ModelResult2, **kwargs)
        self.default_filename = (self.quantity_name + '_relation_evo')
    
    def _plot(self, ModelResult1, ModelResult2, redshift=[0], sigma=1, 
              num = 500, quantity_range=None, y_lims=None, legend=False, 
              linewidth=5):       
        # sort sigma in reverse order
        sigma = np.sort(make_array(sigma))[::-1]
        
        # if no quantity_range is given, use default
        if quantity_range is None:
            log_q1 = ModelResult1.quantity_options['quantity_range']
            log_q1 = np.linspace(log_q1[0], log_q1[-1], 1000)
        else:
            log_q1 = quantity_range
        
        # calculate relation and sigmas for all given sigmas, save as array
        
        q1_q2_relation = {}
        for z in redshift:
            q1_q2_relation[z] = calculate_q1_q2_relation(ModelResult1,
                                                          ModelResult2,
                                                          z, log_q1, 
                                                          sigma=sigma,
                                                          num=num)

        # general plotting configuration
        fig, ax = plt.subplots(1, 1)
        fig.subplots_adjust(**self.plot_limits)
        
        # plot values and confidence interval
        for z in redshift:
            ax.plot(q1_q2_relation[z][1][:,0], q1_q2_relation[z][1][:,1],
                    label=str(z))
            
            # plot_data_with_confidence_intervals(ax, q1_q2_relation[z],
            #                                     self.color,
            #                                     linewidth=linewidth,
            #                                     label = r'$z \sim$ ' + str(z))
        # add axis limits
        ax.set_xlim((log_q1[0], log_q1[-1]))
        if y_lims:
            ax.set_ylim(y_lims)
        # add legend
        if legend:
            add_legend(ax, 0, fontsize=32, loc='upper left')
        return(fig, ax)

class Plot_conditional_ERDF(Plot):
    def __init__(self, ModelResult, **kwargs):
        '''
        Plot sample of reference functions fitted to number density functions
        by randomly drawing from parameter distribution, calculating ndfs and
        then fitting reference functions.
        You can turn off the plotting of the data points using 'datapoints' 
        argument, and adjust the linewidth using 'linewidth'.
        '''
        super().__init__(ModelResult, **kwargs)
        self.default_filename = self.quantity_name + '_conditional_ERDF'
        
    def _plot(self, ModelResult, z=0, parameter=None, linewidth=5):
        
        if ModelResult.physics_name not in ['eddington','eddington_free_ERDF',
                                            'eddington_changing']:
            return NotImplementedError('Only implemented for bolometric '
                                       'luminosity - Eddington models.')

        if ModelResult.parameter.is_None():
            raise AttributeError(
                'best fit parameter have not been calculated.')
            

        # calculate qlf contributions
        if parameter is None:
            parameter = ModelResult.parameter.at_z(z)
        edd_space = np.linspace(-10,9,1000)
        lum       = [35, 40, 45, 50]
        qlf_cont  = ModelResult.calculate_conditional_ERDF(lum, z, parameter,
                                                           edd_space)
                  
        # general plotting configuration
        fig, ax = plt.subplots(1, 1, sharex=True)
        fig.subplots_adjust(**self.plot_limits)
        
        # line styles
        linewidth = linewidth
        linestyle = ['-','--',':', '-.']

        # add axes labels
        ax.set_xlabel(r'log $\lambda$')
        ax.set_ylabel(r'$\xi (\lambda \vert L_\mathrm{bol})$', labelpad=40)

        for i,l in enumerate(lum):
            # plot median
            ax.plot(qlf_cont[l][:,0], qlf_cont[l][:,1], 
                    color='grey', linestyle=linestyle[i],
                    linewidth=linewidth,
                    label=r'$\log L_\mathrm{bol}$ = ' + str(l) + 
                          r' [erg s$^{-1}$]')
            
        # add limits
        ax.set_xlim([edd_space[1],edd_space[-1]])
        ax.set_ylim([0,ax.get_ylim()[1]])
        
        # add legend and minor ticks
        add_legend(ax, 0, fontsize=32)
        ax.minorticks_on()
        return(fig, ax)
    
class Plot_black_hole_mass_distribution(Plot):
    def __init__(self, ModelResult, **kwargs):
        '''
        Plot expected distribution of black hole masses for a given luminosity
        from ERDF. Can take multiple luminosities, but if only one is given it
        also calculates the limits of the model and an adjusted distribution
        that takes the limits into account.
        You can adjust redshift, luminosity, number of samples, plotted
        sigma equivalents and base eddington space.
        You can turn off the plotting of the data points using 'datapoints' 
        argument (so far only data at z=0 from Baron2019 paper) and the legend
        using 'legend' argument.
        '''
        super().__init__(ModelResult, **kwargs)
        self.default_filename = self.quantity_name + '_bh_mass_distribution'
        
    def _plot(self, ModelResult, z=0, lum=45.2, 
              edd_space=np.linspace(-6, 32.13,10000), num=5000,
              sigma=1, datapoints=True, legend=False, linewidth=5):
        
        if ModelResult.physics_name not in ['eddington','eddington_free_ERDF',
                                            'eddington_changing']:
            return NotImplementedError('Only implemented for bolometric '
                                       'luminosity - Eddington models.')

        if ModelResult.parameter.is_None():
            raise AttributeError(
                'best fit parameter have not been calculated.')
        if not np.isscalar(lum):
            raise NotImplementedError('Not yet implemented for multiple '
                                      'luminosities. lum must be scalar.')     
        # sort sigma in reverse order
        sigma = np.sort(make_array(sigma))[::-1]

        # calculate black hole mass distribution(s)
        m_bh_dist = calculate_conditional_ERDF_distribution(
                                        ModelResult, lum, z, 
                                        eddington_space = edd_space,
                                        num=num, sigma=sigma,
                                        black_hole_mass_distribution=True)
        
        # load black hole mass and luminosity data
        if datapoints:
            m_bh_data = load_data_points('mbh_Lbol')
            m_bh_data = m_bh_data[~np.isnan(m_bh_data[:,0])] # remove NaNs
        
        ## calculate upper and lower probable bound of model
        dist = np.copy(m_bh_dist[sigma[0]])
        # lower mass limit, Eddingtion ratio = 1
        eddington_limit = lum-ModelResult.log_eddington_constant   
        # upper mass limit, Eddingtion ratio = 0.01
        Jet_mode_limit  = lum-ModelResult.log_eddington_constant+2
        lower_ind = np.argmin(dist[:,0]<eddington_limit)
        upper_ind = np.argmin(dist[:,0]<Jet_mode_limit)
        
        ## cut distribution to are between limits and normalise to 1
        parameter = ModelResult.parameter.at_z(z)
        dist_map  = ModelResult.calculate_conditional_ERDF(
                                    lum, z, parameter, edd_space, 
                                    black_hole_mass_distribution=True)
        cut_dist = dist_map[lum][lower_ind:upper_ind]
        norm = trapezoid(cut_dist[:,1], cut_dist[:,0])
        cut_dist[:,1] = cut_dist[:,1]/norm
                  
        # general plotting configuration
        fig, ax = plt.subplots(1, 1, sharex=True)
        fig.subplots_adjust(**self.plot_limits)

        # add axes labels
        ax.set_xlabel(r'log $M_\bullet$ [$M_\odot$]')
        ax.set_ylabel(r'$P (M_\bullet \vert L_\mathrm{bol})$', x=0.01)
        
        # plot predicted black hole distribution
        plot_data_with_confidence_intervals(ax, m_bh_dist, self.color, 
                                            median=True,
                                            linewidth=linewidth)
        
            
        # add upper and lower limit and re-normalised distribution
        ax.plot(cut_dist[:,0], cut_dist[:,1], 
                color='grey', linestyle='--',
                label= r'Adjusted Distribution',
                linewidth=linewidth)
        ax.axvline(eddington_limit, color='grey', linewidth=linewidth+1)
        ax.axvline(Jet_mode_limit, color='grey', linewidth=linewidth+1)
        # add text
        ax.text(1.005*eddington_limit, 0.5, '$\lambda = 1$', 
                rotation=90, va='bottom', ha='left')
        ax.text(1.005*Jet_mode_limit, 0.5, '$\lambda = 0.01$', 
                rotation=90, va='bottom', ha='left')
        
        # set limits
        ax.set_xlim([0.9*eddington_limit,1.09*Jet_mode_limit])
        
        # add histogram of data points
        if datapoints:
            ax.hist(m_bh_data[:,0], bins=15, density=True,
                    label = 'Observed Distribution\n(Baron2019, Type 1 AGN)',
                    color='lightgrey', alpha=0.8,zorder=0)
        
        # add legend
        if legend:
            add_legend(ax, 0)
        return(fig, ax)
    
class Plot_quantity_density_evolution(Plot):
    def __init__(self, ModelResults, **kwargs):
        '''
        Plot evolution of integrated ndfs by drawing samples and integrating
        the resulting ndfs. The range of redshift can be chosen using redshift
        argument. The number of samples drawn can be adjusted using num_samples
        and the number of points calculated for integrating the ndf can be
        controlled using num_integral_points. You can also manually choose 
        points where ndf should be evaluated using log_q_space. 
        Color for points that are extrapolated is chosen using 
        additional_color. If rasterized=False, create vector graphic, 
        otherwise fixed resolution graphic. If observational data points 
        are available, you can plot them using datapoints argument. Legend can
        be added using legend argument.
        '''
        super().__init__(ModelResults, **kwargs)
        self.default_filename = (self.quantity_name + '_density_evolution')

    def _plot(self, ModelResult, redshift=None, num_samples=int(1e+4), 
              num_integral_points=100, log_q_space=None,
              additional_color='#7dcc79', rasterized=True, datapoints=False,
              legend=False):
        
        if redshift is None:
            redshift = np.arange(ModelResult.redshift[0],
                                 ModelResult.quantity_options
                                 ['extrapolation_end']+1)
        # make list if input is scalar
        redshift = make_list(redshift)

        # calculate quantity densities
        densities = calculate_quantity_density(ModelResult, redshift, 
                                    sigma=1, return_samples=True,
                                    log_q_space=log_q_space,
                                    num_samples=num_samples, 
                                    num_integral_points=num_integral_points)

        # for UVLF, convert to SFR
        if ModelResult.quantity_name == 'Muv':
            for z in redshift:
                densities[z] = log_L_uv_to_log_sfr(densities[z])

        # general plotting configuration
        fig, ax = plt.subplots(1, 1)
        fig.subplots_adjust(**self.plot_limits)
        
        # quantity specific settings
        xlabel = 'Redshift'
        ylabel = ModelResult.quantity_options['density_ylabel']
        # add axes labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel, labelpad=20)
        
        # create custom color maps
        washed_out_color_obs = blend_color(self.color, 0.2)
        cm_obs = LinearSegmentedColormap.from_list("Custom", 
                                        [washed_out_color_obs, self.color],
                                        N=num_samples)
        washed_out_color_pred = blend_color(additional_color, 0.2)
        cm_pred = LinearSegmentedColormap.from_list("Custom", 
                                               [washed_out_color_pred, 
                                                additional_color],
                                               N=num_samples)
        
        # plot quantity densities
        for z in redshift:
            sample          = densities[z]
            sample          = sample[np.isfinite(sample)]
            sample, color   = sort_by_density(sample)
            
            # create xvalues and add scatter for easier visibility
            x = (np.repeat(z, len(sample)) 
                  + np.random.normal(scale=0.06, size=len(sample)))
            
            cm = cm_obs if z in ModelResult.redshift else cm_pred
            ax.scatter(x, sample, c=color, s=0.1, cmap=cm, 
                       rasterized=rasterized)
            
        # add measured datapoints
        if datapoints:
            plot_data_points(ax, ModelResult)
            
        # add y ticks
        #ax.yaxis.set_major_locator(MaxNLocator(10))
        ax.yaxis.grid(True, which='minor')
        # set y lim by calculating percentiles of all available samples
        ax.set_ylim(ModelResult.quantity_options['density_y_axis_limit'])
        # add x tick for every redshift
        ax.set_xticks(redshift)
        xticklabels = list(redshift[::2])
        for i in range(1,len(xticklabels)+1):
            xticklabels.insert(2*i-1,'')
        ax.set_xticklabels(xticklabels[:len(redshift)])
        
        if legend:
            add_legend(ax, 0, fontsize=32, loc='lower left', ncol=2)
        return(fig, ax)
    
    
class Plot_stellar_mass_density_evolution(Plot_q1_q2_relation):
    def __init__(self, mstar, muv, **kwargs):
        '''
        Plot evolution of integrated stellar mass density calculated directly
        from GSMF and by integrating SFR. The two models must be mstar and
        muv. The number of samples drawn can be adjusted using num_samples
        and the number of points calculated for integrating the ndf can be
        controlled using num_integral_points. You can also manually choose 
        points where ndf should be evaluated using log_q_space.
        Color for points that are extrapolated is chosen using 
        additional_color. Legend can be toggled using legend argument. If 
        rasterized=False, create vector graphic, otherwise fixed resolution
        graphic.
        '''
        super().__init__(mstar, muv, **kwargs)
        self.default_filename = (self.quantity_name + '_SMD_evolution')

    def _plot(self, mstar, muv, num_samples=int(1e+4), num_integral_points=100,
              log_q_space=None, legend=True, rasterized=True):
        # calculate stellar mass densities
        smd_dict, inferred_smd_dict = calculate_stellar_mass_density(mstar,
                                       muv, sigma=1, return_samples=True,
                                       log_q_space=log_q_space,
                                       num_samples=num_samples, 
                                       num_integral_points=num_integral_points)
        
        redshift = list(smd_dict.keys())
        
        # general plotting configuration
        fig, ax = plt.subplots(1, 1)
        fig.subplots_adjust(**self.plot_limits)
        
        # quantity specific settings
        xlabel = 'Redshift'
        ylabel = mstar.quantity_options['density_ylabel']
        # add axes labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel, labelpad=20)
        
        # create custom color map
        washed_out_color = blend_color(self.color, 0.2)
        cm = LinearSegmentedColormap.from_list("Custom", 
                                               [washed_out_color,self.color],
                                               N=num_samples)
        cm_greys = LinearSegmentedColormap.from_list("Custom", 
                                                     ['lightgrey','grey'],
                                                     N=num_samples)
        # plot stellar mass densities
        for z in redshift:
            smd_sample          = smd_dict[z]
            smd_sample          = smd_sample[np.isfinite(smd_sample)]
            inferred_smd_sample = inferred_smd_dict[z]
            inferred_smd_sample = inferred_smd_sample[np.isfinite(
                                                          inferred_smd_sample)]
            
            smd_sample, smd_color = sort_by_density(smd_sample)
            inferred_smd_sample, inferred_smd_color = sort_by_density(
                                                        inferred_smd_sample)
            # create xvalues and add scatter for easier visibility
            x1 = (np.repeat(z, len(smd_sample)) 
                  + np.random.normal(scale=0.06, size=len(smd_sample)))
            x2 = (np.repeat(z, len(inferred_smd_sample)) 
                  + np.random.normal(scale=0.02, 
                                     size=len(inferred_smd_sample)))

            ax.scatter(x1[:-1], smd_sample[:-1], c=smd_color[:-1], s=0.1, 
                       cmap=cm_greys, rasterized=rasterized)
            ax.scatter(x2[:-1], inferred_smd_sample[:-1], 
                       c=inferred_smd_color[:-1], s=0.1, cmap=cm,
                       rasterized=rasterized)
            # last points drawn seperately just to get the colors for the 
            # legend right
            ax.scatter(x1[-1], smd_sample[-1], c='grey', s=0.1, 
                       label='obtained from GSMF', rasterized=rasterized)
            ax.scatter(x2[-1], inferred_smd_sample[-1], c=self.color, s=0.1,
                       label='integrated SFR', rasterized=rasterized)
    
        # add y ticks
        #ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.grid(True, which='minor')
        ax.tick_params(axis='x', which='minor', bottom=False)
        # set y lim by calculating percentiles of all available samples
        all_points = np.array([inferred_smd_dict[z] 
                               for z in redshift]).flatten()
        y_lims = np.percentile(all_points[np.isfinite(all_points)],
                               [0.05,99.99])
        ax.set_ylim(*y_lims)
        # add x tick for every redshift
        ax.set_xticks(redshift)
        ax.set_xticklabels(redshift)
        
        if legend:
            add_legend(ax, 0, fontsize=32, loc='lower left', markersize=100)
        return(fig, ax)
    
    
class Plot_scatter_ndf(Plot):
    def __init__(self, ModelResults, **kwargs):
        '''
        Plot modelled number density functions and data for comparison. Input
        can be a single model object or a list of objects.
        You can turn off the plotting of the data points using 'datapoints' 
        argument. Choose redshift using 'redshift' argument
        '''
        super().__init__(ModelResults, **kwargs)
        self.default_filename = (self.quantity_name + '_scatter_ndf')

    def _plot(self, ModelResult, scatter_name='lognormal', 
              scatter_parameter=[0.2, 0.5, 1], redshift=0,
              quantity_range=None, parameter=None, y_lim=(-6,0),
              legend=True):
        # make list if input is scalar
        scatter_parameter = make_list(scatter_parameter)
        
        if quantity_range is None:
            quantity_range = ModelResult.quantity_options['quantity_range']
        if parameter is None:
            parameter = ModelResult.parameter.at_z(redshift)
        
        # calculate ndfs
        ndfs = {}
        for s in scatter_parameter:
            ndfs[s] = ModelResult.calculate_log_abundance(
                            quantity_range, redshift, parameter,
                            scatter_name=scatter_name, scatter_parameter=s)
        ndf_no_scatter = ModelResult.calculate_log_abundance(quantity_range, 
                                                             redshift, 
                                                             parameter)
        
        # general plotting configuration
        fig, ax = plt.subplots(1, 1)
        fig.subplots_adjust(**self.plot_limits)
        
        # quantity specific settings
        xlabel  = ModelResult.quantity_options['ndf_xlabel']
        ylabel  = ModelResult.quantity_options['ndf_ylabel']

        # add axes labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # add minor ticks and set number of ticks
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        #ax.minorticks_on()
        
        # create custom color map
        washed_out_color = blend_color(self.color, 0.5)
        cm = LinearSegmentedColormap.from_list("Custom", 
                                               [self.color, washed_out_color],
                                               N=len(scatter_parameter))
        
        # plot ndf without scatter
        ax.plot(quantity_range, ndf_no_scatter, label='No scatter', 
                color='grey', zorder=0)
        # plot modelled ndfs with scatter
        for i,s in enumerate(scatter_parameter):
            ax.plot(quantity_range, ndfs[s],
                     label     = r'$\sigma =$ ' + str(s), 
                     linewidth = mpl.rcParams['lines.linewidth']/2,
                     color     = cm(i/len(scatter_parameter)))
        
        # axis limits
        plt.xlim(quantity_range[0], quantity_range[-1])
        plt.ylim(*y_lim)

        if legend:
            add_legend(ax, 0, fontsize=32, loc='upper right')
        return(fig, ax)


class Plot_q1_q2_distribution_with_scatter(Plot_q1_q2_relation):
    def __init__(self, ModelResult1, ModelResult2, columns='double',
                 color = 'C3', additional_color='C3', **kwargs):
        '''
        Plot an illustration of the effect of scatter on the q1-q2 relation.
        Show for scatter with lognormal distribution in both quantities and
        no skewness and with skewness for a given input value log_q2_value. 
        Redshift can be adjusted using redshift, range over which to calculate
        using log_q1 space. The scatter and skewness can be adjusted using 
        scatter_1, scatter_2 and skew.
        '''
        super().__init__(ModelResult1, ModelResult2, **kwargs)
        self.default_filename = (self.quantity_name + '_scatter_distribution')
    
    def _plot(self, ModelResult1, ModelResult2, log_q2_value,
              redshift=0, log_q1=np.linspace(8.51,10.41,500), scatter_1=0.05,
              scatter_2=0.25, skew=-40):

        if (ModelResult1.parameter.is_None() 
            or ModelResult2.parameter.is_None()):
            raise AttributeError('best fit parameter '
                                 'have not been calculated.')
            
        # choose parameter
        parameter_1 = ModelResult1.parameter.at_z(redshift)
        parameter_2 = ModelResult2.parameter.at_z(redshift)
        
        # create distributions
        JointDistribution1 = Joint_distribution(ModelResult1, 'lognormal', 
                                                scatter_1)
        JointDistribution2 = Joint_distribution(ModelResult2, 'lognormal',
                                                scatter_2)

        JointDistribution2_skew = Joint_distribution(ModelResult2,
                                'skewlognormal', scatter_parameter=scatter_2,
                                skew_parameter=skew)
        # calculate conditional probabilities
        log_q1_probabilities = []
        for Distribution2 in [JointDistribution2, JointDistribution2_skew]:
            log_q1_probabilities.append(calculate_q1_q2_conditional_pdf(
                                                    JointDistribution1, 
                                                    Distribution2, 
                                                    log_q1, 
                                                    log_q2_value, 
                                                    parameter_1, 
                                                    parameter_2, 
                                                    redshift))
        no_scatter_value = calculate_q1_q2_relation(ModelResult2, 
                                                    ModelResult1,
                                                    redshift, 
                                                    log_q2_value)[1][0,1]

                  
        # general plotting configuration
        fig, ax = plt.subplots(1, 1, sharex=True)
        fig.subplots_adjust(**self.plot_limits)

        # add axes labels
        y_label = (r'$P ($' 
                   + ModelResult1.quantity_options['quantity_name_tex']
                   + r'$\vert$'  
                   + ModelResult2.quantity_options['quantity_name_tex']
                   + r' $=$ ' + str(log_q2_value) + r'$)$')  
        ax.set_xlabel(ModelResult1.quantity_options['ndf_xlabel'])
        ax.set_ylabel(y_label, x=0.01)

        # plot distributions
        ax.plot(log_q1, log_q1_probabilities[0], color='lightgrey')
        ax.plot(log_q1, log_q1_probabilities[1], color=self.color)
        ax.axvline(no_scatter_value, color='black', alpha=0.8)
        
        # set plot limits
        ax.set_xlim(log_q1[0],log_q1[-1])      
        return(fig, ax)