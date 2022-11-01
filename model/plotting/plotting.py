#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 18:15:34 2022

@author: chris
"""

from model.interface import run_model
from model.data.load import load_data_points
from model.analysis.reference_parametrization import get_reference_function_sample, \
                                                     calculate_best_fit_reference_parameter,\
                                                     calculate_reference_parameter_from_data    
from model.analysis.calculations import calculate_qhmr, calculate_best_fit_ndf,\
                                        calculate_q1_q2_relation,\
                                        calculate_ndf_percentiles,\
                                        calculate_conditional_ERDF_distribution
from model.plotting.convience_functions import  plot_group_data, plot_best_fit_ndf,\
                                                plot_data_with_confidence_intervals,\
                                                add_redshift_text, add_legend,\
                                                add_separated_legend,\
                                                turn_off_axes,\
                                                get_distribution_limits,\
                                                plot_data_points, CurvedText,\
                                                plot_linear_relationship,\
                                                plot_q1_q2_additional,\
                                                plot_feedback_regimes
                                                
from model.helper import make_list, pick_from_list, sort_by_density, t_to_z,\
                         make_array

from pathlib import Path
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

class Plot(object):
    def __init__(self, ModelResult, columns='double', color='C3', **kwargs):
        ''' General Configurations '''
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
            mpl.rcParams['ytick.labelsize'] *= 1.8
            mpl.rcParams['font.size']       *= 2.2
            self.plot_limits = {'top': 0.965, 'bottom': 0.175,
                                'left': 0.135, 'right': 0.995,
                                'hspace': 0.0, 'wspace': 0.0}
        elif columns == 'double':
            mpl.rcParams['lines.markersize'] /= 2
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
        argument. Choose redshift using 'redshift' argument
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
        '''
        super().__init__(ModelResult, **kwargs)
        self.default_filename = self.quantity_name + '_parameter'

    def _plot(self, ModelResult):
        # general plotting configuration
        param_num = ModelResult.quantity_options['model_param_num']
        fig, axes = plt.subplots(param_num,
                                 1, sharex=True)
        fig.subplots_adjust(**self.plot_limits)

        if ModelResult.distribution.is_None():
            raise AttributeError('distributions have not been calculated.')

        # add axes labels
        param_labels =  ModelResult.quantity_options['param_y_labels']
        for i, label in enumerate(param_labels):
            axes[i].set_ylabel(label, multialignment='center')
        axes[-1].set_xlabel(r'Redshift $z$')
        fig.align_ylabels(axes)

        # draw and plot parameter samples
        for z in ModelResult.redshift:
            # draw parameter sample
            num = int(1e+3)
            parameter_sample = ModelResult.draw_parameter_sample(z, num)

            # estimate density using Gaussian KDE and use to assign color
            parameter_sample, color = sort_by_density(parameter_sample)

            # create xvalues and add scatter for easier visibility
            x = np.repeat(z, num) + np.random.normal(loc=0,
                                                     scale=0.03, size=num)
            # plot parameter sample
            for i in range(parameter_sample.shape[1]):
                axes[i].scatter(x, parameter_sample[:, i],
                                c=color, s=0.1, cmap='Greys')

        # add tick for every redshift
        axes[-1].set_xticks(ModelResult.redshift)
        axes[-1].set_xticklabels(ModelResult.redshift)

        # set number for x ticks
        for ax in axes.flatten():
            ax.yaxis.set_major_locator(MaxNLocator(3))

        # second axis for redshift
        if len(ModelResult.redshift)==11: # only include plots with z=0-10
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
            
        # fit everything back into frame
        fig.tight_layout()
        fig.subplots_adjust(hspace = 0)
        
        return(fig, axes)


class Plot_qhmr(Plot):
    def __init__(self, ModelResult, **kwargs):
        '''
        Plot relation between observable quantity and halo mass.
        '''
        super().__init__(ModelResult, **kwargs)
        self.default_filename = self.quantity_name + '_qhmr'

    def _plot(self, ModelResult, sigma=1):
        if not np.isscalar(sigma):
            raise ValueError('Sigma must be scalar.')

        # calculate qhmr
        redshifts = ModelResult.redshift[::2]
        qhmr = {}
        for z in redshifts:
            qhmr[z] = calculate_qhmr(ModelResult, z, sigma=sigma)

        # general plotting configuration
        fig, ax = plt.subplots(1, 1)
        fig.subplots_adjust(**self.plot_limits)

        # add axes labels
        ax.set_xlabel('log $M_\\mathrm{h}$ [$M_\\odot$]')
        ax.set_ylabel('log '+ModelResult.quantity_options['quantity_name_tex'],
                      x=0.01)

        # create custom color map
        cm = LinearSegmentedColormap.from_list("Custom", ['C2', 'C1'],
                                               N=len(ModelResult.redshift))

        for z in redshifts:
            # plot median
            ax.plot(qhmr[z][:, 0], qhmr[z][:, 1], color=cm(z),
                    label='$z$ = ' + str(z))
            # plot 16th/84th percentiles
            ax.fill_between(qhmr[z][:, 0], qhmr[z][:, 2], qhmr[z][:, 3],
                            alpha=0.2, color=cm(z))
            
        # add axis limits
        ax.set_xlim((qhmr[z][0,0], qhmr[z][-1,0]))

        # add legend and minor ticks
        add_legend(ax, 0)
        ax.minorticks_on()
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

    def _plot(self, ModelResult, sigma=1, num=5000, best_fit=False, 
              datapoints=True, feedback_regimes=True, 
              additional_models = None):
        if ModelResult.distribution.is_None():
            raise AttributeError('distributions have not been calculated.')

        # get ndf sample
        ndfs = {}
        for z in ModelResult.redshift:
            ndfs[z] = calculate_ndf_percentiles(ModelResult, z, sigma=sigma,
                                                num=num)

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
        fig.supylabel(ylabel, x=0.01)
        fig.align_ylabels(axes)

        # add minor ticks and set number of ticks
        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(4))
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
            plot_feedback_regimes(axes, ModelResult, color=self.color)
            
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
        halo masses.
        You can plot additional relations with manipulated number densities 
        functions (see model for definition of fudge factor) by rerunning the 
        model with the adjusted values. The scaled_ndf parameter needs to 
        be of the form (model, scale factor), where the scale factor can be 
        scalar of an array.
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
    
    def _plot(self, ModelResult1, ModelResult2, z=0, sigma=1, num = 500,
              datapoints=False, scaled_ndf=None, quantity_range=None,
              log_slopes=None, log_slope_labels=None, masks=False, 
              y_lims=None, legend=False, linewidth=5):       
        # sort sigma in reverse order
        sigma = np.sort(make_array(sigma))[::-1]
        
        # if no quantity_range is given, use default
        if quantity_range is None:
            log_q1 = ModelResult1.quantity_options['quantity_range']
        else:
            log_q1 = quantity_range
        
        # calculate relation and sigmas for all given sigmas, save as array
        q1_q2_relation = calculate_q1_q2_relation(ModelResult1,
                                                  ModelResult2,
                                                  z, log_q1, sigma=sigma,
                                                  num=num)

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
            for fudge_factor in ndf_fudge_factors:
                ax.plot(alt_relations[fudge_factor][:,0],
                        alt_relations[fudge_factor][:,1],
                        color='grey',
                        linewidth=linewidth)
                              
                # # add (curved) text
                text = ( f'{fudge_factor}x ' 
                        + AltModel.quantity_options['ndf_name'])
                CurvedText(x = alt_relations[fudge_factor][:,0],
                           y = alt_relations[fudge_factor][:,1],
                           text=text,
                           va = 'bottom',
                           axes = ax,
                           fontsize=0.65*mpl.rcParams['font.size'])  
                

        # add axis limits
        ax.set_xlim((log_q1[0], log_q1[-1]))
        if y_lims:
            ax.set_ylim(y_lims)

        # add axis labels
        ax.set_xlabel(ModelResult1.quantity_options['ndf_xlabel'])
        ax.set_ylabel(ModelResult2.quantity_options['ndf_xlabel'])
        
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
        ax.set_ylabel(r'$\xi (\lambda|L_\mathrm{bol})$', labelpad=40)

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
              edd_space=np.linspace(-6, 32.13,10000), num=500,
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
        ax.set_ylabel(r'$P (M_\bullet|L_\mathrm{bol})$', x=0.01)
        
        # plot predicted black hole distribution
        plot_data_with_confidence_intervals(ax, m_bh_dist, self.color, median=True,
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
