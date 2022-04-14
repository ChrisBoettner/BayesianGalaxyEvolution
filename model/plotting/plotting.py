#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 18:15:34 2022

@author: chris
"""
from matplotlib import rc_file
rc_file('model/plotting/settings.rc')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

import numpy as np

from pathlib import Path

from model.helper import make_list, pick_from_list, sort_by_density, t_to_z

from model.analysis.calculations import calculate_qhmr                         
                         
from model.plotting.convience_functions import plot_group_data, plot_best_fit_ndf,\
                                               add_redshift_text, add_legend,\
                                               add_separated_legend,\
                                               turn_off_axes
class Plot(object):
    def __init__(self, ModelResult):
        ''' Empty Super Class '''
        self.plot_limits      = {'top'   :0.93 , 'bottom':0.113, 
                                 'left'  :0.075, 'right' :0.991,
                                 'hspace':0.0  , 'wspace':0.0   } 
        
        self.make_plot(ModelResult)
        self.default_name = None
        
    def make_plot(self, ModelResult):
        ModelResult  = make_list(ModelResult)
        
        self.quantity_name = ModelResult[0].quantity_name
        self.prior_name    = ModelResult[0].prior_name
        
        if len(ModelResult) == 1:
            ModelResult = ModelResult[0]
            
        fig, axes          = self._plot(ModelResult)
        
        self.fig           = fig
        self.axes          = axes
        return
    
    def _plot(self, ModelResult):
        return
        
    def save(self, file_format = 'pdf', file_name = None, path = None):
        '''
        Save figure to file. Can change file format (e.g. \'pdf\' or \'png\'),
        file name (but default is set) and save path (but default is set).
        '''
        if self.fig is None:
            raise AttributeError('call make_plot first to create figure.')
        if file_name is None:
            file_name = self.default_filename
        if path is None:
            path = 'plots/' + self.quantity_name + '/'
        
        Path(path).mkdir(parents=True, exist_ok=True)
        file = file_name + '.' + file_format
        
        self.fig.savefig(path + file)
        return(self)
    
        
class Plot_best_fit_ndfs(Plot):
    def __init__(self, ModelResults):
        '''
        Plot modelled number density functions and data for comparison. Input
        can be a single model object or a list of objects.
        '''
        super().__init__(ModelResults)
        self.default_filename = self.quantity_name + '_ndf'

    def _plot(self, ModelResults):
        # make list if input is scalar
        ModelResults = make_list(ModelResults)
        
        # general plotting configuration
        fig, axes = plt.subplots(4, 3, sharey = True)
        axes      = axes.flatten()
        fig.subplots_adjust(**self.plot_limits)
        
        # quantity specific settings
        if ModelResults[0].quantity_name == 'mstar':
            xlabel = r'log $M_*$ [$M_\odot$]'
            ylabel = r'log $\phi(M_*)$ [cMpc$^{-3}$ dex$^{-1}$]'
            ncol = 1
        elif ModelResults[0].quantity_name == 'Muv':
            xlabel = r'$\mathcal{M}_{UV}$'
            ylabel = r'log $\phi(\mathcal{M}_{UV})$ [cMpc$^{-3}$ mag$^{-1}$]'
            ncol = 3
        else:
            raise ValueError('quantity_name not known.')
        
        # add axes labels
        fig.supxlabel(xlabel)
        fig.supylabel(ylabel, x=0.01)
        fig.align_ylabels(axes)
        
        # add minor ticks and set number of ticks
        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.minorticks_on()
        
        # plot group data points
        plot_group_data(axes, ModelResults[0].groups)
        
        # plot modelled number density functions
        for model in ModelResults: 
            ndfs = plot_best_fit_ndf(axes, model)
            
        # add redshift as text to subplots
        add_redshift_text(axes, ModelResults[0].redshift)
        
        # add axes limits
        for ax in axes:
            ax.set_ylim([-6,3]) 
            ax.set_xlim([list(ndfs.values())[0][0,0],
                         list(ndfs.values())[0][-1,0]])
        
        # add legend
        add_separated_legend(axes, separation_point = len(ModelResults),
                             ncol = ncol)
        
        # turn off unused axes
        turn_off_axes(axes)
        return(fig, axes)       
    
class Plot_marginal_pdfs(Plot):
    def __init__(self, ModelResults):
        '''
        Plot marginal probability distributions for model parameter. Input
        can be a single model object or a list of objects.
        '''
        super().__init__(ModelResults)
        self.default_filename = self.quantity_name + '_pdf_' + self.prior_name
    
    def _plot(self, ModelResults):
        # make list if input is scalar
        ModelResults = make_list(ModelResults)
        
        # general plotting configuration
        fig, axes = plt.subplots(3, 11, sharex= 'row', sharey = 'row')
        fig.subplots_adjust(**self.plot_limits)
        fig.subplots_adjust(hspace=0.2)
        
        # quantity specific settings
        if ModelResults[0].quantity_name == 'mstar':
            pass
        elif ModelResults[0].quantity_name == 'Muv':
            ax0_xlim = (17.67,20.21)
            ax1_xlim = (0.001, 1.63)
            ax2_xlim = (0.001, 0.79)
        else:
            raise ValueError('quantity_name not known.')
        
        # add axes labels
        axes[0,0].set_ylabel('$\log A$')
        axes[1,0].set_ylabel(r'$\gamma$')
        axes[2,0].set_ylabel(r'$\delta$')
        fig.supxlabel('Parameter Value')
        fig.supylabel('(Marginal) Probability Density', x = 0.01)
        
        # plot marginal probability distributions
        for model in ModelResults:
            for z in model.redshift:
                for i, param_dist in enumerate(model.distribution.at_z(z).T):
                    axes[i,z].hist(param_dist, bins = 100, density = True,
                                   range  = (model.model.at_z(z).feedback_model.bounds).T[i],
                                   color  = pick_from_list(model.color, z),
                                   label  = model.label, alpha = 0.3)
        # set x limits
        axes[0,0].set_xlim(*ax0_xlim)
        axes[1,0].set_xlim(*ax1_xlim)
        axes[2,0].set_xlim(*ax2_xlim)
        
        # turn of y ticks                         
        for ax in axes.flatten():
           ax.get_yaxis().set_ticks([])
        
        # add redshift as text
        add_redshift_text(axes[0], ModelResults[0].redshift)
        
        # add legend
        add_legend(axes, (0,-1), sort = True, 
                   loc='upper right', bbox_to_anchor=(1.0, 1.2),
                   ncol=3, fancybox=True)
        
        # turn off unused axes
        turn_off_axes(axes)
        return(fig, axes)
        
class Plot_parameter_sample(Plot):
    def __init__(self, ModelResult = None):
        '''
        Plot a sample of the model parameter distribution at every redshift.
        '''
        super().__init__(ModelResult)
        self.default_filename = self.quantity_name + '_parameter'
        
    def _plot(self, ModelResult):
        # general plotting configuration
        fig, axes = plt.subplots(3,1, sharex = True)
        fig.subplots_adjust(**self.plot_limits)
        
        # quantity specific settings
        quantity_name = ModelResult.quantity_name
        if quantity_name == 'mstar':
            ax0_label = r'$\log A$' 
        elif quantity_name == 'Muv':
            ax0_label = r'$\log A$' + '\n' + r'[ergs s$^{-1}$ Hz$^{-1}$ $M_\odot^{-1}$]'
        else:
            raise ValueError('quantity_name not known.')
        
        # add axes labels
        axes[0].set_ylabel(ax0_label, multialignment='center')
        axes[1].set_ylabel(r'$\gamma$')
        axes[2].set_ylabel(r'$\delta$')
        axes[2].set_xlabel(r'Redshift $z$')
        fig.align_ylabels(axes)

        # draw and plot parameter samples
        for z in ModelResult.redshift:
            # draw parameter sample
            num              = int(1e+4)
            parameter_sample = ModelResult.draw_parameter_sample(z, num)
            
            # estimate density using Gaussian KDE and use to assign color
            parameter_sample, color = sort_by_density(parameter_sample)

            # create xvalues and add scatter for easier visibility 
            x = np.repeat(z, num) + np.random.normal(loc = 0, scale = 0.03, size=num) 
            # plot parameter sample
            for i in range(parameter_sample.shape[1]):
                axes[i].scatter(x[::10], parameter_sample[:,i][::10],
                                c = color[::10], s = 0.1, cmap = 'Oranges')
        
        # add tick for every redshift
        axes[-1].set_xticks(     ModelResult.redshift) 
        axes[-1].set_xticklabels(ModelResult.redshift)
        
        # second axis for redshift
        ax_z    = axes[0].twiny(); ax_z.set_xlim(axes[0].get_xlim())

        t_ticks    = np.arange(1,14,1).astype(int); t_ticks = np.append(t_ticks,13.3)
        t_label    = np.append(t_ticks[:-1].astype(int).astype(str),t_ticks[-1].astype(str))
        t_loc      = t_to_z(t_ticks)
        
        ax_z.set_xticks(t_loc)
        ax_z.set_xticklabels(t_label)
        ax_z.set_xlabel('Lookback time [Gyr]')
        return(fig, axes)
    
class Plot_qhmr(Plot):
    def __init__(self, ModelResult = None):
        '''
        Plot relation between observable quantity and halo mass.
        '''
        super().__init__(ModelResult)
        self.default_filename = self.quantity_name + '_qhmr'
        
    def _plot(self, ModelResult):
        if ModelResult.quantity_name != 'mstar':
            raise ValueError('Quantity-halo mass relation plot currently\
                              only supports mstar.')
        
        # calculate qhmr
        log_m_halos = np.linspace(8,14,100) # define halo mass range
        redshifts   = ModelResult.redshift[::2]
        qhmr        = calculate_qhmr(ModelResult, log_m_halos,
                                     redshifts = redshifts)
        
        # general plotting configuration
        fig, ax = plt.subplots(1,1, sharex = True) 
        fig.subplots_adjust(**self.plot_limits)
        
        # add axes labels
        fig.supxlabel('log $M_\mathrm{h}$ [$M_\odot$]')
        fig.supylabel('log($M_*/M_\mathrm{h}$)', x=0.01)
        
        # create custom color map
        cm = LinearSegmentedColormap.from_list("Custom", ['C2','C1'],
                                               N=len(ModelResult.redshift))
        
        for z in redshifts:                    
            # plot median
            ax.plot(log_m_halos, qhmr[z][:,0], color = cm(z),
                    label = '$z$ = ' + str(z))
            
            # plot 16th/84th percentiles
            ax.fill_between(log_m_halos, qhmr[z][:,1],qhmr[z][:,2],
                            alpha = 0.2, color = cm(z))
        
        # add legend and minor ticks
        add_legend(ax, 0)
        ax.minorticks_on()
        return(fig, ax)
    
class Plot_ndf_sample(Plot):
    def __init__(self, ModelResult = None):
        '''
        Plot sample of number density functions by randomly drawing from parameter
        distribution and calculating ndfs.
        '''
        super().__init__(ModelResult)
        self.default_filename = self.quantity_name + '_qhmr'
        
    def _plot(self, ModelResult):
        # get ndf sample
        ndfs = {}
        for z in ModelResult.redshift:
            ndfs[z] = ModelResult.get_ndf_sample(z)
        
        # general plotting configuration
        fig, axes = plt.subplots(4,3, sharey = 'row', sharex=True)
        axes      = axes.flatten()
        fig.subplots_adjust(**self.plot_limits)
        
        # define plot parameter 
        linewidth = 0.2
        alpha     = 0.2
        color     = 'grey'
        
        # quantity specific settings
        if ModelResult.quantity_name == 'mstar':
            xlabel = r'log $M_*$ [$M_\odot$]'
            ylabel = r'log $\phi(M_*)$ [cMpc$^{-3}$ dex$^{-1}$]'
            ncol = 1
        elif ModelResult.quantity_name == 'Muv':
            xlabel = r'$\mathcal{M}_{UV}$'
            ylabel = r'log $\phi(\mathcal{M}_{UV})$ [cMpc$^{-3}$ mag$^{-1}$]'
            ncol = 3
        
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
            for ndf in ndfs[z]:
                axes[z].plot(ndf[:,0], ndf[:,1], color = color,
                             linewidth = linewidth, alpha = alpha)
        
        # plot group data points
        plot_group_data(axes, ModelResult.groups)
        
        # add redshift as text to subplots
        add_redshift_text(axes, ModelResult.redshift)
        
        # add axes limits
        for ax in axes:
            ax.set_ylim([-6,3]) 
            ax.set_xlim([list(ndfs.values())[0][0][0,0],
                         list(ndfs.values())[0][0][-1,0]])
        
        # add legend
        add_separated_legend(axes, separation_point = 0, ncol = ncol)
        
        # turn off unused axes
        turn_off_axes(axes)
        return(fig, axes)
    
class Plot_schechter_sample(Plot):
    def __init__(self, ModelResult = None):
        '''
        Plot sample of Schechter functions fitted to number density functions 
        by randomly drawing from parameter distribution, calculating ndfs and 
        then fitting Schechter functions.
        '''
        super().__init__(ModelResult)
        self.default_filename = self.quantity_name + '_qhmr'
        
    def _plot(self, ModelResult):
        # get ndf sample
        ndfs = {}
        for z in ModelResult.redshift:
            ndfs[z] = ModelResult.get_ndf_sample(z)
        
        # general plotting configuration
        fig, axes = plt.subplots(4,3, sharey = 'row', sharex=True)
        axes      = axes.flatten()
        fig.subplots_adjust(**self.plot_limits)
        
        # define plot parameter 
        linewidth = 0.2
        alpha     = 0.2
        color     = 'grey'
        
        # quantity specific settings
        if ModelResult.quantity_name == 'mstar':
            xlabel = r'log $M_*$ [$M_\odot$]'
            ylabel = r'log $\phi(M_*)$ [cMpc$^{-3}$ dex$^{-1}$]'
            ncol = 1
        elif ModelResult.quantity_name == 'Muv':
            xlabel = r'$\mathcal{M}_{UV}$'
            ylabel = r'log $\phi(\mathcal{M}_{UV})$ [cMpc$^{-3}$ mag$^{-1}$]'
            ncol = 3
        
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
            for ndf in ndfs[z]:
                axes[z].plot(ndf[:,0], ndf[:,1], color = color,
                             linewidth = linewidth, alpha = alpha)
        
        # plot group data points
        plot_group_data(axes, ModelResult.groups)
        
        # add redshift as text to subplots
        add_redshift_text(axes, ModelResult.redshift)
        
        # add axes limits
        for ax in axes:
            ax.set_ylim([-6,3]) 
            ax.set_xlim([list(ndfs.values())[0][0][0,0],
                         list(ndfs.values())[0][0][-1,0]])
        
        # add legend
        add_separated_legend(axes, separation_point = 0, ncol = ncol)
        
        # turn off unused axes
        turn_off_axes(axes)
        return(fig, axes)