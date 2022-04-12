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

from model.helper import make_list, pick_from_list, sort_by_density, t_to_z,\
                         calculate_percentiles
from model.plotting.convience_functions import plot_group_data, plot_best_fit_ndf,\
                                               add_redshift_text, add_legend,\
                                               add_separated_legend,\
                                               turn_off_axes
class Plot(object):
    def __init__(self, CalibrationResult):
        ''' Empty Super Class '''
        self.plot_limits      = {'top'   :0.93 , 'bottom':0.113, 
                                 'left'  :0.075, 'right' :0.991,
                                 'hspace':0.0  , 'wspace':0.0   } 
        
        self.make_plot(CalibrationResult)
        self.default_name = None
        
    def make_plot(self, CalibrationResult):
        CalibrationResult  = make_list(CalibrationResult)
        
        self.quantity_name = CalibrationResult[0].quantity_name
        self.prior_name    = CalibrationResult[0].prior_name
        
        if len(CalibrationResult) == 1:
            CalibrationResult = CalibrationResult[0]
            
        fig, axes          = self._plot(CalibrationResult)
        
        self.fig           = fig
        self.axes          = axes
        return
    
    def _plot(self, CalibrationResult):
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
    
        
class Plot_ndfs(Plot):
    def __init__(self, CalibrationResults):
        '''
        Plot modelled number density functions and data for comparison. Input
        can be a single model object or a list of objects.
        '''
        super().__init__(CalibrationResults)
        self.default_filename = self.quantity_name + '_ndf'

    def _plot(self, CalibrationResults):
        # make list if input is scalar
        CalibrationResults = make_list(CalibrationResults)
        
        # general plotting configuration
        fig, axes = plt.subplots(4, 3, sharey = True)
        axes      = axes.flatten()
        fig.subplots_adjust(**self.plot_limits)
        
        # quantity specific settings
        quantity_name = CalibrationResults[0].quantity_name
        if quantity_name == 'mstar':
            xlabel = r'log $M_*$ [$M_\odot$]'
            ylabel = r'log $\phi(M_*)$ [cMpc$^{-3}$ dex$^{-1}$]'
            ncol = 1
        if quantity_name == 'Muv':
            xlabel = r'$\mathcal{M}_{UV}$'
            ylabel = r'log $\phi(\mathcal{M}_{UV})$ [cMpc$^{-3}$ dex$^{-1}$]'
            ncol = 3
        
        # add axes labels
        fig.supxlabel(xlabel)
        fig.supylabel(ylabel, x=0.01)
        fig.align_ylabels(axes)
        
        # add
        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.minorticks_on()
        
        # plot group data points and modelled number density functions
        plot_group_data(axes, CalibrationResults[0].groups)
        
        # plot modelled number density functions
        for model in CalibrationResults: 
            quantity, _ = plot_best_fit_ndf(axes, model)
            
        # add redshift as text to subplots
        add_redshift_text(axes, CalibrationResults[0].redshift)
        
        # add axes limits
        for ax in axes:
            ax.set_ylim([-6,3]) 
            ax.set_xlim([quantity[0], quantity[-1]])
        
        # add legend
        add_separated_legend(axes, separation_point = len(CalibrationResults),
                             ncol = ncol)
        
        # turn off unused axes
        turn_off_axes(axes)
        return(fig, axes)   
    
class Plot_marginal_pdfs(Plot):
    def __init__(self, CalibrationResults):
        '''
        Plot marginal probability distributions for model parameter. Input
        can be a single model object or a list of objects.
        '''
        super().__init__(CalibrationResults)
        self.default_filename = self.quantity_name + '_pdf_' + self.prior_name
    
    def _plot(self, CalibrationResults):
        # make list if input is scalar
        CalibrationResults = make_list(CalibrationResults)
        
        # general plotting configuration
        fig, axes = plt.subplots(3, 11,
                                 sharex ='row', sharey = 'row')
        fig.subplots_adjust(**self.plot_limits)
        fig.subplots_adjust(hspace=0.2)
        
        
        # add axes labels
        axes[0,0].set_ylabel('$\log A$')
        axes[1,0].set_ylabel(r'$\gamma$')
        axes[2,0].set_ylabel(r'$\delta$')
        fig.supxlabel('Parameter Value')
        fig.supylabel('(Marginal) Probability Density', x = 0.01)
        
        # plot marginal probability distributions
        for model in CalibrationResults:
            for z in model.redshift:
                for i, param_dist in enumerate(model.distribution.at_z(z).T):
                    axes[i,z].hist(param_dist, bins = 100, density = True,
                                   range  = (model.model.at_z(z).feedback_model.bounds).T[i],
                                   color  = pick_from_list(model.color, z),
                                   label  = model.label, alpha = 0.3)
                                 
        # turn of y ticks                         
        for ax in axes.flatten():
           ax.get_yaxis().set_ticks([])
        
        # add redshift as text
        add_redshift_text(axes[0], CalibrationResults[0].redshift)
        
        # add legend
        add_legend(axes, (0,-1), sort = True, 
                   loc='upper right', bbox_to_anchor=(1.0, 1.2),
                   ncol=3, fancybox=True)
        
        # turn off unused axes
        turn_off_axes(axes)
        return(fig, axes)
        
class Plot_parameter_sample(Plot):
    def __init__(self, CalibrationResult = None):
        '''
        Plot a sample of the model parameter distribution at every redshift.
        '''
        super().__init__(CalibrationResult)
        self.default_filename = self.quantity_name + '_parameter'
        
    def _plot(self, CalibrationResult):
        # general plotting configuration
        fig, axes = plt.subplots(3,1, sharex = True)
        fig.subplots_adjust(**self.plot_limits)
        
        # quantity specific settings
        quantity_name = CalibrationResult.quantity_name
        if quantity_name == 'mstar':
            ax0label = r'$\log A$' 
        if quantity_name == 'Muv':
            ax0label = r'$\log A$' + '\n' + r'[ergs s$^{-1}$ Hz$^{-1}$ $M_\odot^{-1}$]'
        
        # add axes labels
        axes[0].set_ylabel(ax0label, multialignment='center')
        axes[1].set_ylabel(r'$\gamma$')
        axes[2].set_ylabel(r'$\delta$')
        axes[2].set_xlabel(r'Redshift $z$')
        fig.align_ylabels(axes)

        # draw and plot parameter samples
        for z in CalibrationResult.redshift:
            # draw parameter sample
            num              = int(1e+4)
            parameter_sample = CalibrationResult.draw_parameter_sample(z, num)
            
            # estimate density using Gaussian KDE and use to assign color
            parameter_sample, color = sort_by_density(parameter_sample)

            # create xvalues and add scatter for easier visibility 
            x = np.repeat(z, num) + np.random.normal(loc = 0, scale = 0.03, size=num) 
            # plot parameter sample
            for i in range(parameter_sample.shape[1]):
                axes[i].scatter(x[::10], parameter_sample[:,i][::10],
                                c = color[::10], s = 0.1, cmap = 'Oranges')
        
        # add tick for every redshift
        axes[-1].set_xticks(     CalibrationResult.redshift) 
        axes[-1].set_xticklabels(CalibrationResult.redshift)
        
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
    def __init__(self, CalibrationResult = None):
        '''
        Plot relation between observable quantity and halo mass.
        '''
        super().__init__(CalibrationResult)
        self.default_filename = self.quantity_name + '_qhmr'
        
    def _plot(self, CalibrationResult):
        if CalibrationResult.quantity_name != 'mstar':
            raise ValueError('Quantity-halo mass relation plot currently\
                              only supports mstar.')
        
        # general plotting configuration
        fig, ax = plt.subplots(1,1, sharex = True) 
        
        # add axes labels
        fig.supxlabel('log $M_\mathrm{h}$ [$M_\odot$]')
        fig.supylabel('log($M_*/M_\mathrm{h}$)', x=0.01)
        
        # create custom color map
        cm = LinearSegmentedColormap.from_list("Custom", ['C2','C1'],
                                               N=len(CalibrationResult.redshift))
        
        # define halo mass range
        log_m_halo = np.linspace(8,14,100)
        
        print('Put this in seperate function and then call')
        for z in CalibrationResult.redshift[::2]:
            # calculate quantity distribution and percentiles
            percentiles = []
            for m_h in log_m_halo:
                q_dist = CalibrationResult.calculate_quantity_distribution(m_h, z, int(1e+4))
                percentiles.append(calculate_percentiles(q_dist))
            percentiles = np.array(percentiles)
                     
            # plot median and 16th/84th percentile
            ax.plot(log_m_halo, percentiles[:,0], color = cm(z),
                    label = '$z$ = ' + str(z))
            
            ax.fill_between(log_m_halo, percentiles[:,1],percentiles[:,2],
                            alpha = 0.2, color = cm(z))
        
        # add legend and minor ticks
        add_legend(ax, 0)
        ax.minorticks_on()
            #ax.set_xlim([8,14])
            #ax.set_ylim([-5.5,0])
        return(fig, ax)