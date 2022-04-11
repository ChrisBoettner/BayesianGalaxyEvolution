#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 18:15:34 2022

@author: chris
"""
from matplotlib import rc_file
rc_file('model/plotting/settings.rc')
import matplotlib.pyplot as plt

from model.helper import make_list
from model.plotting.convience_functions import plot_group_data, plot_best_fit_ndf,\
                                               add_redshift_text, add_separated_legend,\
                                               turn_off_axes, save_image

default_plot_limits = {'top':0.982,  'bottom':0.113, 
                       'left':0.075, 'right':0.991} 

def plot_ndfs(CalibrationResults, save_as = None):
    '''
    Plot modelled number density functions and data for comparison. Input
    can be a single model object or a list of objects.
    If file is supposed to be saved, set saving to wanted file format extension.
    '''
    
    # make list if input is scalar
    CalibrationResults = make_list(CalibrationResults)
    
    # general plotting configuration
    fig, axes = plt.subplots(4,3, sharey = True, sharex=True)
    axes      = axes.flatten()
    fig.subplots_adjust(**default_plot_limits, 
                        hspace=0.0, wspace=0.0)
    
    # quantity specific settings
    quantity_name = CalibrationResults[0].quantity_name
    if quantity_name == 'mstar':
        xlabel = r'log $M_*$ [$M_\odot$]'
        ylabel = r'log $\phi(M_*)$ [cMpc$^{-3}$ dex$^{-1}$]'
        ncol = 1
    if quantity_name == 'Muv':
        xlabel = r'$\mathcal{M}_{UV}$'
        ylabel = r'log $\phi(\mathcal{M}_{UV})$ [cMpc$^{-3}$ dex$^{-1}$]'
        ncol = 2
    
    # plot group data points and modelled number density functions
    plot_group_data(axes, CalibrationResults[0].groups)
    
    # plot modelled number density functions
    for model in CalibrationResults: 
        plot_best_fit_ndf(axes, model)
        
    # add redshift as text to subplots
    add_redshift_text(axes, CalibrationResults[0].redshift)
                   
    # add axes labels and minor ticks
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel, x=0.01)
    axes[0].minorticks_on()
    
    # add axes limits
    axes[0].set_ylim([-6,3]) 
    
    # add legend
    add_separated_legend(axes, separation_point = len(CalibrationResults),
                         ncol = ncol)
    
    # turn off unused axes
    unused_axes_ind = list(set(range(12)) - set(CalibrationResults[0].redshift))
    turn_off_axes(axes, unused_axes_ind)
    
    # save figure
    if save_as:
        save_image(fig, quantity_name, 
                   file_name = 'ndf_'+quantity_name,
                   file_format = save_as)
    return(fig, axes)

def plot_parameter_sample(CalibrationResults):
    
    # general plotting configuration
    fig, axes = plt.subplots(3,1, sharex = True)
    fig.subplots_adjust(**default_plot_limits, 
                        hspace=0.0, wspace=0.0)
    
    # quantity specific settings
    quantity_name = CalibrationResults[0].quantity_name
    if quantity_name == 'mstar':
        ax0label = r'$\log A [ergs s$^{-1}$ Hz$^{-1}$ $M_\odot^{-1}$]',
    if quantity_name == 'Muv':
        ax0label = r'$\log A$' 
    
    # plot parameter sample
    
    # add axes labels and minor ticks
    axes[0].set_ylabel(ax0label, multialignment='center')
    axes[1].set_ylabel(r'$\gamma$')
    axes[2].set_ylabel(r'$\delta$')
    axes[2].set_xlabel(r'Redshift $z$')

    # plot parameter samples
    for z in redshift:
        print(z)
        dist_at_z  = model.distribution.at_z(z)
        # draw random sample of parameter from mcmc dists
        draw_num = int(1e+4)
        random_draw = np.random.choice(range(dist_at_z.shape[0]),
                                       size = draw_num, replace = False)     
        parameter_draw = dist_at_z[random_draw]
        
        # calculate Gaussian kde on data and evaluate from that
        color = gaussian_kde(parameter_draw.T).evaluate(parameter_draw.T)
        idx = color.argsort()
        parameter_draw, color = parameter_draw[idx], color[idx]

        for i in range(parameter_draw.shape[1]):
            x = np.repeat(z,draw_num)+np.random.normal(loc = 0, scale = 0.03, size=draw_num)
            
            ax[i].scatter(x, parameter_draw[:,i], c=color, s = 0.1, cmap = 'Oranges')
            
            #ax[i].set_xscale('log')
            ax[0].set_yscale('log')
            ax[i].set_xticks(range(0,11)); ax[2].set_xticklabels(range(0,11))
            #ax[i].minorticks_on()

    # second axis for redshift
    def z_to_t(z):
        z = np.array(z)
        t = np.array([Planck18.lookback_time(k).value for k in z])
        return(t)
    def t_to_z(t):
        t = np.array(t)
        z = np.array([z_at_value(Planck18.lookback_time, k*u.Gyr).value for k in t])
        return(z)
    ts      = np.arange(1,14,1)
    ts     = np.append(ts,13.3)
    z_at_ts = t_to_z(ts)
    ax_z    = ax[0].twiny()
    ax_z.set_xlim(ax[0].get_xlim())
    ax_z.set_xticks(z_at_ts)
    ax_z.set_xticklabels(np.append(ts[:-1].astype(int).astype(str),ts[-1].astype(str)))
    ax_z.set_xlabel('Lookback time [Gyr]')


    fig.align_ylabels(ax)

    fig.subplots_adjust(
    top=0.92,
    bottom=0.09,
    left=0.08,
    right=0.99,
    hspace=0.0,
    wspace=0.0)
    
    
