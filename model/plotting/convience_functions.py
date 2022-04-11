#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 18:16:20 2022

@author: chris
"""
import numpy as np

from model.helper import make_list, pick_from_list

################ PLOT DATA ####################################################
def plot_group_data(axes, groups):
    ''' Use list of group objects to plot group data. '''
    for g in groups:
        for z in g.redshift:
            axes[z].errorbar(g.data_at_z(z).quantity, 
                             g.data_at_z(z).phi,
                             [g.data_at_z(z).lower_error,g.data_at_z(z).upper_error],
                             capsize = 3,
                             fmt = g.marker,
                             color = g.color,
                             label = g.label,
                             alpha = 0.4)
    return

def plot_best_fit_ndf(axes, CalibrationResult):
    ''' Calculate and plot best fit number density functions. '''
    for z in CalibrationResult.redshift:
        quantity, ndf = CalibrationResult.calculate_ndf_curve(z, 
                                                              CalibrationResult.parameter.at_z(z))
            
        color = pick_from_list(CalibrationResult.color, z)
        axes[z].plot(quantity,
                     ndf,
                     linestyle = CalibrationResult.linestyle,
                     label     = CalibrationResult.label,
                     color     = color)
    return(quantity, ndf)


################ ADD TEXT TO PLOT #############################################
def add_redshift_text(axes, redshifts):
    ''' Add current redshift as text to upper plot corner. '''
    for z in redshifts:
        axes[z].text(0.97, 0.94, 'z=' +str(z), size = 11,
               horizontalalignment='right',
               verticalalignment='top',
               transform=axes[z].transAxes)
    return


################ LEGEND STUFF #################################################
def add_legend(axes, ind, sort = False, **kwargs):
    '''
    Add legend at axis given by ind. If sort is true, sort labels before
    displaying legend.
    '''
    
    labels = remove_double_labels(axes)
    if sort:
        labels =  dict(sorted(labels.items()))
    
    axes[ind].legend(list(labels.values()),
                     list(labels.keys()),
                     frameon=False,
                     prop={'size': 12}, **kwargs)
    return

def add_separated_legend(axes, separation_point, ncol = 1):
    ''' 
    Add part of legend to first subplot and part to last subplot, devided by 
    separation_point. Can also adjust number of columns of legend.     
    '''
    labels = remove_double_labels(axes)
    
    axes[0].legend(list(labels.values())[:separation_point],
                   list(labels.keys())[:separation_point],
                   frameon=False,
                   prop={'size': 12}, loc = 9)
    axes[-1].legend(list(labels.values())[separation_point:],
                    list(labels.keys())[separation_point:],
                    frameon=False,
                    prop={'size': 12}, loc = 4, ncol = ncol)
    return

def remove_double_labels(axes):
    '''  Remove duplicates in legend that have same label. '''
    handles, labels = [], []
    for a in axes.flatten():
        handles_, labels_ = a.get_legend_handles_labels()
        handles += handles_
        labels  += labels_
    by_label = dict(zip(labels, handles))
    return(by_label)

################ AXES AND LIMITS ##############################################    
def turn_off_axes(axes):
    ''' Turn of axes for subplots that are not used. '''
    for ax in axes.flatten():
        if (not ax.lines) and (not ax.patches):
            ax.axis('off')
    return