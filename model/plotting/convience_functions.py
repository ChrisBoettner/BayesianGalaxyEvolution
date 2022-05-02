#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 18:16:20 2022

@author: chris
"""
import numpy as np
from matplotlib.lines import Line2D

from model.helper import make_array, make_list, pick_from_list
from model.analysis.calculations import calculate_best_fit_ndf

################ PLOT DATA ####################################################


def plot_group_data(axes, groups):
    ''' Use list of group objects to plot group data. '''
    for g in groups:
        for z in g.redshift:
            axes[z].errorbar(g.data_at_z(z).quantity,
                             g.data_at_z(z).phi,
                             [g.data_at_z(z).lower_error,
                              g.data_at_z(z).upper_error],
                             capsize=3,
                             fmt=g.marker,
                             color=g.color,
                             label=g.label,
                             alpha=0.4)
    return


def plot_best_fit_ndf(axes, ModelResult):
    ''' Calculate and plot best fit number density functions. '''
    ndfs = calculate_best_fit_ndf(ModelResult, ModelResult.redshift)

    for z in ModelResult.redshift:
        color = pick_from_list(ModelResult.color, z)
        label = pick_from_list(ModelResult.label, z)
        axes[z].plot(ndfs[z][:, 0],
                     ndfs[z][:, 1],
                     linestyle=ModelResult.linestyle,
                     label=label,
                     color=color)
    return(ndfs)

################ ADD TEXT TO PLOT #############################################


def add_redshift_text(axes, redshifts):
    ''' Add current redshift as text to upper plot corner. '''
    for z in redshifts:
        axes[z].text(0.97, 0.94, 'z=' + str(z), size=11,
                     horizontalalignment='right',
                     verticalalignment='top',
                     transform=axes[z].transAxes)
    return


################ LEGEND STUFF #################################################


def add_legend(axes, ind, sort=False, **kwargs):
    '''
    Add legend at axis given by ind. If sort is true, sort labels before
    displaying legend.
    '''
    axes = make_array(axes)

    labels = remove_double_labels(axes)
    if sort:
        labels = dict(sorted(labels.items()))

    axes[ind].legend(list(labels.values()),
                     list(labels.keys()),
                     frameon=False,
                     prop={'size': 12}, **kwargs)
    return


def add_separated_legend(axes, separation_point, ncol=1):
    '''
    Add part of legend to first subplot and part to last subplot, devided by
    separation_point. Can also adjust number of columns of legend.
    '''
    labels = remove_double_labels(axes)

    axes[0].legend(list(labels.values())[:separation_point],
                   list(labels.keys())[:separation_point],
                   frameon=False,
                   prop={'size': 12}, loc=9)
    axes[-1].legend(list(labels.values())[separation_point:],
                    list(labels.keys())[separation_point:],
                    frameon=False,
                    prop={'size': 10}, loc=4, ncol=ncol)
    return


def remove_double_labels(axes):
    '''  
    Remove duplicates in legend that have same label. Also sorts labels so that
    Line2D objects appear first.
    '''
    axes = make_array(axes)

    handles, labels = [], []
    for a in axes.flatten():
        handles_, labels_ = a.get_legend_handles_labels()
        handles += handles_
        labels += labels_

    # sort so that Lines2D objects appear first
    handles        = np.array(handles , dtype='object')
    labels         = np.array(labels, dtype='object')
    lines_idx      = [isinstance(handle, Line2D) for handle in handles]
    handles_sorted = np.concatenate((handles[lines_idx],
                                    handles[np.logical_not(lines_idx)]))
    labels_sorted  = np.concatenate((labels[lines_idx],
                                    labels[np.logical_not(lines_idx)]))
    
    by_label = dict(zip(labels_sorted, handles_sorted))
    return(by_label)

################ AXES AND LIMITS ##############################################


def turn_off_axes(axes):
    ''' Turn of axes for subplots that are not used. '''
    for ax in axes.flatten():
        if (not ax.lines) and (not ax.patches):
            ax.axis('off')
    return

def get_distribution_limits(ModelResults):  
    ''' 
    Get minimum and maximum values for distributions across redshifts
    and different models (e.g. for pdf plot limits).
    '''
    
    ModelResults = make_list(ModelResults)
    
    max_values, min_values = {}, {}
    for Model in ModelResults:
        for z in Model.redshift:
            distribution = Model.distribution.at_z(z)
            for i in range(distribution.shape[1]):
                max_values.setdefault(i, -np.inf) # check if key already exists,
                                                  # if not, create it and put value
                                                  # to -infinity
                min_values.setdefault(i, np.inf)  
                
                current_max = np.amax(distribution[:,i])
                current_min = np.amin(distribution[:,i])
                
                # update maximum and minimum values
                if current_max>max_values[i]:
                    max_values[i] = current_max
                if current_min< min_values[i]:
                    min_values[i] = current_min    
                    
    # return as list of limits
    limits = list(zip(min_values.values(),max_values.values()))
    return(limits)
    
    
    
    