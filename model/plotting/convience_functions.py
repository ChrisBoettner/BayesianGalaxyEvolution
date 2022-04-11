#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 18:16:20 2022

@author: chris
"""
from pathlib import Path

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
        
        if isinstance(CalibrationResult.color,str):
            color = CalibrationResult.color
        else:
            color = CalibrationResult.color[z]
            
        axes[z].plot(quantity,
                     ndf,
                     linestyle = CalibrationResult.linestyle,
                     label     = CalibrationResult.label,
                     color     = color)
    return


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
def add_separated_legend(axes, separation_point, ncol = 1):
    ''' 
    Add part of legend to first subplot and part to last subplot, devided by 
    separation_point. Can also adjust number of columns of legend.     
    '''
    labels = remove_double_labeles(axes)
    
    axes[0].legend(list(labels.values())[:separation_point],
                   list(labels.keys())[:separation_point],
                   frameon=False,
                   prop={'size': 12}, loc = 9)
    axes[-1].legend(list(labels.values())[separation_point:],
                    list(labels.keys())[separation_point:],
                    frameon=False,
                    prop={'size': 12}, loc = 4, ncol = ncol)
    return

def remove_double_labeles(axes):
    '''
    Remove duplicates in legend that have same label.
    '''
    handles, labels = [], []
    for a in axes:
        handles_, labels_ = a.get_legend_handles_labels()
        handles += handles_
        labels += labels_
    by_label = dict(zip(labels, handles))
    return(by_label)

################ AXES AND LIMITS ##############################################    
def turn_off_axes(axes, indeces):
    ''' Turn of axes for subplots that are not used (specified by their indices.'''
    for ax in axes:
        if not ax.lines:
            ax.axis('off')
    return


################ SAVING #######################################################
def save_image(fig, quantity_name, file_name, file_format):
    '''
    Save plot to file. Input fig object that contains the graphic, the 
    quantity_name (for deciding saving path), name of the file, and
    file extension (e.g. \'pdf\' or \'png\').'
    '''
    path = 'plots/' + quantity_name + '/'
    Path(path).mkdir(parents=True, exist_ok=True) 
    file = file_name + '.' + file_format
    fig.savefig(path + file)
