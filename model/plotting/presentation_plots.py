#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 13:56:37 2022

@author: chris
"""
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc_file
rc_file('model/plotting/settings.rc')

from model.plotting.plotting import Plot
from model.plotting.convience_functions import turn_off_frame


class Plot_feedback_illustration(Plot):
    def __init__(self, ModelResults):
        '''
        Plot an illustration of the feedback effects by comparing ndf with
        and without feedback.
        '''
        super().__init__(ModelResults)
        self.quantity_name    = 'presentations' # save in seperate presentation
                                                # folder
        self.default_filename = 'feedback_illustration'

    def _plot(self, ModelResult):
        
        if ModelResult.feedback_name not in ['stellar_blackhole', 'changing']:
            raise ValueError('Currently only supports stellar_blackhole feedback.')

        # calculate ndfs
        z = ModelResult.redshift[0]
        
        parameter = ModelResult.parameter.at_z(z)
        ndf_with_feedback    = np.array(ModelResult.calculate_ndf(z, parameter)).T
        ndf_without_feedback = np.array(ModelResult.calculate_ndf(z, 
                                                                  [parameter[0],
                                                                   0,0])).T
        turnover_mass        = ModelResult.feedback_model.at_z(z).\
                                           calculate_log_quantity(ModelResult.\
                                           quantity_options['log_m_c'], 
                                           *parameter) 
        turnover_idx         = np.argmax(ndf_with_feedback[:,0]>turnover_mass)
        
        
        # general plotting configuration
        fig, ax = plt.subplots(1, 1, sharex=True)
        fig.subplots_adjust(**self.plot_limits)
        

        # quantity specific settings
        xlabel, ylabel  = ModelResult.quantity_options['ndf_xlabel'],\
                          ModelResult.quantity_options['ndf_ylabel']

        # add axes labels
        fig.supxlabel(xlabel)
        fig.supylabel(ylabel, x=0.01)

        # plot group data points
        groups = ModelResult.groups    
        for g in groups:
            if z in g.redshift:
                ax.errorbar(g.data_at_z(z).quantity,
                            g.data_at_z(z).phi,
                            [g.data_at_z(z).lower_error,
                             g.data_at_z(z).upper_error],
                            capsize=3,
                            fmt=g.marker,
                            color=g.color,
                            label=g.label,
                            alpha=ModelResult.quantity_options['marker_alpha'])
                
        ## add model ndfs
        #  no feedback
        ax.plot(ndf_without_feedback[:, 0],ndf_without_feedback[:, 1],
                color = 'black')
        #  stellar feedback dominated
        ax.plot(ndf_with_feedback[:turnover_idx+1, 0],
                ndf_with_feedback[:turnover_idx+1, 1],
                color = 'C1')
        #  AGN feedback dominated        
        ax.plot(ndf_with_feedback[turnover_idx:, 0],
                ndf_with_feedback[turnover_idx:, 1],
                color = 'C2')

        # add axes limits
        quantity_range = ModelResult.quantity_options['quantity_range']
        ax.set_ylim(ModelResult.quantity_options['ndf_y_axis_limit'])
        ax.set_xlim([quantity_range[0],
                     quantity_range[-1]])
        
        return(fig, ax)