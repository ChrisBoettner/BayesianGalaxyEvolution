#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 10:43:45 2021

@author: boettner
"""

from matplotlib import rc_file
rc_file('plots/settings.rc')  # <-- the file containing your settings

import numpy as np
import matplotlib.pyplot as plt

from smf_modelling   import model_container
from data_processing import load_data

import astropy.units as u
from astropy.cosmology import Planck18, z_at_value

# coose option
fitting_method = 'mcmc' 
mode           = 'loading'  

# load data
groups, smfs, hmfs = load_data()

# load model
feedback_model = ['both']*4+['sn']*6
prior_model    = 'full'

# create model smfs
model = model_container(smfs, hmfs, feedback_model, fitting_method,
                        prior_model, mode).plot_parameter('C0', 'o', '-', '')
################## PLOTTING ###################################################
#%%
plt.close('all')

redshift = np.arange(1,11)  
## OVERVIEW      
fig, ax = plt.subplots(3,1, sharex = True)
ax[0].set_ylabel('A')
ax[1].set_ylabel(r'$\alpha$')
ax[2].set_ylabel(r'$\beta$')
#ax[2].set_xlabel('Lookback time [Gyr]')
ax[2].set_xlabel('Redshift z')
for z in redshift:
    param_at_z = model.parameter.at_z(z)
    if fitting_method == 'mcmc':
        dist_at_z = model.distribution.at_z(z)
        lower     = np.percentile(dist_at_z, 16, axis = 0)
        upper     = np.percentile(dist_at_z, 100, axis = 0)
    for i in range(len(param_at_z)):
        #t = Planck18.lookback_time(z).value
        if fitting_method == 'mcmc':
            ax[i].errorbar(z, param_at_z[i], yerr = np.array([[lower[i],upper[i]]]).T, capsize=3,
                          marker = model.marker, label = model.label, color = model.color)
        else:
            ax[i].scatter(z, param_at_z[i],
                          marker = model.marker, label = model.label, color = model.color)
        ax[i].set_xscale('log')
        ax[i].set_xticks(range(1,11)); ax[2].set_xticklabels(range(1,11))
        ax[i].minorticks_off()

        
fig.align_ylabels(ax)

#second axis
# def z_to_t(z):
#     z = np.array(z)
#     t = np.array([Planck18.lookback_time(k).value for k in z])
#     return(t)
# def t_to_z(t):
#     t = np.array(t)
#     z = np.array([z_at_value(Planck18.lookback_time, k*u.Gyr) for k in t])
#     return(z)
# zax = ax[0].secondary_xaxis('top', functions=(z_to_t, t_to_z))
# zax.set_xlabel('Redshift $z$')
# _ = zax.set_xticks(range(1,11))