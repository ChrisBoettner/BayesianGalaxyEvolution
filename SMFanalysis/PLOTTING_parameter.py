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
                        prior_model, mode).plot_parameter(['C2']*4+['C1']*6,
                                                          'o',
                                                          '-', 
                                                          ['Stellar + Black Hole Feedback']*4\
                                                          +['Stellar Feedback']*6)
################## PLOTTING ###################################################
#%%
plt.close('all')

redshift = np.arange(1,11)  
## OVERVIEW      
fig, ax = plt.subplots(3,1, sharex = True)
ax[0].set_ylabel('A')
ax[1].set_ylabel(r'$\alpha$')
ax[2].set_ylabel(r'$\beta$')
ax[2].set_xlabel('Lookback time [Gyr]')
for z in redshift:
    param_at_z = model.parameter.at_z(z)
    if fitting_method == 'mcmc':
        dist_at_z = model.distribution.at_z(z)
        lower     = param_at_z - np.percentile(dist_at_z, 16, axis = 0)
        upper     = np.percentile(dist_at_z, 84, axis = 0) - param_at_z
    for i in range(len(param_at_z)):
        t = Planck18.lookback_time(z).value
        if fitting_method == 'mcmc':
            ax[i].errorbar(t, param_at_z[i], yerr = np.array([[lower[i],upper[i]]]).T, capsize=3,
                          marker = model.marker, label = model.label[z-1], color = model.color[z-1])
        else:
            ax[i].scatter(t, param_at_z[i],
                          marker = model.marker, label = model.label[z-1], color = model.color[z-1])
        #ax[i].set_xscale('log')
        #ax[i].set_yscale('log')
        #ax[i].set_xticks(range(1,11)); ax[2].set_xticklabels(range(1,11))
        ax[i].minorticks_on()

# second axis for redshift
def z_to_t(z):
    z = np.array(z)
    t = np.array([Planck18.lookback_time(k).value for k in z])
    return(t)
def t_to_z(t):
    t = np.array(t)
    z = np.array([z_at_value(Planck18.lookback_time, k*u.Gyr) for k in t])
    return(z)
ts_at_z = z_to_t(redshift)
ax_z = ax[0].twiny()
ax_z.set_xlim(ax[0].get_xlim())
ax_z.set_xticks(ts_at_z)
ax_z.set_xticklabels(redshift.astype(str))
ax_z.set_xlabel(r"Redshift $z$")


fig.align_ylabels(ax)

fig.subplots_adjust(
top=0.917,
bottom=0.092,
left=0.049,
right=0.991,
hspace=0.0,
wspace=0.0)