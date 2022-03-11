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

from scipy.stats import gaussian_kde

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
feedback_model = ['both']*5+['sn']*6
prior_model    = 'full'

# create model smfs
model = model_container(smfs, hmfs, feedback_model, fitting_method,
                        prior_model, mode).plot_parameter(['C2']*5+['C1']*6,
                                                          'o',
                                                          '-', 
                                                          ['Stellar + Black Hole Feedback']*5\
                                                          +['Stellar Feedback']*6)
################## PLOTTING ###################################################
#%%
plt.close('all')

redshift = np.arange(0,11)  
## OVERVIEW      
fig, ax = plt.subplots(3,1, sharex = True)
ax[0].set_ylabel(r'$A$')
ax[1].set_ylabel(r'$\alpha$')
ax[2].set_ylabel(r'$\beta$')
ax[2].set_xlabel(r'Redshift $z$')
    
# plot 'best-fit' parameter
# for z in redshift:
#     param_at_z = model.parameter.at_z(z)
#     if fitting_method == 'mcmc':
#         dist_at_z = model.distribution.at_z(z)
#         lower     = param_at_z - np.percentile(dist_at_z, 16, axis = 0)
#         upper     = np.percentile(dist_at_z, 84, axis = 0) - param_at_z
#     for i in range(len(param_at_z)):
#         if fitting_method == 'mcmc':
#             ax[i].errorbar(z, param_at_z[i], yerr = np.array([[lower[i],upper[i]]]).T, capsize=3,
#                           marker = model.marker, label = model.label[z], color = model.color[z])
#         else:
#             ax[i].scatter(z, param_at_z[i],
#                           marker = model.marker, label = model.label[z], color = model.color[z])
#         #ax[i].set_xscale('log')
#         ax[0].set_yscale('log')
#         ax[i].set_xticks(range(0,11)); ax[2].set_xticklabels(range(0,11))
#         #ax[i].minorticks_on()

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
        ax[i].set_xticks(redshift); ax[2].set_xticklabels(redshift)
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
left=0.055,
right=0.99,
hspace=0.0,
wspace=0.0)