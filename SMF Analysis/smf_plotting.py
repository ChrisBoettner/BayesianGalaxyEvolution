#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 12:10:37 2021

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
fitting_method = 'mcmc'    # 'least_squares' or 'mcmc'   
prior_model    = 'uniform' # 'uniform', 'marginal' or 'full'
mode           = 'loading' # 'saving', 'loading' or 'temp'

# load data
groups, smfs, hmfs = load_data()

# create model smfs
no_feedback   = model_container(smfs, hmfs, 'none', fitting_method,
                          prior_model, mode).plot_parameter('black', 'o', '-',  'No Feedback')
sn_feedback   = model_container(smfs, hmfs, 'sn',   fitting_method,
                          prior_model, mode).plot_parameter('C1',    's', '--', 'Stellar Feedback')
snbh_feedback = model_container(smfs, hmfs, 'both', fitting_method,
                          prior_model, mode).plot_parameter('C2',    'v', '-.', 'Stellar + Black Hole Feedback')
models = [no_feedback, sn_feedback, snbh_feedback]

################## PLOTTING ###################################################
plt.close('all')

## STELLAR MASS FUNCTION
redshift = np.arange(1,11)  
fig, ax = plt.subplots(4,3, sharex=True, sharey=True); ax = ax.flatten()
fig.subplots_adjust(top=0.982,bottom=0.113,left=0.075,right=0.991,hspace=0.0,wspace=0.0)
# plot group data
for g in groups:
    for z in g.redshift:
        ax[z-1].errorbar(g.data_at_z(z).mass, g.data_at_z(z).phi, [g.data_at_z(z).lower_error,g.data_at_z(z).upper_error],
                         capsize = 3, fmt = g.marker, color = g.color, label = g.label, alpha = 0.4)
        ax[z-1].set_xlim([6.67,12.3])
        ax[z-1].set_ylim([-6,3])  
# plot modelled smf
for model in models:
    for z in redshift:
        ax[z-1].plot(np.log10(model.smf.at_z(z)[:,0]), np.log10(model.smf.at_z(z)[:,1]),
                  linestyle=model.linestyle, label = model.label, color = model.color)      
# fluff
fig.supxlabel('log[$M_*/M_\odot$]')
fig.supylabel('log[$\phi(M_*)$ cMpc$^3$ dex]', x=0.01)
for i, a in enumerate(ax):
    a.minorticks_on()
    if i<len(redshift):
        a.text(0.97, 0.94, str(i+0.5) + '$<$z$<$' +str(i+1.5), size = 11,
               horizontalalignment='right', verticalalignment='top', transform=a.transAxes)
# legend
handles, labels = [], []
for a in ax:
    handles_, labels_ = a.get_legend_handles_labels()
    handles += handles_
    labels += labels_
by_label = dict(zip(labels, handles))
ax[0].legend( list(by_label.values())[:3], list(by_label.keys())[:3], frameon=False,
             prop={'size': 12}, loc = 3)
ax[-1].legend(list(by_label.values())[3:], list(by_label.keys())[3:], frameon=False,
              prop={'size': 12})

## PARAMETER EVOLUTION      
fig, ax = plt.subplots(3,1, sharex=True)
ax[0].set_ylabel('A')
ax[1].set_ylabel(r'$\alpha$')
ax[2].set_ylabel(r'$\beta$')
ax[2].set_xlabel('Lookback time [Gyr]')
for model in models:
    parameter_number = len(model.parameter.data[0])
    for i in range(parameter_number):
        param_at_z = [model.parameter.at_z(z)[i] for z in redshift]
        t          = [Planck18.lookback_time(z).value for z in redshift]
        ax[i].scatter(t, param_at_z,
                      marker = model.marker, label = model.label, color = model.color)
# second axis
def z_to_t(z):
    z = np.array(z)
    t = np.array([Planck18.lookback_time(k).value for k in z])
    return(t)
def t_to_z(t):
    t = np.array(t)
    z = np.array([z_at_value(Planck18.lookback_time, k*u.Gyr) for k in t])
    return(z)
zax = ax[0].secondary_xaxis('top', functions=(t_to_z, z_to_t))
zax.set_xlabel('Redshift $z$')
_ = zax.set_xticks(range(1,11))
