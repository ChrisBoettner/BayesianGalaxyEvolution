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

from uvlf_modelling  import model_container
from data_processing import load_data, lum_to_mag

# coose option
fitting_method = 'least_squares'     # 'least_squares' or 'mcmc'   
prior_model    = 'uniform'  # 'uniform', 'marginal' or 'full'
mode           = 'temp'   # 'saving', 'loading' or 'temp'

# load data
groups, lfs, hmfs = load_data()

# create model lfs
no_feedback   = model_container(lfs, hmfs, 'none', fitting_method,
                          prior_model, mode).plot_parameter('black', 'o', '-',  'No Feedback')
sn_feedback   = model_container(lfs, hmfs, 'sn',   fitting_method,
                          prior_model, mode).plot_parameter('C1',    's', '--', 'Stellar Feedback')
snbh_feedback = model_container(lfs, hmfs, 'both', fitting_method,
                          prior_model, mode).plot_parameter('C2',    'v', '-.', 'Stellar + Black Hole Feedback')
models = [no_feedback, sn_feedback, snbh_feedback]

################## PLOTTING ###################################################
#%%
plt.close('all')

redshift = np.arange(0,11)  
## LUMINOSTIY FUNCTION
fig, ax = plt.subplots(4,3, sharey='row', sharex=True); ax = ax.flatten()
fig.subplots_adjust(top=0.982,bottom=0.113,left=0.075,right=0.991,hspace=0.0,wspace=0.0)
# plot group data
for g in groups:
    for z in g.redshift:
        ax[z].errorbar(g.data_at_z(z).mag, g.data_at_z(z).phi, 
                       [g.data_at_z(z).lower_error,g.data_at_z(z).upper_error],
                       capsize = 3, fmt = g.marker, color = g.color,
                       label = g.label, alpha = 0.4)
        if z<3:
            ax[z].set_ylim([-5,3])
        else:
            ax[z].set_ylim([-6,3])         
# plot modelled lf
for model in models:
    for z in redshift:
        mag = lum_to_mag(model.lf.at_z(z)[:,0]*1e+18) # return units back, before converting to moag
        phi = np.log10(model.lf.at_z(z)[:,1])
        phi[phi == phi[-1]] = np.nan # do not plot phi values that correspond to m_h above m_h_max
        ax[z].plot(mag, phi, linestyle=model.linestyle, label = model.label,
                   color = model.color)      
# fluff
#fig.supxlabel('log $L_{\\nu}^{UV}/10^{18}$ [ergs s$^{-1}$ Hz$^{-1}$]')
fig.supxlabel(r'$M_{UV}$')
fig.supylabel('log $\phi(L_\\nu^{UV})$ [cMpc$^{-3}$ dex$^{-1}$]', x=0.01)
for i, a in enumerate(ax):
    a.minorticks_on()
    if i == 0:
        a.text(0.97, 0.94, 'z$<$' +str(i+0.5), size = 11,
               horizontalalignment='right', verticalalignment='top',
               transform=a.transAxes)
    elif i<len(redshift):
        a.text(0.97, 0.94, str(i-0.5) + '$<$z$<$' +str(i+0.5), size = 11,
               horizontalalignment='right', verticalalignment='top',
               transform=a.transAxes)
ax[-1].axis('off');
# legend
handles, labels = [], []
for a in ax:
    handles_, labels_ = a.get_legend_handles_labels()
    handles += handles_
    labels += labels_
by_label = dict(zip(labels, handles))
ax[0].legend( list(by_label.values())[:3], list(by_label.keys())[:3],
             frameon=False, prop={'size': 12})
ax[-1].legend(list(by_label.values())[3:], list(by_label.keys())[3:],
              frameon=False, prop={'size': 12}, loc = 4, ncol = 2)