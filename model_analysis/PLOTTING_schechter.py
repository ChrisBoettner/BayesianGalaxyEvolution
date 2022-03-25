#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 13:32:25 2022

@author: chris
"""

from matplotlib import rc_file
rc_file('plots/settings.rc')  # <-- the file containing your settings

import numpy as np
import matplotlib.pyplot as plt

from modelling import model, calculate_schechter_parameter, log_schechter_function, lum_to_mag

from data.data_processing_smf  import load_data as load_data_smf
from data.data_processing_uvlf import load_data as load_data_uvlf

################## LOAD DATA AND MODEL ########################################
#%%
quantity_name = 'lum'

mod = model(quantity_name)

if quantity_name == 'mstar':
    groups, _, _ = load_data_smf()
if quantity_name == 'lum':
    groups, _, _ = load_data_uvlf() 

################## CALCULATE SCHECHTER PARAMETER ##############################

redshift = range(11)

# Schechter parameters
num        = 1000
if quantity_name == 'mstar':
    quantity   = np.logspace(6,12,100)
if quantity_name == 'lum':
    quantity   = np.logspace(24,30,100)
_,_,_, dist = calculate_schechter_parameter(mod, quantity, redshift, num = num)

################## PLOTTING ###################################################
#%%
plt.close('all')

linewidth = 0.2
alpha     = 0.4
color     = 'grey'

fig, ax = plt.subplots(4,3, sharey = 'row', sharex=True); ax = ax.flatten()
fig.subplots_adjust(top=0.982,bottom=0.113,left=0.075,right=0.991,hspace=0.0,wspace=0.0)

if quantity_name == 'mstar':
    fig.supxlabel('log $M_*$ [$M_\odot$]')
    fig.supylabel('log $\phi(M_*)$ [cMpc$^{-3}$ dex$^{-1}$]', x=0.01)                 
    # plot Schechter functions
    for z in redshift:
        for i in range(len(dist[z])):
            log_schechter = log_schechter_function(np.log10(quantity), *dist[z][i])
            ax[z].plot(np.log10(quantity), log_schechter, linewidth = linewidth,
                       alpha = alpha, color = color)
    # plot group data
    for g in groups:
        for z in g.redshift:
            ax[z].errorbar(g.data_at_z(z).mass, g.data_at_z(z).phi, [g.data_at_z(z).lower_error,g.data_at_z(z).upper_error],
                             capsize = 3, fmt = g.marker, color = g.color, label = g.label, alpha = 0.4)
            #ax[z].set_xlim([6.67,12.3])
            if z<3:
                ax[z].set_ylim([-5,3])
            else:
                ax[z].set_ylim([-6,3]) 
    
if quantity_name == 'lum':
    fig.supxlabel(r'$M_{UV}$')
    fig.supylabel('log $\phi(M_\\nu^{UV})$ [cMpc$^{-3}$ dex$^{-1}$]', x=0.01)        
    # plot Schechter functions
    for z in redshift:
        for i in range(len(dist[z])):
            log_schechter = log_schechter_function(np.log10(quantity), *dist[z][i])
            mag = lum_to_mag(quantity)
            ax[z].plot(mag, log_schechter, linewidth = linewidth,
                       alpha = alpha, color = color)
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

# fluff    
for i, a in enumerate(ax):
    a.minorticks_on()
    if i == 0:
        a.text(0.97, 0.94, 'z$<$' +str(i+0.5), size = 11,
               horizontalalignment='right', verticalalignment='top', transform=a.transAxes)
    elif i<len(redshift):
        a.text(0.97, 0.94, str(i-0.5) + '$<$z$<$' +str(i+0.5), size = 11,
               horizontalalignment='right', verticalalignment='top', transform=a.transAxes)
        
ax[-1].axis('off');
# legend
handles, labels = [], []
for a in ax:
    handles_, labels_ = a.get_legend_handles_labels()
    handles += handles_
    labels += labels_
by_label = dict(zip(labels, handles))
ax[-1].legend(list(by_label.values()), list(by_label.keys()), frameon=False,
              prop={'size': 12}, loc = 4)