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

from scipy.optimize import curve_fit

from modelling import model, calculate_schechter_parameter, log_schechter_function, lum_to_mag, find_mode

from data.data_processing_smf  import load_data as load_data_smf
from data.data_processing_uvlf import load_data as load_data_uvlf

################## LOAD DATA AND MODEL ########################################
#%%
quantity_name = 'lum'
sample        = 'schechter' # model or schechter

mod = model(quantity_name)

if quantity_name == 'mstar':
    groups, data, _ = load_data_smf()
if quantity_name == 'lum':
    groups, data, _ = load_data_uvlf() 

log_data = [np.log10(d) for d in data]
################## CALCULATE SAMPLE FUNCTIONS #################################
#%%
redshift = range(11)

num = 100
if quantity_name == 'mstar':
    quantity   = np.logspace(6,12,100)
if quantity_name == 'lum':
    quantity   = np.logspace(24,30,100)
 
if sample == 'schechter':
    # fit Schechter functions
    _,_,_, dist = calculate_schechter_parameter(mod, quantity, redshift, num = num)
    # calculate Schechter function samples
    log_schechter = []
    for z in redshift:
        log_schechter_z = []
        for i in range(len(dist[z])):
            log_schechter_z.append(log_schechter_function(np.log10(quantity), *dist[z][i]))
        log_schechter.append(log_schechter_z)
    # calculate reference Schechter function directly fitted to data
    log_schechter_ref = []
    for z in redshift:
        inp = log_data[z][log_data[z][:,1]>-6] #remove datapoints that have number density of <10^-6
        p0 = find_mode(dist[z], [1e-20, np.amin(np.log10(quantity)), -5], [1, np.amax(np.log10(quantity)), 0])
        data_fit_params,_ = curve_fit(log_schechter_function, inp[:,0], inp[:,1],
                                      p0 = p0, maxfev = int(1e+5)) 
        log_schechter_ref.append(log_schechter_function(np.log10(quantity), *data_fit_params))
        

if sample == 'model':
    # calculate number density functions obtained from model directly
    number_dens_func = []
    for z in redshift:
        A, alpha, beta       = mod.get_parameter_sample(z, num = num)
        number_dens_func.append(mod.number_density_function(quantity, z, A, alpha, beta))
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
    # plot Schechter or model functions
    for z in redshift:
        if sample == 'schechter':
            for i in range(len(log_schechter[z])):
                ax[z].plot(np.log10(quantity), log_schechter[z][i], 
                           linewidth = linewidth, alpha = alpha, color = color) 
            # add Schechter function directly fitted to data
            ax[z].plot(np.log10(quantity), log_schechter_ref[z],
                       linewidth = 10*linewidth, alpha = 2*alpha, color = 'tomato')                                              
        if sample == 'model':
            for i in range(len(number_dens_func[z])):
                ax[z].plot(np.log10(quantity), np.log10(number_dens_func[z][i]),
                            linewidth = linewidth, alpha = alpha, color = color)
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
    mag = lum_to_mag(quantity)
    for z in redshift:
        if sample == 'schechter':
            for i in range(len(log_schechter[z])):
                ax[z].plot(mag, log_schechter[z][i], 
                           linewidth = linewidth, alpha = alpha, color = color)
            # add Schechter function directly fitted to data
            ax[z].plot(mag, log_schechter_ref[z],
                       linewidth = 10*linewidth, alpha = 2*alpha, color = 'red',
                       linestyle = '--')            
        if sample == 'model':
            for i in range(len(number_dens_func[z])):
                ax[z].plot(mag, np.log10(number_dens_func[z][i]), linewidth = linewidth,
                            alpha = alpha, color = color)
    # plot group data
    for g in groups:
        for z in g.redshift:
            ax[z].errorbar(g.data_at_z(z).mag, g.data_at_z(z).phi, 
                           [g.data_at_z(z).lower_error,g.data_at_z(z).upper_error],
                           capsize = 3, fmt = g.marker, color = g.color,
                           label = g.label, alpha = 0.4)
            if z<3:            # ax[z].plot(mag, np.log10(number_dens_func[z][i]), linewidth = linewidth,
            #             alpha = alpha, color = color)
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
              prop={'size': 12}, loc = 4, ncol = 2)