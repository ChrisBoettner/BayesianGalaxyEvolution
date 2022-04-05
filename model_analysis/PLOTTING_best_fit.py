#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 18:24:14 2022

@author: chris
"""

from matplotlib import rc_file
rc_file('plots/settings.rc')  # <-- the file containing your settings

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from modelling import model, log_schechter_function, lum_to_mag

from data.data_processing_smf  import load_data as load_data_smf
from data.data_processing_uvlf import load_data as load_data_uvlf

################## LOAD DATA AND MODEL ########################################
#%%
quantity_name = 'lum'

mod = model(quantity_name)

if quantity_name == 'mstar':
    groups, data, _ = load_data_smf()
if quantity_name == 'lum':
    groups, data, _ = load_data_uvlf() 

log_data = [np.log10(d) for d in data]
log_data = [l[l[:,1]>-6] for l in log_data] #remove datapoints that have number density of <10^-6

redshift = range(11)
if quantity_name == 'mstar':
    quantity   = np.logspace(6,12,100)
if quantity_name == 'lum':
    quantity   = np.logspace(24,30,100)
################## CALCULATE BEST FITS ########################################
#%%
# Calculate best fit model function (with successive prior)
best_fit_model = []
for z in redshift:
    best_model_params = mod.parameter.at_z(z)
    if len(best_model_params) == 2: # beta = 0, if beta not in model
       best_model_params =  np.append(best_model_params,0)
    best_fit_model.append(mod.number_density_function(quantity, z, *best_model_params)[0])

# Calculate best fit Schechter parameter from data directly
best_fit_schechter_data = []
for z in redshift:
    schechter_params,_ = curve_fit(log_schechter_function, log_data[z][:,0], log_data[z][:,1],
                                   maxfev = int(1e+5)) 
    print(schechter_params)
    best_fit_schechter_data.append(log_schechter_function(np.log10(quantity), *schechter_params))

print()
# Calculate best fit Schechter parameter to model function 
best_fit_schechter_model = []
for z in redshift:
    ndf = np.copy(best_fit_model[z])
    # cut repeating end
    idx = np.argwhere(np.around(ndf/ndf[-1],4) == 1)[0][0] # index of first occurence of repeated value
    ndf = ndf[:idx]
    q   = quantity[:idx] # cut input variable to same length for fitting
    schechter_params,_ = curve_fit(log_schechter_function, np.log10(q), np.log10(ndf),
                                   maxfev = int(1e+5)) 
    print(schechter_params)
    best_fit_schechter_model.append(log_schechter_function(np.log10(quantity), *schechter_params))


################## PLOTTING ###################################################
#%%
plt.close('all')

linewidth = 3
alpha     = 0.9
color     = 'grey'

fig, ax = plt.subplots(4,3, sharey = 'row', sharex=True); ax = ax.flatten()
fig.subplots_adjust(top=0.982,bottom=0.113,left=0.075,right=0.991,hspace=0.0,wspace=0.0)

if quantity_name == 'mstar':
    fig.supxlabel('log $M_*$ [$M_\odot$]')
    fig.supylabel('log $\phi(M_*)$ [cMpc$^{-3}$ dex$^{-1}$]', x=0.01)                 
    # plot Schechter or model functions
    for z in redshift:
            ax[z].plot(np.log10(quantity), best_fit_schechter_data[z],
                        alpha = alpha, color = color, 
                        linestyle = '--',
                        label = 'Best Fit Schechter Function To Data')  
            ax[z].plot(np.log10(quantity), best_fit_schechter_model[z],
                       alpha = alpha, color = color,
                        label = 'Best Fit Schechter Function To Model') 
            ax[z].plot(np.log10(quantity), np.log10(best_fit_model[z]), 
                       alpha = alpha, color = 'C2',
                       label = 'Best Fit Model Function') 
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
            ax[z].plot(mag, best_fit_schechter_data[z],
                       alpha = alpha, color = color, 
                        linestyle = '--',
                        label = 'Best Fit Schechter Function To Data') 
            ax[z].plot(mag, best_fit_schechter_model[z],
                        alpha = alpha, color = color,
                        label = 'Best Fit Schechter Function To Model')
            ax[z].plot(mag, np.log10(best_fit_model[z]), 
                       alpha = alpha, color = 'C2',
                       label = 'Best Fit Model Function') 
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
ax[0].legend( list(by_label.values())[:3][::-1], list(by_label.keys())[:3][::-1],
             frameon=False, prop={'size': 12})
ax[-1].legend(list(by_label.values())[3:], list(by_label.keys())[3:],
              frameon=False, prop={'size': 12}, loc = 4, ncol = 2)

# TEST INDIVIDUALLY
# num = 1
# z = 7
# A, alpha, beta       = mod.get_parameter_sample(z, num = num)
# ndf     = np.log10(mod.number_density_function(quantity, z, A, alpha, beta)[0])
# idx = np.argwhere(np.around(ndf,2) == np.around(ndf[-1],2))[0][0]
# ndf = ndf[:idx]
# q   = np.log10(quantity)[:idx]
# data_fit_params,_    = curve_fit(log_schechter_function, q, ndf, maxfev = int(1e+5),
#                                   bounds = [[0, np.amin(np.log10(quantity)), -5], [1, np.amax(np.log10(quantity)), 0]]) 
# ls = log_schechter_function(q, *data_fit_params)
# plt.close('all')
# plt.plot(q, ndf, color = 'grey')
# plt.plot(q, ls, color = 'red')