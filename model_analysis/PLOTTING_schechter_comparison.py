#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:33:27 2022

@author: chris
"""

from matplotlib import rc_file
rc_file('plots/settings.rc')  # <-- the file containing your settings

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from modelling import model, log_schechter_function, lum_to_mag
from data_processing import load_best_fit_parameter

from data.data_processing_smf  import load_data as load_data_smf
from data.data_processing_uvlf import load_data as load_data_uvlf

################## LOAD DATA AND MODEL ########################################
#%%
quantity_name = 'mstar'

redshift = range(11)
if quantity_name == 'mstar':
    quantity   = np.logspace(6,12,100)
    groups, data, _ = load_data_smf()
if quantity_name == 'lum':
    quantity   = np.logspace(24,30,100)
    groups, data, _ = load_data_uvlf() 

log_data = [np.log10(d) for d in data]
log_data = [l[l[:,1]>-6] for l in log_data] #remove datapoints that have number density of <10^-6

mod = model(None) # dummy model just to call number density function
################## SCHECHTER FITS #############################################
#%%
## Best Fit Schechter parameter from data directly
schechter_data = []; schechter_p_data = []
for z in redshift:
    schechter_params,_ = curve_fit(log_schechter_function, log_data[z][:,0], log_data[z][:,1],
                                   maxfev = int(1e+5)) 
    schechter_p_data.append(schechter_params)
    schechter_data.append(log_schechter_function(np.log10(quantity), *schechter_params))

## Best Fit Schechter parameter fit using successive prior
# Calculate Model
def calculate_schechter(quantity_name, prior_model, p0 = None):
    '''
    Calculate Schechter parameter by first loading pre-calculated best fit model
    parameter, then calculating the model function and then fitting a Schechter
    function to this model function.
    If p0 is given, should be same length as redshift.
    
    IMPORTANT: model values with phi<1e-6 are cutoff in accordance with data
               fitting procedure.
    '''
    params = load_best_fit_parameter(quantity_name, prior_model)
    model_func, schechter_func, schechter_params = [], [], []
    for z in redshift:
        # load parameter
        p = params[z]
        if len(p) == 2: # beta = 0, if beta not in model
           p =  np.append(p,0)
        # calculate model number density function
        ndf = mod.number_density_function(quantity, z, *p)[0]
        model_func.append(ndf)
        
        # cut end of ndf, either because values are repeating due to end of HMF data
        # or if number densities become smaller than 1e-6 cutoff also used in fitting
        idx_rep = np.argwhere(np.around(ndf/ndf[-1],4) == 1)[0][0] # index of first occurence of repeated value
        if np.any(ndf<1e-6):
            idx_min = np.argwhere(ndf<1e-6)[0][0]                  # index where density becomes very small
        else:
            idx_min = len(ndf)-1                                   # last index if it never becomes that small
        idx     = np.amin([idx_rep,idx_min]) # choose whatever comes first
        ndf = ndf[:idx]
        q   = quantity[:idx] # cut input variable to same length for fitting
        #import pdb; pdb.set_trace()
        # fit schechter function
        schechter_p,_ = curve_fit(log_schechter_function, np.log10(q), np.log10(ndf),
                                  maxfev = int(1e+5), p0 = p0[z])
        schechter_params.append(schechter_p)
        schechter_func.append(log_schechter_function(np.log10(quantity), *schechter_p))
    return(model_func, schechter_func, schechter_params)

model_full, schechter_full, schechter_p_full = calculate_schechter(quantity_name, 'full', p0 = schechter_p_data)
model_uni, schechter_uni, schechter_p_uni    = calculate_schechter(quantity_name, 'uniform', p0 = schechter_p_data)

################## PLOTTING ###################################################
#%%
plt.close('all')

plot_model     = [2.75, 0.5, 'grey'] # [linewidth, alpha, color]
plot_schechter = [1.75, 1, 'C2']

fig, ax = plt.subplots(4,3, sharey = 'row', sharex=True); ax = ax.flatten()
fig.subplots_adjust(top=0.982,bottom=0.113,left=0.075,right=0.991,hspace=0.0,wspace=0.0)

if quantity_name == 'mstar':
    x = np.log10(quantity)
elif quantity_name == 'lum':
    x = lum_to_mag(quantity)
# plot Schechter and model functions
for z in redshift:
        ax[z].plot(x, np.log10(model_uni[z]), linewidth = plot_model[0],
                    alpha = plot_model[1], color = 'C1', 
                    linestyle = '--',
                    label = 'Model (Uniform Prior)') 
        ax[z].plot(x, np.log10(model_full[z]), linewidth = plot_model[0],
                    alpha = plot_model[1] , color = 'C2', 
                    linestyle = '-.',
                    label = 'Model (Successive Prior)') 
        
        ax[z].plot(x, schechter_data[z], linewidth = plot_schechter[0],
                    alpha = plot_schechter[1] , color = 'grey', 
                    linestyle = '-',
                    label = 'Schechter Function (Data)') 
        ax[z].plot(x, schechter_uni[z], linewidth = plot_schechter[0],
                    alpha = plot_schechter[1] , color = 'C1', 
                    linestyle = '--',
                    label = 'Schechter Function (Model - Uniform Prior)') 
        ax[z].plot(x, schechter_full[z], linewidth = plot_schechter[0],
                    alpha = plot_schechter[1] , color = 'C2', 
                    linestyle = '-.',
                    label = 'Schechter Function (Model - Successive Prior)') 
# plot group data
for g in groups:
    for z in g.redshift:
        if quantity_name == 'mstar':
            x = g.data_at_z(z).mass
        elif quantity_name == 'lum':
            x =  g.data_at_z(z).mag
        ax[z].errorbar(x, g.data_at_z(z).phi, [g.data_at_z(z).lower_error,g.data_at_z(z).upper_error],
                         capsize = 3, fmt = g.marker, color = g.color, label = g.label, alpha = 0.4)
        #ax[z].set_xlim([6.67,12.3])
        if z<3:
            ax[z].set_ylim([-5,3])
        else:
            ax[z].set_ylim([-6,3]) 

if quantity_name == 'mstar':
    fig.supxlabel('log $M_*$ [$M_\odot$]')
    fig.supylabel('log $\phi(M_*)$ [cMpc$^{-3}$ dex$^{-1}$]', x=0.01) 
elif quantity_name == 'lum':
    fig.supxlabel(r'$M_{UV}$')
    fig.supylabel('log $\phi(M_\\nu^{UV})$ [cMpc$^{-3}$ dex$^{-1}$]', x=0.01)    

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
ax[0].legend( list(by_label.values())[:2][::-1], list(by_label.keys())[:2][::-1],
             frameon=False, prop={'size': 12}, loc = 2)
ax[1].legend( list(by_label.values())[2:5][::-1], list(by_label.keys())[2:5][::-1],
             frameon=False, prop={'size': 12}, loc = 2)
ax[-1].legend(list(by_label.values())[5:], list(by_label.keys())[5:],
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