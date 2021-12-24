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
from matplotlib.colors import LinearSegmentedColormap

from smf_modelling   import model_container
from data_processing import load_data

import astropy.units as u
from astropy.cosmology import Planck18, z_at_value

# coose option
fitting_method = 'least_squares' 
mode           = 'temp'  

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
    for i in range(len(param_at_z)):
        #t = Planck18.lookback_time(z).value
        ax[i].scatter(z, param_at_z[i],
                      marker = model.marker, label = model.label, color = model.color)
        ax[i].set_xscale('log')
        ax[i].set_xticks(range(1,11)); ax[2].set_xticklabels(range(1,11))
        ax[i].minorticks_off()
fig.align_ylabels(ax)

## SHMR
fig, ax = plt.subplots(1,1, sharex = True) #figsize = [8.4, 16.6]
fig.supxlabel('log $M_\mathrm{h}$ [$M_\odot$]')
fig.supylabel('log($M_*/M_\mathrm{h}$)', x=0.01)

base_unit = 1e+10
m_halo    = hmfs[0][:,0]/base_unit
color = 'C2'
i = 0


cm = LinearSegmentedColormap.from_list(
        "Custom", ['C2','C1']
        , N=10)
marker = ['^','X','o','v','^','X','o','v','d','s']
for z in redshift:
    m_star = model.model.at_z(z).feedback_model.calculate_m_star(m_halo, *model.parameter.at_z(z))
    ax.plot(np.log10(m_halo*base_unit), np.log10(m_star/m_halo), alpha = 0.7, color = cm(z),
            markevery=10, marker = 'o', label = '$z$ = ' + str(z))
    ax.set_xlim([8,14])
    ax.set_ylim([-5.5,-0.8])
    
    if z==4:
        color = 'C1'
        #i += 1
ax.minorticks_on()
ax.legend()
fig.subplots_adjust(top=0.984,
bottom=0.098,
left=0.123,
right=0.969,
hspace=0.0,
wspace=0.20)

## PDFs        
# fig, ax = plt.subplots(3, 10, sharex ='row', sharey = 'row')
# ax[0,0].set_ylabel('A')
# ax[1,0].set_ylabel(r'$\alpha$')
# ax[2,0].set_ylabel(r'$\beta$')
# fig.supylabel('(Marginal) Probability Density', x = 0.01)
# for z in redshift:
#     dist_at_z = model.distribution.at_z(z)
#     bounds    = [[0,0.3], [0,3], [0,1/np.log(10)]]
#     for i in range(dist_at_z.shape[1]):
#         ax[i,z-1].hist(dist_at_z[:,i], density = True, bins = 100, range = bounds[i])
# for a in ax.flatten():
#     a.get_yaxis().set_ticks([])
# for i, a in enumerate(ax[0,:]):
#     a.set_title(r'$z=$ ' + str(i+1))
    
# ax[0,0].set_xlim(0,ax[0,0].get_xlim()[1])
# ax[0,0].set_xticks([0,0.1,0.2]); ax[0,0].set_xticklabels(['0','0.1','0.2'])
# ax[1,0].set_xlim(0,ax[1,0].get_xlim()[1])
# ax[1,0].set_xticks([0,1,2]); ax[1,0].set_xticklabels(['0','1','2'])
# ax[2,0].set_xlim(0,ax[2,0].get_xlim()[1])
# ax[2,0].set_xticks([0,0.15,0.3]); ax[2,0].set_xticklabels(['0','0.15','0.3'])
    
# plt.tight_layout()
# fig.subplots_adjust(wspace=0)
# fig.align_ylabels(ax[:, 0])