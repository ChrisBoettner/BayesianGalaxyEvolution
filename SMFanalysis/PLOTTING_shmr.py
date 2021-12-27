#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 17:39:23 2021

@author: chris
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
