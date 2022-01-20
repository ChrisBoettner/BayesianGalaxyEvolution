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

redshift = [1,3,5,7,9]
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
    param_at_z = model.parameter.at_z(z)
    dist_at_z  = model.distribution.at_z(z)
    # lower and upper limit on parameter (error bars)
    lower      = np.percentile(dist_at_z, 16, axis = 0)
    upper      = np.percentile(dist_at_z, 84, axis = 0)
    # switch A values since it has the opposite effect to the slopes
    temp = lower[0]; lower[0] = upper[0]; upper[0] = temp
    
    
    m_star = model.model.at_z(z).feedback_model.calculate_m_star(m_halo, *model.parameter.at_z(z))
    m_star_l = model.model.at_z(z).feedback_model.calculate_m_star(m_halo, *upper)
    m_star_u = model.model.at_z(z).feedback_model.calculate_m_star(m_halo, *lower)
    
    ax.plot(np.log10(m_halo*base_unit), np.log10(m_star/m_halo), color = cm(z),
            markevery=10, marker = 'o', label = '$z$ = ' + str(z))
    ax.fill_between(np.log10(m_halo*base_unit), np.log10(m_star_u/m_halo), np.log10(m_star_l/m_halo),
                    alpha = 0.2, color = cm(z))
    ax.set_xlim([8,14])
    ax.set_ylim([-5.5,0])
    
    if z==4:
        color = 'C1'
        #i += 1
ax.minorticks_on()
ax.legend()
fig.subplots_adjust(
top=0.92,
bottom=0.09,
left=0.06,
right=0.99,
hspace=0.0,
wspace=0.0)
