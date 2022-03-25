#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 12:09:07 2022

@author: chris
"""
from matplotlib import rc_file
rc_file('plots/settings.rc')  # <-- the file containing your settings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from modelling import model, lum_to_sfr, calculate_observable, calculate_halo_mass

################## LOAD MODELS ################################################
#%%
print('Loading models..')
mstar_model = model('mstar')
lum_model   = model('lum')
print('Loading done')

################## CALCULATION ################################################
#%%
# Calculate relation between stellar mass and SFR in following fashion:
# First create a distribution of halo masses from input stellar mass and
# stellar mass model (for every redshift). Then calculate for each of these halo
# mass estimates a distribution of luminosities based on the luminosity model

# choose how to treat beta parameter in stellar feedback-only models
beta = 'zero'

# define stellar mass space and redshift range
num_m_star = 100
m_star     = np.logspace(8,11,num_m_star)
redshift = [0,2,4,6,8,10]
# choose sample size that is drawn from distributions
num_m_h = 200
num_lum = 1000

# calculate distribution of halo masses from stellar masses and model
m_h_dist = calculate_halo_mass(mstar_model, m_star, redshift,
                               num = num_m_h, beta = beta)

# use each element of halo mass distribution to calculate a lumiosity distribution
# and then combine all of those as final distribution that relates stellar mass
# to luminosity
lum_dist = []
for i,z in enumerate(redshift):
    lum_dist_at_z = np.empty([num_m_star, num_m_h*num_lum])
    for j in range(num_m_star):
            halo_masses = m_h_dist[i][j,:]
            lum_dist_at_z[j,:] = calculate_observable(lum_model, halo_masses, z, 
                                                      num = num_lum, beta = beta)[0].flatten()
    lum_dist.append(np.array(lum_dist_at_z))

# calculate percentiles and calculate SFR from luminosities
sfr_median = [lum_to_sfr(np.nanpercentile(l, 50, axis = 1)) for l in lum_dist]
sfr_lower  = [lum_to_sfr(np.nanpercentile(l, 16, axis = 1)) for l in lum_dist]
sfr_upper  = [lum_to_sfr(np.nanpercentile(l, 84, axis = 1)) for l in lum_dist]    
################## PLOTTING ###################################################
#%%    
plt.close('all')
## M* vs SFR
fig, ax = plt.subplots(1,1)
fig.supxlabel('log $M_\mathrm{*}$ [$M_\odot$]')
fig.supylabel('log SFR [$M_\odot$ yr$^{-1}$])', x=0.01)
cm = LinearSegmentedColormap.from_list(
        "Custom", ['C2','C1']
        , N=11)

for i, z in enumerate(redshift):
    ax.plot(np.log10(m_star), np.log10(sfr_median[i]), color = cm(z),
            markevery=10, marker = 'o', label = '$z$ = ' + str(z))
    ax.fill_between(np.log10(m_star), np.log10(sfr_lower[i]),
                    np.log10(sfr_upper[i]), alpha = 0.2, color = cm(z))
    #ax.set_xlim([8,14])
    #ax.set_ylim([-5.5,0])
    
ax.minorticks_on()
ax.legend()
fig.subplots_adjust(
top=0.92,
bottom=0.09,
left=0.06,
right=0.99,
hspace=0.0,
wspace=0.0)

## M* vs sSFR
fig, ax = plt.subplots(1,1)
fig.supxlabel('log $M_\mathrm{*}$ [$M_\odot$]')
fig.supylabel('log sSFR [yr$^{-1}$])', x=0.01)
cm = LinearSegmentedColormap.from_list(
        "Custom", ['C2','C1']
        , N=11)

for i, z in enumerate(redshift):
    ax.plot(np.log10(m_star), np.log10(sfr_median[i]/m_star), color = cm(z),
            markevery=10, marker = 'o', label = '$z$ = ' + str(z))
    ax.fill_between(np.log10(m_star), np.log10(sfr_lower[i]/m_star),
                    np.log10(sfr_upper[i]/m_star), alpha = 0.2, color = cm(z))
    #ax.set_xlim([8,14])
    #ax.set_ylim([-5.5,0])
    
ax.minorticks_on()
ax.legend()
fig.subplots_adjust(
top=0.92,
bottom=0.09,
left=0.06,
right=0.99,
hspace=0.0,
wspace=0.0)
