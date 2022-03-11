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

# coose option
fitting_method = 'mcmc' 
mode           = 'loading'  

# load data
groups, smfs, hmfs = load_data()

# load model
feedback_model = ['both']*5+['sn']*6
prior_model    = 'full'

# create model smfs
model = model_container(smfs, hmfs, feedback_model, fitting_method,
                        prior_model, mode).plot_parameter('C0', 'o', '-', '')
################## PLOTTING ###################################################
#%%
plt.close('all')

redshift = [0,2,4,6,8,10]
## SHMR
fig, ax = plt.subplots(1,1, sharex = True) #figsize = [8.4, 16.6]
fig.supxlabel('log $M_\mathrm{h}$ [$M_\odot$]')
fig.supylabel('log($M_*/M_\mathrm{h}$)', x=0.01)

base_unit = 1e+10
m_halo    = hmfs[0][:,0]/base_unit

cm = LinearSegmentedColormap.from_list(
        "Custom", ['C2','C1']
        , N=11)

for z in redshift:
    dist_at_z  = model.distribution.at_z(z)
    
    # draw random sample of parameter from mcmc dists
    random_draw = np.random.choice(range(dist_at_z.shape[0]),
                                   size = int(1e+6), replace = False)     
    parameter_draw = dist_at_z[random_draw]
    
    # split parameter up, to easily put into calculate_m_star function
    if model.feedback_name[z] == 'both':
        A       = parameter_draw[:,0]
        alpha   = parameter_draw[:,1]
        beta    = parameter_draw[:,2]
    if model.feedback_name[z] == 'sn':
        A       = parameter_draw[:,0]
        alpha   = parameter_draw[:,1]
    
    median = []; lower = []; upper = []
    # calculate m_star for every m_halo value and set of parameters and save
    # percentiles
    for m_h in m_halo:
         #import pdb; pdb.set_trace()
         if model.feedback_name[z] == 'both':
             m_star = model.model.at_z(z).feedback_model.calculate_log_observable(np.log10(m_h), A, alpha, beta)
         if model.feedback_name[z] == 'sn':
             m_star = model.model.at_z(z).feedback_model.calculate_log_observable(np.log10(m_h), A, alpha)
         m_star = np.power(10,m_star)
         median.append(np.percentile(m_star, 50))
         lower.append( np.percentile(m_star, 16))
         upper.append( np.percentile(m_star, 84))        
    
    ax.plot(np.log10(m_halo*base_unit), np.log10(np.array(median)/m_halo), color = cm(z),
            markevery=10, marker = 'o', label = '$z$ = ' + str(z))
    ax.fill_between(np.log10(m_halo*base_unit), np.log10(np.array(lower)/m_halo),
                    np.log10(np.array(upper)/m_halo), alpha = 0.2, color = cm(z))
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
