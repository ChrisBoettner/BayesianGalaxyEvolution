#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 13:59:54 2022

@author: chris
"""

from matplotlib import rc_file
rc_file('plots/settings.rc')  # <-- the file containing your settings

import numpy as np
import matplotlib.pyplot as plt

from uvlf_modelling  import model_container
from data_processing import load_data

# coose option
fitting_method = 'mcmc' 
mode           = 'loading'  

# load data
groups, lfs, hmfs = load_data()

prior_model = 'full'

# load model
if prior_model == 'uniform':
    no_feedback   = model_container(lfs, hmfs, 'none', fitting_method,
                              prior_model, mode).plot_parameter(['black']*11, 'o', '-',  ['No Feedback']*11)
    sn_feedback   = model_container(lfs, hmfs, 'sn',   fitting_method,
                              prior_model, mode).plot_parameter(['C1']*11,    's', '--', ['Stellar Feedback']*11)
    snbh_feedback = model_container(lfs, hmfs, 'both', fitting_method,
                              prior_model, mode).plot_parameter(['C2']*11,    'v', '-.', ['Stellar + Black Hole Feedback']*11) 
    models = [no_feedback,sn_feedback,snbh_feedback]
if prior_model == 'full':
    feedback_model = ['both']*5+['sn']*6
    model   = model_container(lfs, hmfs, feedback_model, fitting_method,
                              prior_model, mode).plot_parameter(['C2']*5+['C1']*6,
                                                                'o',
                                                                '-', 
                                                                ['Stellar + Black Hole Feedback']*5\
                                                                +['Stellar Feedback']*6)
    models = [model]
################## PLOTTING ###################################################
#%%
plt.close('all')

redshift = np.arange(0,11)
## PDFs        
fig, ax = plt.subplots(3, 11, sharex ='row', sharey = 'row')
ax[0,0].set_ylabel('$A/10^{18}$\n[ergs s$^{-1}$ Hz$^{-1}$ $M_\odot^{-1}$]',
                   multialignment='center')
ax[1,0].set_ylabel(r'$\alpha$')
ax[2,0].set_ylabel(r'$\beta$')
fig.supxlabel('Parameter Value')
fig.supylabel('(Conditional) Probability Density', x = 0.01)
for model in models:
    for z in redshift:
        #if (model.model.at_z(z).feedback_model.name == 'both') and (z>4):
        #    continue
        param_at_z   = model.parameter.at_z(z)
        dist_at_z    = model.distribution.at_z(z)
        bounds       = np.array(model.model.at_z(z).feedback_model.bounds).T
        bin_widths   = (bounds[:,1]-bounds[:,0])/100
        cond_lims = np.array([param_at_z-5*bin_widths,param_at_z+5*bin_widths]).T
        for i in range(dist_at_z.shape[1]):
            mask = np.copy(dist_at_z).astype(bool)
            cond_var = list(range(dist_at_z.shape[1])); cond_var.remove(i)
            for j in cond_var:
                mask[:,j][dist_at_z[:,j] < cond_lims[j,0]] = False
                mask[:,j][dist_at_z[:,j] > cond_lims[j,1]] = False
            mask = np.prod(mask,axis=1).astype(bool)
            ax[i,z].hist(dist_at_z[:,i][mask], density = True, bins = 100, range = bounds[i],
                           label = model.label[z], color = model.color[z], alpha =0.3)
for a in ax.flatten():
    a.get_yaxis().set_ticks([])
for i, a in enumerate(ax[0,:]):
    a.set_title(r'$z=$ ' + str(i))


# legend
handles, labels = [], []
for a in ax.flatten():
    handles_, labels_ = a.get_legend_handles_labels()
    handles += handles_
    labels += labels_
by_label = dict(zip(labels, handles))

# setting based on prior model
if prior_model == 'uniform':
    ax[0,0].set_ylim(0,ax[0,0].get_ylim()[1]/5)
    ax[0,0].set_xlim(0,2)
    ax[0,0].set_xticks([0,1]); ax[0,0].set_xticklabels(['0','1'])
    ax[1,0].set_xlim(0,3)
    ax[1,0].set_xticks([0,1,2]); ax[1,0].set_xticklabels(['0','1','2'])
    ax[2,0].set_xlim(0,0.8)
    ax[2,0].set_xticks([0,0.5]); ax[2,0].set_xticklabels(['0','0.5'])
    ax[0,-1].legend(list(by_label.values())[:3][::-1], list(by_label.keys())[:3][::-1], frameon=False,
                 prop={'size': 12}, loc='upper right', bbox_to_anchor=(1.0, 1.3),
                 ncol=3, fancybox=True)
if prior_model == 'full':
    # turn off empty axes
    [ax[2,i].axis('off') for i in range(5,11)]
    ax[0,0].set_xlim(0,2)
    ax[0,0].set_xticks([0,1]); ax[0,0].set_xticklabels(['0','1'])
    ax[1,0].set_xlim(0,3)
    ax[1,0].set_xticks([0,1,2]); ax[1,0].set_xticklabels(['0','1','2'])
    ax[2,0].set_xlim(0,0.8)
    ax[2,0].set_xticks([0.1,0.5]); ax[2,0].set_xticklabels(['0.1','0.5'])
    ax[0,-1].legend(list(by_label.values())[:3], list(by_label.keys())[:3], frameon=False,
                 prop={'size': 12}, loc='upper right', bbox_to_anchor=(1.0, 1.3),
                 ncol=3, fancybox=True)

fig.align_ylabels(ax[:, 0])

fig.subplots_adjust(
top=0.9,
bottom=0.11,
left=0.075,
right=0.99,
hspace=0.2,
wspace=0.0)
