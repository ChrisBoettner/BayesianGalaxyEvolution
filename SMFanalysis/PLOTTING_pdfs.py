#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 17:14:12 2021

@author: chris
"""

from matplotlib import rc_file
rc_file('plots/settings.rc')  # <-- the file containing your settings

import numpy as np
import matplotlib.pyplot as plt

from smf_modelling   import model_container
from data_processing import load_data

# coose option
fitting_method = 'mcmc' 
mode           = 'loading'  

# load data
groups, smfs, hmfs = load_data()

# load model
prior_model = 'uniform'
# create model smfs
# no_feedback   = model_container(smfs, hmfs, 'none', fitting_method,
#                           prior_model, mode).plot_parameter(['black']*10, 'o', '-',  ['No Feedback']*10)
# sn_feedback   = model_container(smfs, hmfs, 'sn',   fitting_method,
#                           prior_model, mode).plot_parameter(['C1']*10,    's', '--', ['Stellar Feedback']*10)
# snbh_feedback = model_container(smfs, hmfs, 'both', fitting_method,
#                           prior_model, mode).plot_parameter(['C2']*10,    'v', '-.', ['Stellar + Black Hole Feedback']*10)
# models = [no_feedback,sn_feedback,snbh_feedback]

# load model
feedback_model = ['both']*4+['sn']*6
prior_model = 'full'
# create model smfs
model   = model_container(smfs, hmfs, feedback_model, fitting_method,
                          prior_model, mode).plot_parameter(['C2']*4+['C1']*6,
                                                            'o',
                                                            '-', 
                                                            ['Stellar + Black Hole Feedback']*4\
                                                            +['Stellar Feedback']*6)
models = [model]
################## PLOTTING ###################################################
#%%
plt.close('all')

redshift = np.arange(1,11)  
## PDFs        
fig, ax = plt.subplots(3, 10, sharex ='row', sharey = 'row')
ax[0,0].set_ylabel('A')
ax[1,0].set_ylabel(r'$\alpha$')
ax[2,0].set_ylabel(r'$\beta$')
fig.supxlabel('Parameter Value')
fig.supylabel('(Marginal) Probability Density', x = 0.01)
for model in models:
    for z in redshift:
        dist_at_z = model.distribution.at_z(z)
        bounds    = [[0,0.3], [0,3], [0,1/np.log(10)]]
        for i in range(dist_at_z.shape[1]):
            ax[i,z-1].hist(dist_at_z[:,i], density = True, bins = 100, range = bounds[i],
                           label = model.label[z-1], color = model.color[z-1], alpha =0.3)
for a in ax.flatten():
    a.get_yaxis().set_ticks([])
for i, a in enumerate(ax[0,:]):
    a.set_title(r'$z=$ ' + str(i+1))

print("CORRECT PLOT LIMITS")    
ax[0,0].set_xlim(0,0.7)
ax[0,0].set_xticks([0,0.1,0.2]); ax[0,0].set_xticklabels(['0','0.1','0.2'])
ax[1,0].set_xlim(0,7)
ax[1,0].set_xticks([0,1,2]); ax[1,0].set_xticklabels(['0','1','2'])
ax[2,0].set_xlim(0,0.7)
ax[2,0].set_xticks([0,0.1,0.2]); ax[2,0].set_xticklabels(['0','0.1','0.2'])

#ax[0,0].set_ylim(0,ax[0,0].get_ylim()[1]/2)    

# turn off empty axes
if len(models)==1:
    [ax[2,i].axis('off') for i in range(4,10)]

# legend
handles, labels = [], []
for a in ax.flatten():
    handles_, labels_ = a.get_legend_handles_labels()
    handles += handles_
    labels += labels_
by_label = dict(zip(labels, handles))
ax[0,-1].legend(list(by_label.values())[:3], list(by_label.keys())[:3], frameon=False,
             prop={'size': 12}, loc='upper right', bbox_to_anchor=(1.0, 1.3),
             ncol=3, fancybox=True)

fig.align_ylabels(ax[:, 0])

fig.subplots_adjust(top=0.9,
bottom=0.11,
left=0.055,
right=0.99,
hspace=0.2,
wspace=0.0)
