#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 12:10:37 2021

@author: boettner
"""

from matplotlib import rc_file
rc_file('plots/settings.rc')  # <-- the file containing your settings

import numpy as np
import matplotlib.pyplot as plt

from smf_modelling import fit_SMF_model
from data_processing import group, z_ordered_data

SOMETHING IS GOING WRONG I THINK (WITH CONSTANT A). CHECK NOTEBOOK

################## CHOOSE FITTING METHOD ######################################
fitting_method = 'least_squares'    
mode           = 'loading'          # 'saving', 'loading' or 'temp'

################## LOAD DATA ##################################################
# get z=1,2,3,4 for Davidson, z=1,2,3 for Ilbert
davidson    = np.load('data/Davidson2017SMF.npz'); davidson = {i:davidson[j] for i,j in [['0','2'],['1','4'],['2','6'],['3','8']]}
ilbert      = np.load('data/Ilbert2013SMF.npz');   ilbert = {i:ilbert[j] for i,j in [['0','2'],['1','4'],['2','6']]}
duncan      = np.load('data/Duncan2014SMF.npz')      
song        = np.load('data/Song2016SMF.npz')       
bhatawdekar = np.load('data/Bhatawdekar2018SMF.npz')
stefanon    = np.load('data/Stefanon2021SMF.npz')
hmfs        = np.load('data/HMF.npz'); hmfs = [hmfs[str(i)] for i in range(20)]   

## TURN DATA INTO GROUP OBJECTS, INCLUDING PLOT PARAMETER
davidson    = group(davidson,    [1,2,3,4]   ).plot_parameter('black', 'o', 'Davidson2017')
ilbert      = group(ilbert,      [1,2,3]     ).plot_parameter('black', 'H', 'Ilbert2013')
duncan      = group(duncan,      [4,5,6,7]   ).plot_parameter('black', 'v', 'Duncan2014')
song        = group(song,        [6,7,8]     ).plot_parameter('black', 's', 'Song2016')
bhatawdekar = group(bhatawdekar, [6,7,8,9]   ).plot_parameter('black', '^', 'Bhatawdekar2019')
stefanon    = group(stefanon,    [6,7,8,9,10]).plot_parameter('black', 'X', 'Stefanon2021')
groups      = [davidson, ilbert, duncan, song, bhatawdekar, stefanon]

## DATA SORTED BY REDSHIFT
smfs = z_ordered_data(groups)
# undo log for easier fitting
raise10 = lambda list_log: [10**list_log[i] for i in range(len(list_log))]
smfs = raise10(smfs)
hmfs = raise10(hmfs)

################## CALCULATE SHMR #############################################
# model classes with estimated smfs and plotting information
class smf_model():
    def __init__(self, smfs, hmfs, feedback_name, fitting_method, mode):
        parameter, modelled_smf, cost = fit_SMF_model(smfs, hmfs, feedback_name,
                                                      fitting_method, mode)
        self.parameter = smf_object(parameter)
        self.smf       = smf_object(modelled_smf)
        self.cost      = smf_object(cost)
        
        self.feedback_name = feedback_name
    def plot_parameter(self, color, marker, linestyle, label):
        self.color     = color
        self.marker    = marker
        self.linestyle = linestyle
        self.label     = label
        return(self)
    
class smf_object():
    def __init__(self, data):
        self.data = data
    def at_z(self, redshift):
        if redshift == 0:
            raise ValueError('Redshift 0 not in data')
        return(self.data[redshift-1])

## CREATE MODEL OBJECTS
no_feedback   = smf_model(smfs, hmfs, 'none',
                          fitting_method, mode).plot_parameter('black', 'o', '-',  'No Feedback')
sn_feedback   = smf_model(smfs, hmfs, 'sn',
                          fitting_method, mode).plot_parameter('C1',    's', '--', 'Stellar Feedback')
snbh_feedback = smf_model(smfs, hmfs, 'both',
                          fitting_method, mode).plot_parameter('C2',    'v', '-.', 'Stellar + Black Hole Feedback')
models = [no_feedback, sn_feedback, snbh_feedback]

################## PLOTTING ###################################################
plt.close('all')

## STELLAR MASS FUNCTION
redshift = np.arange(1,11)  
fig, ax = plt.subplots(4,3, sharex=True, sharey=True); ax = ax.flatten()
fig.subplots_adjust(top=0.982,bottom=0.113,left=0.075,right=0.991,hspace=0.0,wspace=0.0)
# plot group data
for g in groups:
    for z in g.redshift:
        ax[z-1].errorbar(g.data_at_z(z).mass, g.data_at_z(z).phi, [g.data_at_z(z).lower_error,g.data_at_z(z).upper_error],
                         capsize = 3, fmt = g.marker, color = g.color, label = g.label, alpha = 0.4)
        ax[z-1].set_xlim([6.67,12.3])
        ax[z-1].set_ylim([-6,3])  
# plot modelled smf
for model in models:
    for z in redshift:
        ax[z-1].plot(np.log10(model.smf.at_z(z)[:,0]), np.log10(model.smf.at_z(z)[:,1]),
                  linestyle=model.linestyle, label = model.label, color = model.color)      
# fluff
fig.supxlabel('log[$M_*/M_\odot$]')
fig.supylabel('log[$\phi(M_*)$ cMpc$^3$ dex]', x=0.01)
for i, a in enumerate(ax):
    a.minorticks_on()
    if i<len(redshift):
        a.text(0.97, 0.94, str(i+0.5) + '$<$z$<$' +str(i+1.5), size = 11,
               horizontalalignment='right', verticalalignment='top', transform=a.transAxes)
# legend
handles, labels = [], []
for a in ax:
    handles_, labels_ = a.get_legend_handles_labels()
    handles += handles_
    labels += labels_
by_label = dict(zip(labels, handles))
ax[0].legend( list(by_label.values())[:3], list(by_label.keys())[:3], frameon=False,
             prop={'size': 12}, loc = 3)
ax[-1].legend(list(by_label.values())[3:], list(by_label.keys())[3:], frameon=False,
              prop={'size': 12})

## PARAMETER EVOLUTION      
fig, ax = plt.subplots(3,1, sharex=True)
ax[0].set_ylabel('A')
ax[1].set_ylabel(r'$\alpha$')
ax[2].set_ylabel(r'$\beta$')
fig.supxlabel('$z$')
for model in models:
    parameter_number = len(model.parameter.data[0])
    for i in range(parameter_number):
        param_at_z = [model.parameter.at_z(z)[i] for z in redshift]
        ax[i].scatter(redshift, param_at_z,
                      marker = model.marker, label = model.label, color = model.color)
ax[0].legend()
fig.align_ylabels(ax)
fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)
for a in ax:
    a.minorticks_on()
    a.tick_params(axis='x', which='minor', bottom=False)
#exclude out of bounds data points
#ax[2].set_ylim([-0.012,0.5])
#ax[2].arrow(8, 0.44, 0, 0.03,
#          head_width=0.04, head_length=0.02, color = snbh_feedback.color)
#ax[2].arrow(10, 0.44, 0, 0.03,
#          head_width=0.04, head_length=0.02, color = snbh_feedback.color)
