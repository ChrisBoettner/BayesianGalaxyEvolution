#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 15:48:37 2021

@author: boettner
"""


from matplotlib import rc_file
rc_file('plots/settings.rc')  # <-- the file containing your settings

import numpy as np
import matplotlib.pyplot as plt

from data_processing import group, z_ordered_data, mag_to_lum
from uvlf import calculate_LHMR

################## LOAD DATA ##################################################
# get z=1,2,3,4 for Madau
madau       = np.load('data/Madau2014UVLF.npz')
madau = {i:madau[j] for i,j in [['0','1'],['1','4'],['2','7'],['3','8'], ['4','9']]}
duncan      = np.load('data/Duncan2014UVLF.npz')      
bouwens     = np.load('data/Bouwens2015UVLF.npz')       
bouwens2    = np.load('data/Bouwens2021UVLF.npz')
oesch       = np.load('data/Oesch2018UVLF.npz')
parsa       = np.load('data/Parsa2016UVLF.npz')
bhatawdekar = np.load('data/Bhatawdekar2019UVLF.npz')
atek        = np.load('data/Atek2018UVLF.npz')
livermore   = np.load('data/Livermore2017UVLF.npz')
hmfs        = np.load('data/HMF.npz'); hmfs = [hmfs[str(i)] for i in range(20)]   

## TURN DATA INTO GROUP OBJECTS, INCLUDING PLOT PARAMETER
madau       = group(madau,    range(0,5) ).plot_parameter('black', 'o', 'Cucciati2012')
duncan      = group(duncan,      range(4,8) ).plot_parameter('black', 'v', 'Duncan2014')
bouwens     = group(bouwens,     range(4,9) ).plot_parameter('black', 's', 'Bouwens2015')
bouwens2    = group(bouwens2,    range(2,11)).plot_parameter('black', '^', 'Bouwens2021')
oesch       = group(oesch,       range(1,3) ).plot_parameter('black', 'X', 'Oesch2018')
atek        = group(atek,        range(6,7) ).plot_parameter('black', 'o', 'Atek2018')
bhatawdekar = group(bhatawdekar, range(6,10)).plot_parameter('black', '<', 'Bhatawdekar2019')
parsa       = group(parsa,       range(2,5) ).plot_parameter('black', '>', 'Parsa2016')
livermore   = group(livermore,   range(6,9) ).plot_parameter('black', 'H', 'Livermore2017')
groups      = [madau, duncan, bouwens, bouwens2, oesch, atek, bhatawdekar, parsa, livermore]

## DATA SORTED BY REDSHIFT
lfs = z_ordered_data(groups)
# undo log for easier fitting, turn magnitudes into luminosities
raise10 = lambda list_log: [10**list_log[i] for i in range(len(list_log))]
def lfs_conversion(lfs):
    for lf in lfs:
        lf[:,0] = mag_to_lum(lf[:,0])
        lf[:,1] = raise10(lf[:,1])
    return(lfs)
lfs  = lfs_conversion(lfs)
hmfs = raise10(hmfs)

################## CALCULATE SHMR #############################################
# model classes with estimated smfs and plotting information
class lf_model():
    def __init__(self, lfs, hmfs, feedback):
        parameter, sfe, modelled_lf, sfrf, ssr = calculate_LHMR(lfs, hmfs, feedback)
        self.parameter = lf_object(parameter)
        self.sfe       = lf_object(sfe)
        self.lf        = lf_object(modelled_lf)
        self.sfrf      = lf_object(sfrf)
        self.ssr       = lf_object(ssr)      
    def plot_parameter(self, color, marker, linestyle, label):
        self.color     = color
        self.marker    = marker
        self.linestyle = linestyle
        self.label     = label
        return(self)
    
class lf_object():
    def __init__(self, data):
        self.data = data
    def at_z(self, redshift):
        return(self.data[redshift-1])
    
## CREATE MODEL OBJECTS
no_feedback    = lf_model(lfs, hmfs, 'none' ).plot_parameter('black', 'o', '-',  'No Feedback')
sn_feedback    = lf_model(lfs, hmfs, 'sn'  ).plot_parameter('C1',    's', '--', 'Stellar Feedback')
snbh_feedback  = lf_model(lfs, hmfs, 'both').plot_parameter('C2',    'v', '-.', 'Stellar + Black Hole Feedback')
models = [no_feedback, sn_feedback, snbh_feedback]
################## PLOTTING ###################################################
plt.close('all')

## LUMINOSITY FUNCTION
redshift = np.arange(0,11)  
fig, ax = plt.subplots(4,3, sharex=True, sharey=True); ax = ax.flatten()
fig.subplots_adjust(top=0.982,bottom=0.113,left=0.075,right=0.991,hspace=0.0,wspace=0.0)
# plot group data
for g in groups:
    for z in g.redshift:
        ax[z].errorbar(np.log10(g.data_at_z(z).lum), g.data_at_z(z).phi, [g.data_at_z(z).lower_error,g.data_at_z(z).upper_error],
                         capsize = 3, fmt = g.marker, color = g.color, label = g.label, alpha = 0.4)
        #ax[z-1].set_xlim([2.5,14.9])
        ax[z].set_ylim([-7,3])        
# plot modelled lf
#for model in models:
#    for z in redshift:
#        ax[z-1].plot(np.log10(model.lf.at_z(z)[:,0]), np.log10(model.lf.at_z(z)[:,1]),
#                  linestyle=model.linestyle, label = model.label, color = model.color)  
# fluff
fig.supxlabel('log[$L_{\\nu}^{UV}$ ergs$^{-1}$ s Hz]')
fig.supylabel('log[$\phi(L_\\nu^{UV})$ cMpc$^{3}$ dex]', x=0.01)
for i, a in enumerate(ax):
    a.minorticks_on()
    if i<len(redshift):
        a.text(0.97, 0.94, str(i-0.5) + '$<$z$<$' +str(i+0.5), size = 11,
               horizontalalignment='right', verticalalignment='top', transform=a.transAxes)
# legend
handles, labels = [], []
for a in ax:
    handles_, labels_ = a.get_legend_handles_labels()
    handles += handles_
    labels += labels_
by_label = dict(zip(labels, handles))
#ax[0].legend( list(by_label.values())[:3], list(by_label.keys())[:3], frameon=False,
#             prop={'size': 12}, loc = 3)
ax[-1].legend(list(by_label.values())[:], list(by_label.keys())[:], frameon=False,
              prop={'size': 12}, ncol=2)

## STAR FORMATION EFFICIENCY      
# fig, ax = plt.subplots(1,1, sharex=True)
# ax.set_ylabel('log[SFE/yr]')
# fig.supxlabel('log[$z$]')
# for model in models:
#     ax.scatter(np.log(redshift), np.log10(model.sfe.data),
#                   marker = model.marker, label = model.label, color = model.color)
# ax.legend()
# fig.align_ylabels(ax)
# fig.tight_layout()
# fig.subplots_adjust(hspace=0, wspace=0)
# ax.minorticks_on()
# ax.tick_params(axis='x', which='minor', bottom=False)