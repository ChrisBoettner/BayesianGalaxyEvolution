#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 16:58:11 2021

@author: boettner
"""

from matplotlib import rc_file
rc_file('plots/settings.rc')  # <-- the file containing your settings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## MAIN FUNCTION
## calc scale parameters for given dataset
def calc_scaling(dataset, plot = False):
    smf_data, hmf_data, smf_vals,hmf_vals, cutoff = load_data(dataset)
    
    scale = []
    for i in range(len(smf_vals)):
        smf = smf_data[smf_vals[i]]; smf[:,1] = np.power(10, smf[:,1])
        hmf = hmf_data[hmf_vals[i]]; hmf[:,1] = np.power(10, hmf[:,1])
        
        smf = smf[np.where(smf[:,0]<cutoff),:][0,:,:]
        
        scale.append(scaling_fit(find_closest(smf,hmf)))
    scale = np.array(scale) 
    
    if plot == True:    
        fig, ax = plt.subplots(3, 2); ax = ax.flatten()
        for i in range(len(smf_vals)):
              smf = smf_data[smf_vals[i]]; smf[:,1] = np.power(10, smf[:,1])
              hmf = hmf_data[hmf_vals[i]]; hmf[:,1] = np.power(10, hmf[:,1])
              
              smf = smf[np.where(smf[:,0]<cutoff),:][0,:,:]
          
              ax[i].scatter(hmf[:,0], hmf[:,1])            
              ax[i].scatter(hmf[:,0], scale[i]*hmf[:,1]) 
    
              ax[i].scatter(smf[:,0],smf[:,1])
              
              ax[i].set_yscale('log')
              #ax[i].set_xlim([8,12])
              #ax[i].set_ylim([-7,2])
    return(scale)

## SUBFUNCTIONS
#  load data and params
def load_data(dataset):
    if dataset == 'Duncan':
        smf_data   = np.load('data/Duncan2014SMF.npz')
        cutoff     = 10.5
        smf_vals = np.arange(0,4).astype(str); hmf_vals= np.arange(4,8).astype(str) #get 4 to 7
    if dataset == 'Ilbert':
        smf_data   = np.load('data/Ilbert2013SMF.npz')
        cutoff     = 10.5
        smf_vals = np.array([2,5]).astype(str); hmf_vals= np.arange(1,3).astype(str) #get 1 and 2
    if dataset == 'Davidson':
        smf_data   = np.load('data/Davidson2017SMF.npz')   
        cutoff     = 10.5
        smf_vals = np.array([2,5,7,8]).astype(str); hmf_vals= np.arange(1,6).astype(str) #get redshifts 1 to 4
    if dataset == 'Bhatawdekar':
        smf_data   = np.load('data/Bhatawdekar2018SMF.npz')
        cutoff     = 10.5
        smf_vals = np.arange(0,4).astype(str); hmf_vals= np.arange(6,10).astype(str) #get redshifts 6 to 9
    if dataset == 'Song':
        smf_data   = np.load('data/Song2016SMF.npz')   
        cutoff     = 10.5
        smf_vals = np.arange(0,3).astype(str); hmf_vals= np.arange(6,9).astype(str) #get redshifts 6 to 8
    hmf_data   = np.load('data/HMF.npz')
    return(smf_data, hmf_data, smf_vals, hmf_vals, cutoff)

# find phi values for clostest matching m values
def find_closest(smf,hmf):
    vals        = np.empty([len(smf),2])
    vals[:,0]   = smf[:,1]
    for i in range(len(smf)):
        diff = np.abs(hmf[:,0]- smf[i,0]) 
        vals[i,1] = hmf[np.argmin(diff),1]
    return(vals)

# calculate best fit scaling parameter between matching datapoints
def scaling_fit(vals):
    return(np.linalg.lstsq(vals[:,1,np.newaxis],vals[:,0],rcond=None)[0][0])

###############################################################################

#Ilbert_scaling      = calc_scaling('Ilbert');       Ilbert_z      = np.arange(1,3)
Davidson_scaling    = calc_scaling('Davidson');     Davidson_z    = np.arange(1,5)
Duncan_scaling      = calc_scaling('Duncan');       Duncan_z      = np.arange(4,8) 
Song_scaling        = calc_scaling('Song');         Song_z        = np.arange(6,9)
Bhatawdekar_scaling = calc_scaling('Bhatawdekar');  Bhatawdekar_z = np.arange(6,10)   

plt.close('all')
plt.figure()
#plt.scatter(Ilbert_z,      Ilbert_scaling,      label = 'Ilbert',       marker='.')
plt.scatter(Davidson_z,    Davidson_scaling,    label = 'Davidson',     marker='*')
plt.scatter(Duncan_z,      Duncan_scaling,      label = 'Duncan',       marker='>')  # /2
plt.scatter(Song_z,        Song_scaling,        label = 'Song',         marker='<')  # *2
plt.scatter(Bhatawdekar_z, Bhatawdekar_scaling, label = 'Bhatawdekar',  marker='x')  # /2

#plt.yscale('log')
#plt.xscale('log')

plt.xlabel('z')
plt.ylabel('scaling factor')

plt.legend()

####
x = np.concatenate([Ilbert_z, Davidson_z, Duncan_z, Song_z, Bhatawdekar_z])
y = np.concatenate([Ilbert_scaling, Davidson_scaling, Duncan_scaling,
                             Song_scaling, Bhatawdekar_scaling])

#fit_lin = np.polynomial.polynomial.polyfit(x, y, 1)
fit_pow = np.polynomial.polynomial.polyfit(np.log10(x), np.log10(y), 1)
#fit_pow_h = np.polynomial.polynomial.polyfit(np.log10(x)[x>4], np.log10(y)[x>4], 1)

x_space = np.linspace(1, 10, 100)
plt.plot(x_space, np.power(10,fit_pow[0])*x_space**fit_pow[1])
#plt.plot(x_space, np.power(10,fit_pow_h[0])*x_space**fit_pow_h[1])
#plt.plot(x_space, fit_lin[0]+x_space*fit_lin[1],'--')
