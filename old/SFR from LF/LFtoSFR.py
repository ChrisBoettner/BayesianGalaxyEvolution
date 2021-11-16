#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 16:46:47 2021

@author: boettner
"""
import numpy as np
import matplotlib.pyplot as plt

sf = np.load('sfrf.npy', allow_pickle = True)
sm = np.load('sn_sm.npy')

def find_closest(data,reference):
    '''
    Find indices of closest matching values in reference array compared to
    data (observations) array
    '''
    ind = []
    for i in range(len(data)):
        diff = np.abs(reference-data[i]) 
        ind.append(np.argmin(diff))
    return(ind)

vals = []

for i in range(9):
    o = find_closest(sf[i][:,1],sm[i][:,1])
    m = sm[i][o][:,0]; sfr = sf[i][:,0]
    v = np.array([m,sfr]).T
    vals.append(v)
    
plt.close('all')
for i in range(9):
    v = vals[i]
    
    plt.scatter(v[:,0], v[:,1])
    plt.yscale("log");plt.xscale("log")
    plt.xlabel('$M_*/M_\odot$');plt.ylabel('SFR[$M_\odot$ yr$^{-1}$]')
    
fig, ax = plt.subplots(3,3, sharex=True, sharey=True); ax = ax.flatten()
for i in range(9): 
    v = vals[i]
       
    ax[i].scatter(v[:,0], v[:,1])
    ax[i].set_xscale('log')
    ax[i].set_yscale('log')              
    #ax[i].set_xlim([5e+6,2e+12])
    #ax[i].set_ylim([1e-6,1e+3])   
fig.suptitle ('SFR vs $M_*$')
ax[7].set_xlabel('$M_*/M_\odot$')
ax[3].set_ylabel('SFR[$M_\odot$ yr$^{-1}$]')
fig.set_tight_layout(True)