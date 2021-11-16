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

## IMPORT DATA
Duncan      = np.load('data/Duncan2014UVLF.npz')
Bouwens     = np.load('data/Bouwens2015UVLF.npz')
Bouwens2    = np.load('data/Bouwens2021UVLF.npz')
Madau       = np.load('data/Madau2014UVLF.npz')
# turn log values to actual values for fitting, and sort to correct redshift
z1  = np.concatenate([Madau['4'] [:,:2]])                                     
z2  = np.concatenate([Bouwens2['0'],Madau['7'] [:,:2]])
z3  = np.concatenate([Madau['8'] [:,:2],Bouwens2['1']])
z4  = np.concatenate([Madau['9'][:,:2], Duncan['0'] [:,:2], Bouwens['0'],Bouwens2['2']])
z5  = np.concatenate([Duncan['0'] [:,:2], Bouwens['1'],Bouwens2['3']])
z6  = np.concatenate([Duncan['1'] [:,:2], Bouwens['2'],Bouwens2['4']])
z7  = np.concatenate([Duncan['2'] [:,:2], Bouwens['3'],Bouwens2['5']])
z8  = np.concatenate([Bouwens['4'],Bouwens2['6']])
z9  = np.concatenate([Bouwens2['7']])
z10 = np.concatenate([Bouwens2['8']])
lfs = [z1,z2,z3,z4,z5,z6,z7,z8,z9,z10]

for lf in lfs:
    lf[:,1] = np.power(10, lf[:,1])


plt.close('all')
fig, ax = plt.subplots(3,3, sharex=True, sharey=True); ax = ax.flatten()
for i in range(9):        
    ax[i].scatter(lfs[i][:,0],lfs[i][:,1])
    #ax[i].set_xscale('log')
    ax[i].set_yscale('log')              
    #ax[i].set_xlim([5e+6,2e+12])
    #ax[i].set_ylim([1e-6,1e+3])   
fig.suptitle ('UV Luminosity Function')
ax[7].set_xlabel('$M_{UV}$')
ax[3].set_ylabel('$\phi(L_{UV})$ [cMpc$^{-1}$ dex$^{-1}$]')
fig.set_tight_layout(True)

# turn into SFRs
sfr_conv = lambda M: 3.4e-8*np.power(10,-M/2.5) 

sfrfs = np.array(lfs)
for sfrf in sfrfs:
    sfrf[:,0] = sfr_conv(sfrf[:,0])
    

np.save('sfrf', sfrfs)
