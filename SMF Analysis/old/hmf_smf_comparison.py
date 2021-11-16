#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 12:09:39 2021

@author: boettner
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from help_functions import plot_schechter

## SMF
# Data Points
smf_data    = np.load('data/highzSMF.npz')
#  Schechter Parameter
smf         = pd.read_hdf('data/highzSMFparams.hdf').values

#smf[:,0] = np.log()

## HMF
#  Data Points
ex          = lambda i: pd.read_csv('data/HMF/mVector_PLANCK-SMT z: '+str(i)+'.0.txt',
                                    skiprows=12, sep=' ', header=None, usecols=[0,7]).to_numpy().T 
hmf_data    = {'z'+str(i):ex(i) for i in range(4,9)}

## Plot
plt.close()
fig, ax = plt.subplots(3, 2)
ax = ax.flatten()
for i in range(0,5):
    ax[i].loglog(*hmf_data['z'+str(i+4)],label = 'Halo Mass Function')
    ax[i].loglog(*plot_schechter(*smf[i,:], mode='dndlogm'), label = 'Stellar Mass Function (Song 2016)',linestyle='--')
    ax[i].scatter(np.power(10,smf_data['z'+str(i+4)].T[0]), smf_data['z'+str(i+4)].T[1], color ='C1')
    ax[i].set_ylim([1e-19,1e+5])
    if i ==0:
        fig.legend()
fig.text(0.5, 0.04, '$M$ [$M_\u2609$]', ha='center', va='center')
fig.text(0.04, 0.5, 'd$n$/dlog$m$ [Mpc$^{-1}$ dex$^{-1}$]', ha='center', va='center', rotation='vertical')

