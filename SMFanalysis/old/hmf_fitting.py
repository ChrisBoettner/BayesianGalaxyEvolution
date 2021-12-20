#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 15:41:38 2021

@author: boettner
"""
from matplotlib import rc_file
rc_file('plots/settings.rc')  # <-- the file containing your settings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#  data
dataset = 'Davidson'

if dataset == 'Duncan':
    smf_data   = np.load('data/Duncan2014SMF.npz')
    cutoff     = 10.5
    smf_vals = np.arange(0,4).astype(str); hmf_vals= np.arange(4,8).astype(str) #get 4 to 7
if dataset == 'Davidson':
    smf_data   = np.load('data/Davidson2017SMF.npz')   
    cutoff     = 11.2
    smf_vals = np.array([2,5,7,8]).astype(str); hmf_vals= np.arange(1,5).astype(str) #get redshifts 1 to 4
hmf_data   = np.load('data/HMF.npz')

# parameter fitting
def params(data):
    data    = data[np.where(data[:,0]<cutoff),:][0,:,:]
    # fit power law (include bounds)
    fit     = np.polynomial.polynomial.polyfit(data[:,0], data[:,1], 1)
    #[log10(A), alpha]
    return(fit)

smf_powerlaw = np.array([params(smf_data[i]) for i in smf_vals])
hmf_powerlaw = np.array([params(hmf_data[i]) for i in hmf_vals])


# plot
plt.close('all')

fig, ax = plt.subplots(3, 2)
ax = ax.flatten()
x  = np.linspace(8.5,cutoff,1000)
for i in range(len(smf_vals)):
      ax[i].plot(x, hmf_powerlaw[i,0]+x*hmf_powerlaw[i,1],color='C1', label='HMF', linestyle='--')
      ax[i].scatter(hmf_data[hmf_vals[i]][:,0],hmf_data[hmf_vals[i]][:,1],color='C0',label='HMF data')
     
      ax[i].plot(x, smf_powerlaw[i,0]+x*smf_powerlaw[i,1],color='C1', label='SMF')
      ax[i].scatter(smf_data[smf_vals[i]][:,0],smf_data[smf_vals[i]][:,1],color='C0',label='SMF data', marker = 'x')
      
      ax[i].set_xlim([8,14])
      if i ==0:
          fig.legend()
fig.text(0.5, 0.04, '$log M$ [$M_\odot$]', ha='center', va='center')
fig.text(0.04, 0.5, 'log d$n$/dlog$m$ [Mpc$^{-1}$ dex$^{-1}$]', ha='center', va='center', rotation='vertical')

