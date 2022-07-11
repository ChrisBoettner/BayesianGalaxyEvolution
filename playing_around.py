#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:41:12 2022

@author: chris
"""

from model.interface import load_model, run_model, save_model
from model.plotting.plotting import *

mstar  = load_model('mstar','changing')
mbh    = load_model('mbh','quasar', prior_name='successive')
lbol   = load_model('Lbol', 'eddington', prior_name='successive')


## BLACK HOLE MASS STELLAR MASS RELATION
#%%
import numpy as np
import matplotlib.pyplot as plt
z= 0
halo_masses = np.linspace(10,16,100)

# masses
mstar_m     = mstar.physics_model.at_z(z).\
              calculate_log_quantity(halo_masses, *mstar.parameter.at_z(z))
mbh_m       = mbh.physics_model.at_z(z).\
              calculate_log_quantity(halo_masses, *mbh.parameter.at_z(z))
              
data_mask = mbh_m>7 # for which we have actual data

# plot
plt.figure()
plt.plot(mstar_m[data_mask], mbh_m[data_mask], label='constrained by data (observed black hole mass function)')
plt.plot(mstar_m[np.logical_not(data_mask)], 
         mbh_m[np.logical_not(data_mask)],  '--',label='not constrained by data')
plt.title('z=' + str(z))
plt.xlabel(r'Stellar Mass [$M_\odot$]')
plt.ylabel(r'Black Hole Mass [$M_\odot$]')

# fit
params = np.around(np.polyfit(mstar_m[data_mask]-11, mbh_m[data_mask],1),2)

baron_params = [1.64, 7.88]
plt.text(6,7, r'High Mass (Blue Part) fit : $\log M_{bh}=$ '+ str(params[0])+\
             r' $\cdot\log (M_*/10^{11} M_\odot)$ + ' +str(params[1]))
    
# reference plot
baron_params  = [1.64, 7.88]
reines_params = [1.05, 7.45]
bentz_params  = [1.84, 8.40]

def calc_lin(mstar_m, params):
    mstar_m = mstar_m[mstar_m>9]
    mbh_ref = params[0]*(mstar_m-11)+params[1]
    return([mstar_m, mbh_ref])
    
plt.plot(*calc_lin(mstar_m, baron_params), '-', alpha = 0.4, linewidth = 2, color='black', label = 'Baron2019')
plt.plot(*calc_lin(mstar_m, reines_params), '-.', alpha = 0.6, linewidth = 2, color='black', label = 'Reines2015')
plt.plot(*calc_lin(mstar_m, bentz_params), ':', alpha = 0.8, linewidth = 2, color='black', label = 'Bentz2018')

plt.legend()