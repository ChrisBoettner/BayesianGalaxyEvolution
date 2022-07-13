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
from model.analysis.calculations import calculate_q1_q2_relation

log_q1=np.linspace(9,13,100)
z=0

mstar_mbh_rel = calculate_q1_q2_relation(mstar, mbh, z, log_q1)[z]
              
data_mask = mstar_mbh_rel[:,1]>7 # for which we have actual data black hole data

# plot
plt.figure()

# fill 
from matplotlib import cm
colormap = cm.Greys 
mstar_mbh_rel_2 = calculate_q1_q2_relation(mstar, mbh, z, log_q1, sigma=2)[z]
mstar_mbh_rel_3 = calculate_q1_q2_relation(mstar, mbh, z, log_q1, sigma=3)[z]
plt.fill_between(mstar_mbh_rel_3[:, 0],
                 mstar_mbh_rel_3[:, 2], mstar_mbh_rel_3[:,3],
                 color=colormap(0.3), label=r'99.7\% percentiles')
plt.fill_between(mstar_mbh_rel_2[:, 0],
                 mstar_mbh_rel_2[:, 2], mstar_mbh_rel_2[:,3],
                 color=colormap(0.6), label=r'95\% percentiles')
plt.fill_between(mstar_mbh_rel[:, 0],
                 mstar_mbh_rel[:, 2], mstar_mbh_rel[:,3],
                 color=colormap(0.9), label=r'68\% percentiles')

# plot lines
plt.plot(mstar_mbh_rel[:,0][data_mask],
         mstar_mbh_rel[:,1][data_mask],
         label='constrained by data (observed black hole mass function)',
         color='lightgrey')
plt.plot(mstar_mbh_rel[:,0][np.logical_not(data_mask)],
         mstar_mbh_rel[:,1][np.logical_not(data_mask)],
         '--',label='not constrained by data',
         color='lightgrey')


plt.title('z=' + str(z))
plt.xlabel(r'Stellar Mass [$M_\odot$]')
plt.ylabel(r'Black Hole Mass [$M_\odot$]')

# fit
weights = 1/(mstar_mbh_rel_3[:,3][data_mask]-mstar_mbh_rel_3[:,2][data_mask])**2
params = np.around(np.polyfit(mstar_mbh_rel[:,0][data_mask]-11,
                              mstar_mbh_rel[:,1][data_mask],1,
                              w=weights),
                              2)

plt.text(9,12, r'High Mass (Solid Line) fit : $\log M_{bh}=$ '+ str(params[0])+\
                r' $\cdot\log (M_*/10^{11} M_\odot)$ + ' +str(params[1]),
                fontsize=16)
    
# reference plot
# baron_params  = [1.64, 7.88]
# reines_params = [1.05, 7.45]
# bentz_params  = [1.84, 8.40]

# def calc_lin(mstar_m, params):
#     mstar_m = mstar_m[mstar_m>9]
#     mbh_ref = params[0]*(mstar_m-11)+params[1]
#     return([mstar_m, mbh_ref])
    
# plt.plot(*calc_lin(mstar_mbh_rel[:,0], baron_params),
#          '-', linewidth = 2, color='black', alpha=0.7, label = 'Baron2019')
# plt.plot(*calc_lin(mstar_mbh_rel[:,0], reines_params),
#          '-.', linewidth = 2, color='black', alpha=0.7, label = 'Reines2015')
# plt.plot(*calc_lin(mstar_mbh_rel[:,0], bentz_params),
#          ':', linewidth = 2, color='black', alpha=0.7, label = 'Bentz2018')
from model.data.load import load_data_points
data = load_data_points('mstar_mbh')
labels = ['Baron2019','Reines2015','Bentz2018']
for i in range(3):
    data_i = data[data[:,2]==i]    
    plt.scatter(data_i[:,0],data_i[:,1], s=20, alpha = 0.5, label=labels[i])

plt.legend()
plt.tight_layout()