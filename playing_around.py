#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:41:12 2022

@author: chris
"""
from model.interface import load_model, run_model, save_model
from model.plotting.plotting import *

#%%
#mstar = load_model('mstar', 'stellar_blackhole')
muv   = load_model('Muv', 'stellar_blackhole')
#mbh   = load_model('mbh', 'quasar')
#lbol  = load_model('Lbol', 'eddington')

from model.analysis.surveys import calculate_expected_number

o = np.rint(calculate_expected_number(muv, [7, 9, 11, 13, 15], 1929, 28, 
                                      sigma=2, num_samples=int(1e+4),                                      
                                      percentiles_mode='uncertainties')[2]
                                      ).astype(int)

# then:
# get lum function from model, integrate up to lim_mag (do for sample)
# (similar to calculate_quantity_density method, but without multiplying by quantity)
# multiply by volume to get expected number of sources


# JWST redshifts: 7, 9, 11, 13, 15
# Euclid redshifts: 5, 6, 7, 8, 11
# use one lim_mag per instrument/field, take from 
#   https://arxiv.org/pdf/2207.12356.pdf
#   https://www.aanda.org/articles/aa/pdf/2022/10/aa43950-22.pdf
#   https://arxiv.org/pdf/2211.07865.pdf (table 3)
# maybe just use mag = 26 for euclid, mag = 30 for JWST
# fields:
#    euclid deep
#    JWST ceers, cosmos-web, jades..
# (all results without lensing)
# upper bounds for depths
# two tables, one euclid, one jwst
# reference to papers
# check area of euclid again
# only calculate upper bounds?

#%%
# muv   = load_model('Muv', 'stellar_blackhole',
#                   sfr=True)

# plt.close('all')

# # Plot_q1_q2_relation(mstar, muv, z=0, ratio=True,
# #                     quantity_range=np.linspace(8.12,11.92,100), columns='single')

# from model.analysis.calculations import calculate_q1_q2_relation
# from model.plotting.convience_functions import plot_data_with_confidence_intervals

# rels = {z:calculate_q1_q2_relation(mstar,muv,z,np.linspace(8.12,11.12,100), ratio=True) 
#         for z in range(8)}

# fig, ax = plt.subplots(1,1)
# for z in rels.keys():
#     d = rels[z][1]
#     ax.plot(d[:,0], d[:,1], label=str(z))
#     #plot_data_with_confidence_intervals(ax, rels[z], 'black')
    
# plt.legend()

#%%
# from model.analysis.calculations import calculate_q1_q2_relation

# muv   = load_model('Muv', 'stellar_blackhole',
#                   sfr=True)

# ms = [8.5,10, 11, 12]
# zs = range(9)

# rels = {z:calculate_q1_q2_relation(mstar,muv,z, ms, ratio=True) 
#         for z in zs}

# ssfr = {}

# for i in range(4):
#     vals = []
#     for z in zs:
#         v = 10**rels[z][1][i]
#         v[2] = v[1]-v[2]
#         v[3] = v[3]-v[1]
#         vals.append([z, *v[[1,2,3]]])
#     ssfr[ms[i]] = np.array(vals)
    
# for m in ms:
#     np.savetxt('m='+str(m)+'.txt', ssfr[m], delimiter='    ',
#                fmt=('%1.0f','%1.4e','%1.4e','%1.4e'),
#                header = 'z    median    lower error    upper error')
    
# plt.close()
    
# fig, ax = plt.subplots(1,1)

# for m in ms[::-1]:
#     ax.errorbar(ssfr[m][:,0], ssfr[m][:,1], yerr=ssfr[m][:,2:].T,
#                 elinewidth=3, capsize=7, markersize=10, linewidth=3,
#                 label=f'log m = {m}' + r' [$M_\odot$]')
    
# ax.set_yscale('log')
# ax.legend()
# ax.set_xlabel('Redshift')
# ax.set_ylabel(r'sSFR [1/yr$^{-1}$]')
# fig.tight_layout()

