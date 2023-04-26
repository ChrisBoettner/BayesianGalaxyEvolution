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
#muv   = load_model('Muv', 'stellar_blackhole')
#mbh   = load_model('mbh', 'quasar')
lbol  = load_model('Lbol', 'eddington')

# #%%
# import pandas as pd

# from model.data.load import load_ndf_data

# quantities = ['mstar', 'Muv', 'mbh', 'Lbol']
# ndf = ['SMF', 'UVLF', 'BHMF', 'QLF']

# columns = ['z', 'fill', 'log10 Phi [ cMpc^-1 dex^-1]', 'lower uncertainity (log10)', 
#            'upper uncertainity (log10)', 'Source']
# quantity_val = ['log10 M_star [M_sun]', 'UV Magnitude', 'log10 M_bh [M_sun]', 
#                 'log10 L_bol [erg s^-1]'] 

# for j in range(4):
#     all_data  = []
#     groups = load_ndf_data(quantities[j], -100)[0]
    
#     for group in groups:
#         for i in range(len(group.redshift)):
#             data = group._data[str(i)]
#             z = np.repeat(group.redshift[i], len(data))[:,np.newaxis]
#             name =  np.repeat(group.label, len(data))[:,np.newaxis]
            
#             data = np.append(z, data, axis=1)
#             data = np.append(data, name, axis=1)
            
#             if len(all_data)==0:
#                 all_data = data
#             else:
#                 all_data = np.append(all_data, data, axis=0)
            
#     all_data = pd.DataFrame(all_data)
#     all_data = all_data.astype({0:'float'}).astype({0:'int'})
#     all_data = all_data.astype({1:'float', 2:'float', 3:'float', 4:'float', 5:'str'})
#     all_data = all_data.sort_values([0,5])
    
#     col = columns.copy()
#     col[1] = quantity_val[j]
    
#     all_data.columns = col
    
#     all_data.to_csv(f'{ndf[j]}.csv', index=False)
    
#         # z_sort = np.argsort(all_data[:,0].astype(float).astype(int))
#         # all_data = all_data[z_sort]