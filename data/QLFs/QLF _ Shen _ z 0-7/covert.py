#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 16:00:49 2022

@author: chris
"""
import numpy as np
import pandas as pd

data = pd.read_pickle('alldata.pkl')

lum      = data['L_OBS']
phi      = data['P_OBS']
phi_err  = data['D_OBS']
redshift = data ['Z'] 

dat_processed = {}

for z in range(8):
    
    z_mask = np.where((redshift > z-0.5) & (redshift<=z+0.5))
    
    log_lum_at_z     = lum[z_mask]
    log_phi_at_z     = phi[z_mask]
    log_phi_err_at_z = phi_err[z_mask]

    
    data_at_z = np.array([log_lum_at_z, log_phi_at_z, 
                          log_phi_err_at_z, log_phi_err_at_z]).T
    
    dat_processed[str(z)] = data_at_z

np.savez('Shen2020QLF.npz', **dat_processed)
