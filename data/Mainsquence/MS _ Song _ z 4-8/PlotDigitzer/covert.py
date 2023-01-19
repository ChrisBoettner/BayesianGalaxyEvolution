#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 16:31:21 2021

@author: boettner
"""
import pandas as pd
import numpy as np

dat_processed = {}

for i in range(4,9):
    data = pd.read_csv('z' + str(i) + '.csv')
    
    mag = data.iloc[1:,0].to_numpy(dtype=float)
    mstar  = data.iloc[1:,1].to_numpy(dtype=float)
    
    # remove nans
    mag   = mag[np.isfinite(mag)]
    mstar = mstar[np.isfinite(mstar)]     
    
    ## screwed up while grabbing, fix now
    sort_data = np.argsort(mag)
    mag = mag[sort_data]; mstar = mstar[sort_data] 
    
    dat = np.array([mag, mstar]).T
    
    dat_processed[str(i)] = dat

np.savez('Muv_mstar_Song2016.npz', **dat_processed)