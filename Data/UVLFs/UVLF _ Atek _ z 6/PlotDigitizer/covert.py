#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 16:31:21 2021

@author: boettner
"""
import pandas as pd
import numpy as np

data      = pd.read_csv('data.csv')
upper_err = pd.read_csv('eu.csv')
lower_err = pd.read_csv('el.csv')

dat_processed = {}

for i in range(1):
    mag   = data.iloc[1:,0+2*i].to_numpy(dtype=float)
    phi   = data.iloc[1:,1+2*i].to_numpy(dtype=float)
    
    ## screwed up while grabbing, have to move one datapoint now
    sort = np.argsort(mag)
    mag = mag[sort]
    phi = phi[sort]    
    l_err = lower_err.iloc[1:,1+2*i].to_numpy(dtype=float)
    l_err = l_err[sort]
    ##
        
    l_err = phi - lower_err.iloc[1:,1+2*i].to_numpy(dtype=float)
    u_err = upper_err.iloc[1:,1+2*i].to_numpy(dtype=float) - phi
    
    mask = ~np.isnan(mag)
    mag   = mag[mask]
    phi   = phi[mask]
    l_err = l_err[mask] 
    u_err = u_err[mask]
    
    phi = phi + np.log10(2.5) # change normalsation from per mag to per dex
    
    dat = np.array([mag, phi, l_err, u_err]).T
    
    dat_processed[str(i)] = dat

np.savez('Atek2018UVLF.npz', **dat_processed)