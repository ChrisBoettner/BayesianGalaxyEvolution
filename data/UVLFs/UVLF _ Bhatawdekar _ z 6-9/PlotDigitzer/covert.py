#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 16:31:21 2021

@author: boettner
"""
import pandas as pd
import numpy as np

data = pd.read_csv('data.csv')
upper_err = pd.read_csv('eu.csv')
lower_err = pd.read_csv('el.csv')

dat_processed = {}

for i in range(4):
    mag = data.iloc[1:,0+2*i].to_numpy(dtype=float)
    phi  = np.log10(data.iloc[1:,1+2*i].to_numpy(dtype=float))
    
    ## screwed up while grabbing, fix now
    sort_data = np.argsort(mag)
    mag = mag[sort_data]; phi = phi[sort_data] 
    l_err_sort = np.argsort(lower_err.iloc[1:,0+2*i].to_numpy(dtype=float))
    u_err_sort = np.argsort(upper_err.iloc[1:,0+2*i].to_numpy(dtype=float))
    l_err = np.log10(lower_err.iloc[1:,1+2*i].to_numpy(dtype=float)[l_err_sort]) 
    u_err = np.log10(upper_err.iloc[1:,1+2*i].to_numpy(dtype=float)[u_err_sort])
    ##
    
    l_err = phi - l_err
    u_err = u_err - phi
    
    mask = ~np.isnan(mag)
    mag  = mag[mask]
    phi   = phi[mask]
    l_err = l_err[mask]
    u_err = u_err[mask]
    
    dat = np.array([mag, phi, l_err, u_err]).T
    
    if i == 1:
        dat[:,1:] = dat[:,1:][::-1]
    
    dat_processed[str(i)] = dat

np.savez('Bhatawdekar2019UVLF.npz', **dat_processed)