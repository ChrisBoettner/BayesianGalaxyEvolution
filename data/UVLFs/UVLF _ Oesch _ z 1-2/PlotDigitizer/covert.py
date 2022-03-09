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

for i in range(2):
    mag = data.iloc[1:,0+2*i].to_numpy(dtype=float)
    phi  = np.log10(data.iloc[1:,1+2*i].to_numpy(dtype=float))
    l_err = phi - np.log10(lower_err.iloc[1:,1+2*i].to_numpy(dtype=float))
    u_err = np.log10(upper_err.iloc[1:,1+2*i].to_numpy(dtype=float)) - phi
    
    mask = ~np.isnan(mag)
    mag  = mag[mask]
    phi   = phi[mask]
    l_err = l_err[mask]
    u_err = u_err[mask]
    
    phi = phi + np.log10(2.5) # change normalsation from per mag to per dex
    
    dat = np.array([mag, phi, l_err, u_err]).T
    
    dat_processed[str(i)] = dat

np.savez('Oesch2010UVLF.npz', **dat_processed)
