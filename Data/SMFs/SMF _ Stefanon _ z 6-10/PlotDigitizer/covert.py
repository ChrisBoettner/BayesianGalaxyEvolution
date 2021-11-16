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

for i in range(5):
    mass = data.iloc[1:,0+2*i].to_numpy(dtype=float)
    phi  = np.log10(data.iloc[1:,1+2*i].to_numpy(dtype=float))
    l_err = phi - np.log10(lower_err.iloc[1:,1+2*i].to_numpy(dtype=float))
    u_err = np.log10(upper_err.iloc[1:,1+2*i].to_numpy(dtype=float)) - phi
    
    mask = ~np.isnan(mass)
    mass = mass[mask]
    phi   = phi[mask]
    l_err = l_err[mask]
    u_err = u_err[mask]
    
    dat = np.array([mass, phi, l_err, u_err]).T
    
    dat_processed[str(i)] = dat

np.savez('Stefanon2021SMF.npz', **dat_processed)