#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:19:59 2021

@author: boettner
"""
import numpy as np
import pandas as pd

dicto = {}

# note: already dust corrected

for i in range(0,10):
    data_i = pd.read_csv('LF_Vmax_udFUV_z'+str(i)+'.dat', 
                         delim_whitespace=True).values
    
    data = data_i.copy()
    
    data[:,1] = data[:,1] + np.log10(2.5) # change normalisation from per mag to per dex
    
    data[:,0] = data[:,0] 
    
    data[:,2] = data_i[:,3] # lower error
    data[:,3] = data_i[:,2] # upper error
    
    #data = data[~np.isnan(data).any(axis=1)]
    
    dicto[str(i)] = data
 
dictz = {}
dictz['0'] = np.concatenate([dicto['0'], dicto['1']])
dictz['1'] = np.concatenate([dicto['2'], dicto['3'], dicto['4'], dicto['5'], dicto['6']])
dictz['2'] = dicto['7']    
dictz['3'] = dicto['8']
dictz['4'] = dicto['9']    
    
np.savez('Cucciati2012UVLF.npz', **dictz)
