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
    data_i = pd.read_csv('LF_Vmax_udFUV_z'+str(i)+'.dat', delim_whitespace=True).values
    
    data = data_i.copy()
    
    data[:,1] = data[:,1] + np.log10(2.5) # change normalsation from per mag to per dex
    
    data[:,0] = data[:,0] 
    
    data[:,2] = data_i[:,3] # lower error
    data[:,3] = data_i[:,2] # upper error
    
    dicto[str(i)] = data
    
np.savez('Madau2014UVLF.npz', **dicto)
