#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:19:59 2021

@author: boettner
"""
import numpy as np
import pandas as pd

dicto = {}
for i in range(4,8):
    data = pd.read_csv('Duncan14_MF_z'+str(i)+'.cat', delim_whitespace=True).values[:,:-1]
    
    value       = np.log10(data[:,1])
    lower_error = value - np.log10(data[:,1]-data[:,2])
    upper_error = np.log10(data[:,1]+data[:,3]) - value
    
    data[:,1] = value
    data[:,2] = lower_error
    data[:,3] = upper_error    
    
    data[np.where(data == np.inf)] = 999
    
    dicto[str(i-4)] = data
    
np.savez('Duncan2014SMF.npz', **dicto)
