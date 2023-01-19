#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:46:09 2021

@author: boettner
"""

import numpy as np
import pandas as pd

dicto = {}
for i in range(0,9):
    data_i = pd.read_csv('mf_mass2b_fl5b_act_Vmax'+str(i)+'.dat',
                         delim_whitespace=True, header = None).values
    
    data = data_i.copy()  
    
    data[:,2] = data_i[:,3]
    data[:,3] = data_i[:,2]
    
    #data[:,1:] = np.power(10, data[:,1:])
    #data[:,2] = data[:,2]+data[:,1]
    #data[:,3] = data[:,3]-data[:,1]
    
    dicto[str(i)] = data
    
dictz = {}
dictz['0'] = dicto['0']
dictz['1'] = np.concatenate([dicto['1'], dicto['2'], dicto['3']])
dictz['2'] = np.concatenate([dicto['4'], dicto['5']])
dictz['3'] = np.concatenate([dicto['6'], dicto['7']])
dictz['4'] = dicto['8']    

np.savez('Davidson2017SMF.npz', **dictz)