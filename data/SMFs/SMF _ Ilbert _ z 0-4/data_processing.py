#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:46:09 2021

@author: boettner
"""

import numpy as np
import pandas as pd

dicto = {}
for i in range(1,9):
    data_i = pd.read_csv('MF_Vmax_blue_'+str(i)+'.dat', delim_whitespace=True, header = None).values
    
    data = data_i.copy()  
    
    data[:,2] = data_i[:,3]
    data[:,3] = data_i[:,2]
    
    #data[:,1:] = np.power(10, data[:,1:])
    #data[:,2] = data[:,2]+data[:,1]
    #data[:,3] = data[:,3]-data[:,1]
    
    dicto[str(i-1)] = data
 
# check that before, which IMF? what are the modelling assumptions? read paper    
np.savez('Ilbert2013SMF.npz', **dicto)