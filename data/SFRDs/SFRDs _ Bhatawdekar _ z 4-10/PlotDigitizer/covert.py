#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 16:31:21 2021

@author: boettner
"""
import pandas as pd
import numpy as np

data = pd.read_csv('data.csv')
sources = ['McLeod16',  'Finkelstein15',  'Bouwens15', 'Ishigaki17', 
           'Bhatawdekar18']

data_dict   = {}
for i in range(0, len(sources)):
    ind = int(2*i)
    d = data.iloc[1:,[ind,ind+1]].values.astype(float)
    d = d[~np.isnan(d).any(axis=1)]
    
    data_dict[sources[i]] = d
    
np.savez('Bhatawdekar2018SFRD.npz', **data_dict)