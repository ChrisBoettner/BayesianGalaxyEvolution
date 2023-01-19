#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 16:31:21 2021

@author: boettner
"""
import pandas as pd
import numpy as np

data = pd.read_csv('data.csv')
sources = ['Elsner08',  'Santini12',  'Mortlock11', 'Gonzalez11', 'Stark13', 
           'Grazian15', 'Song2016', 'Bhatawdekar2018']

data_dict   = {}
for i in range(0, len(sources)):
    ind = int(2*i)
    d = data.iloc[1:,[ind,ind+1]].values.astype(float)
    d = d[~np.isnan(d).any(axis=1)]
    data_dict[sources[i]] = d
    
np.savez('Bhatawdekar2018SMD.npz', **data_dict)