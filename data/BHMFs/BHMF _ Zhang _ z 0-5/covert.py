#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 16:00:49 2022

@author: chris
"""
import numpy as np
import pandas as pd

dat_processed = {}

for z in range(6):
    data_at_z             = pd.read_csv('z=' + str(z), header=None, 
                                        delimiter = ' ').values    
    dat_processed[str(z)] = data_at_z

np.savez('Zhang2021BHMF.npz', **dat_processed)