#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 15:09:05 2023

@author: chris
"""

import pandas as pd
import numpy as np

sources = ['Wyder05',  'Schiminovich05',  'Robotham11', 'Cucciati12', 
           'Dahlen07', 'Reddy09', 'Bouwens12', 'Schenker13']

wyder        = [[0.05, -1.82]]
schiminovich = [[0.3,-1.5], [0.5,-1.39], [0.7,-1.2], [1,-1.25]]
robotham     = [[0.05, -1.77]]
cucciati     = [[0.1, -1.75], [0.3, -1.55], [0.5, -1.44], [0.7, -1.24],
                [0.9, -0.99], [1.1,-0.94], [1.5, -0.95], [2.2, -0.75],
                [3, -1.04], [4, -1.69]]
dahlen       = [[1.1,-1.02], [1.7,-0.75], [2.2, -0.87]]
reddy        = [[2.3, -0.75], [3.1, -0.97]]
bouwens      = [[3.8, -1.29], [4.9, -1.42], [5.9, -1.65], [7, -1.79], 
                [7.9, -2.09]]
schenker     = [[7, -2], [8, -2.21]]
    
processed_data = [wyder, schiminovich, robotham, cucciati, 
                  dahlen, reddy, bouwens, schenker]
processed_data = [np.array(d) for d in processed_data]


for d in processed_data:
    d[:,1] = d[:,1] - np.log10(1.8/1.15)

data_dict = {sources[i]:processed_data[i] for i in range(len(sources))}

np.savez('Madau2014SFRD.npz', **data_dict)