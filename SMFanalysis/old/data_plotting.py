#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 10:50:42 2021

@author: boettner
"""
from matplotlib import rc_file
rc_file('plots/settings.rc')  # <-- the file containing your settings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Duncan      = np.load('data/Duncan2014SMF.npz')
Ilbert      = np.load('data/Ilbert2013SMF.npz');   Ilbert_vals   = np.array([2,5])
Davidson    = np.load('data/Davidson2017SMF.npz'); Davidson_vals = np.array([2,5,7,8])
Song        = np.load('data/Song2016SMF.npz')
Bhatawdekar = np.load('data/Bhatawdekar2018SMF.npz')

z1 = [Ilbert['2'],Davidson['2']]
z2 = [Ilbert['5'],Davidson['5']]
z3 = [Davidson['7']]
z4 = [Davidson['8'], Duncan['0']]
z5 = [Duncan['1']]
z6 = [Duncan['2'], Song['0'], Bhatawdekar['0']]
z7 = [Duncan['3'], Song['1'], Bhatawdekar['1']]
z8 = [Song['2'], Bhatawdekar['2']]
z9 = [Bhatawdekar['3']]

z = [z1,z2,z3,z4,z5,z6,z7,z8,z9]

plt.close('all')

fig, ax = plt.subplots(3,3)
ax      = ax.flatten()

colors      = ['C0','C1','C2','C3', 'C4']
color_start = [0,0,1,1,2,2,2,3,4]

for i in range(9):
    data = z[i]
    for j in range(len(data)):
        d = data[j]
        ax[i].errorbar(d[:,0], d[:,1], d[:,2:].T, fmt='o', capsize = 3, markersize = 2,
                       color = colors[color_start[i]+j])
        
        
        ax[i].set_xlim(7.9,12); ax[i].set_ylim(-6,0)

plt.tight_layout()


