#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 09:01:22 2022

@author: chris
"""

import numpy as np
import pandas as pd

# Baldry

def convert(name, zs, znums, year):
    dictonary = {}
    for i, z in enumerate(zs):
        d = []
        for n in znums[i]:
            file_name = name + '_z' + str(z) + '_' + str(n) + '.smf'
            data = pd.read_csv(file_name, comment='#',
                               sep=' ', header=None).to_numpy()
            
            m = (data[:,0]+data[:,1])/2
            
            mean = data[:,2]
            upper = data[:,2] + data[:,3]
            lower = data[:,2] - data[:,4]
            
            if np.all(mean>0):
                val = np.log10(mean)
                err_u = np.log10(upper)-val
                err_l = val-np.log10(lower)
                err_l[np.isnan(err_l)] = 999
            else:
                val   = mean
                err_u = data[:,3]
                err_l = data[:,4]
            
            d.append(np.array([m, val, err_l, err_u]).T)
        d=np.concatenate(d)
        d=d[np.argsort(d[:,0])]  
        dictonary[str(i)] = d
    if name == 'muzzin_ilbert':
        name = 'ilbert'
    np.savez(name.capitalize() + str(year) + 'SMF.npz', **dictonary)
    return(dictonary)
    
baldry    = convert('baldry', [0], [[1]], 2012)
moustakas = convert('moustakas', [0, 1], [[1,2,3,4],[1,2,3]], 2013)
muzzin    = convert('muzzin_ilbert', [0, 1, 2, 3], [[1],[1,2],[1,2],[1]], 2013)
song      = convert('song', [4, 5, 6, 7, 8], [[1],[1],[1],[1],[1]], 2016)
tomczak   = convert('tomczak', [0,1,2,3], [[1],[1,2,3,4],[1,2],[1]], 2014)