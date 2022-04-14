#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 16:31:21 2021

@author: boettner
"""
import pandas as pd
import numpy as np

bh = {}
song = {}

for i in range(4):
    # def process(z):
    data = pd.read_csv(str(i+6)+'.csv')
    
    # bha
    bm = np.around(data.iloc[1:,0].to_numpy(dtype=float),2) 
    b  = np.log10(data.iloc[1:,1].to_numpy(dtype=float))     
    bl = b-np.log10(data.iloc[1:,3].to_numpy(dtype=float)) 
    bu = np.log10(data.iloc[1:,5].to_numpy(dtype=float))-b     
    
    b  = np.array([bm,b,bl,bu]).T ; b = b[~np.isnan(b).any(axis=1)]
    
    bh[str(i)] = b
    
    if i != 3:
    
        # song    
        sm = np.around(data.iloc[1:,6].to_numpy(dtype=float),2) 
        s  = np.log10(data.iloc[1:,7].to_numpy(dtype=float))     
        sl = s-np.log10(data.iloc[1:,9].to_numpy(dtype=float)) 
        su = np.log10(data.iloc[1:,11].to_numpy(dtype=float))-s     
        
        s  = np.array([sm,s,sl,su]).T ; s = s[~np.isnan(s).any(axis=1)]
    
        song[str(i)] = s
        
np.savez('Song2016SMF.npz', **song)
np.savez('Bhatawdekar2018SMF.npz', **bh)