#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:19:59 2021

@author: boettner
"""
import numpy as np
import pandas as pd

dicto = {}

def extr(i):
    data = pd.read_csv('mVector_PLANCK-SMT z: '+str(i)+'.0.txt',
                                       skiprows=12, sep=' ', header=None, usecols=[0,7]).to_numpy()
    data = np.log10(data)
    return(data)

dicto   = {str(i):extr(i) for i in range(0,20)}
    
np.savez('HMF.npz', **dicto)
