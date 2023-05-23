#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:43:31 2023

@author: chris
"""
import numpy as np
import pandas as pd

data = pd.read_csv("schulze2010total.csv")

data['l_err'] = np.ones(data.shape[0])
data['u_err'] = np.ones(data.shape[0])

data = {'0': data.to_numpy()}

np.savez('Schulze2010total.npz', **data)