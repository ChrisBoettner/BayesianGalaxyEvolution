#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 17:02:09 2023

@author: chris
"""

import pandas as pd
import numpy as np

# cross match shen and baron catalogue
shen2011  = pd.read_csv('Shen2011_catalogue.tsv', sep=";")
baron2019 = pd.read_csv('Baron2019_catalogue.csv', sep=", ")

merged = pd.merge(baron2019, shen2011, how='left', on=['plate','fiber','mjd'])
merged['e_log_BH_mass'] = merged['e_logBH']

merged = merged[['log_BH_mass', 'e_log_BH_mass', 'log_L_bol']]

merged = merged.to_numpy()

np.save('mbh_Lbol_Baron2019.npy', merged)