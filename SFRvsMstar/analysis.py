#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 13:26:00 2021

@author: boettner
"""

import numpy as np
import matplotlib.pyplot as plt

sfrf_nf   = np.load('sfrf_nf.npy')
sfrf_sn   = np.load('sfrf_sn.npy')
sfrf_snbh = np.load('sfrf_snbh.npy')
smf_nf    = np.load('smf_nf.npy')
smf_sn    = np.load('smf_sn.npy')
smf_snbh  = np.load('smf_snbh.npy')

smf  = smf_snbh
sfrf = sfrf_snbh

a = []; b = []
for i in range(10):
    l = np.where(smf[i,:,0]>1e+7)[0][0]; u = np.where(smf[i,:,0]<1e+11)[0][-1]
    par = np.polyfit(np.log10(smf[i,l:u,0]/np.power(10,9.7)), np.log10(sfrf[i,l:u,0]), deg=1)
    a.append(par[0])
    b.append(par[1])

plt.close('all')
plt.figure()
plt.xlim([1e+7,1e+11])
plt.ylim([1e-3,1e+2])
plt.xscale('log')
plt.yscale('log')
for i in range(7):
    plt.scatter(smf[i,:,0], sfrf[i,:,0], label = str(i))

plt.legend()

plt.figure()
plt.scatter(range(10),a)
plt.figure()
plt.scatter(range(10),b)