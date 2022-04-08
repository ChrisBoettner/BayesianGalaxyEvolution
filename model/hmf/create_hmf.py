#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:00:12 2022

@author: chris
"""
import numpy as np
from hmf import MassFunction

def create_hmfs(redshifts, Mmin = 4, Mmax = 18):
    hmfs = {}
    for z in range(20):
        mf = MassFunction(z=z, hmf_model = 'ST', Mmin=Mmin, Mmax=Mmax) # create hmf object
        log_m    = np.log10(mf.m)
        # take log of phi, some values are 0 though, so we do it in two parts
        phi      = mf.dndlog10m
        log_phi = np.copy(phi)
        log_phi[log_phi<=0] = -np.inf
        log_phi[log_phi>0] = np.log10(log_phi[log_phi>0])
        # write to dictionary
        hmfs[str(z)] = np.array([log_m, log_phi]).T
    return(hmfs)