#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:00:12 2022

@author: chris
"""
import numpy as np
from hmf import MassFunction

def create_hmfs(Mmin=3, Mmax=21, redshift=range(20)):
    '''
    Create Halo Mass Functions using hmf package and save to file. 
    Currently build in such a way that HMFs get created for integer redshift,
    where HMF model is Sheth-Tormen. Saves to 3 files,
        HMF.npz : 
            dictonary of HMFs of form {str{z}: [log_halo_mass, log_phi]}
        turnover_mass.npz :
            dictonary of (log of) turnover mass where the HMF switches from 
            powerlaw-like behaviour to exponential behaviour, of form
            {str{z}: log_m_turnover}
        total_halo_number.npz :
            dictonary of total number of halos within cosmic volume between
            Mmin and Mmax, of form {str{z}: total_halo_num}
            
    Can adjust minimum and maximum halo mass.
    '''
    hmfs, log_m_nonlinear, log_total_halo_number = {}, {}, {}
    log_m_nonlinear = {}
    for z in redshift:
        # create hmf object
        mf = MassFunction(z=z, hmf_model='ST', Mmin=Mmin,
                          Mmax=Mmax)  # create hmf object
        # get halo masses
        log_m = np.log10(mf.m)
        # calculate log of phi
        phi = mf.dndlog10m
        log_phi = np.copy(phi)
        log_phi[log_phi <= 0] = -np.inf
        log_phi[log_phi > 0] = np.log10(log_phi[log_phi > 0])
        # write to dictionary
        hmfs[str(z)] = np.array([log_m, log_phi]).T
        
        # get (log of) turnover mass where the HMF switches from 
        # powerlaw-like behaviour to exponential behaviour for input redshift
        log_m_nonlinear[str(z)] = np.array([np.log10(mf.mass_nonlinear)])
        
        # get total number of halos in volume (under assumption that hmf is
        # 0 outside of [Mmin, Mmax])
        log_total_halo_number[str(z)] = np.array([mf.ngtm[0]])
    
    # save everything
    path = 'model/data/HMF/'
    np.savez(path + 'HMF.npz', **hmfs)
    np.savez(path + 'turnover_mass.npz', **log_m_nonlinear)
    np.savez(path + 'total_halo_number.npz', **log_total_halo_number)
    return
