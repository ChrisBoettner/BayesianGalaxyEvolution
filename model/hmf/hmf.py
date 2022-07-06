#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:00:12 2022

@author: chris
"""
import numpy as np
from hmf import MassFunction

from model.helper import make_array

################ CALL FUNCTIONS ###############################################

def get_log_mass_nonlinear(redshifts):
    '''
    Return (log of) turnover mass where the HMF switches from powerlaw-like 
    behaviour to exponential behaviour for input redshift. Convienience 
    function where masses are retrieved from dictionary created using 
    calculate_log_mass_nonlinear so that calculation does not need to be
    redone every time
    '''
    redshifts = make_array(redshifts)
    masses = {0: 12.641732938023212,
              1: 11.19481037359994,
              2: 9.95792461057341,
              3: 8.985185160624045,
              4: 8.200879226980947,
              5: 7.5481524076095345,
              6: 6.9906124460355805,
              7: 6.504570343810428,
              8: 6.074011310053344,
              9: 5.687658678743158,
              10: 5.3373399364162015,
              11: 5.0169440795795985,
              12: 4.721740450021632,
              13: 4.4480627721112,
              14: 4.192976137035432,
              15: 3.954139767585932,
              16: 3.7295503278644326,
              17: 3.5176079649479837,
              18: 3.316975197110768,
              19: 3.126485380975656}
    return([masses[z] for z in redshifts])

################ CREATION FUNCTIONS ###########################################

def calculate_log_mass_nonlinear(redshift):
    '''
    Calculate the (log of) turnover mass where the HMF switches from 
    powerlaw-like behaviour to exponential behaviour for input redshift.

    '''
    redshift = make_array(redshift)
    
    log_m_nonlinear = {}
    for z in redshift:
        mf = MassFunction(z=z, hmf_model='ST')
        log_m_nonlinear[z] = np.log10(mf.mass_nonlinear)
    return(log_m_nonlinear)
        
    

def create_hmfs(Mmin=0.5, Mmax=21):
    '''
    Create Halo Mass Functions using hmf package. Currently build in such a way
    that HMFs get created for integer redshifts from 0 to 19. HMF model is
    Sheth-Tormen.
    Can adjust minimum and maximum value.
    '''
    hmfs = {}
    for z in range(20):
        mf = MassFunction(z=z, hmf_model='ST', Mmin=Mmin,
                          Mmax=Mmax)  # create hmf object
        log_m = np.log10(mf.m)
        # take log of phi, some values are 0 though, so we do it in two parts
        phi = mf.dndlog10m
        log_phi = np.copy(phi)
        log_phi[log_phi <= 0] = -np.inf
        log_phi[log_phi > 0] = np.log10(log_phi[log_phi > 0])
        # write to dictionary
        hmfs[str(z)] = np.array([log_m, log_phi]).T
    return(hmfs)
