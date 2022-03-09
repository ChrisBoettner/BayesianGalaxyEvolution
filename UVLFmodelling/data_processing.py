#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 12:10:37 2021

@author: boettner
"""

import numpy as np

## MAIN FUNCTION
def load_data():
    # get z=0,1,2,3,4 for Madau
    madau       = np.load('data/Madau2014UVLF.npz')
    madau = {i:madau[j] for i,j in [['0','1'],['1','4'],['2','7'],['3','8'], ['4','9']]}
    duncan      = np.load('data/Duncan2014UVLF.npz')      
    bouwens     = np.load('data/Bouwens2015UVLF.npz')       
    bouwens2    = np.load('data/Bouwens2021UVLF.npz')
    oesch       = np.load('data/Oesch2010UVLF.npz')
    parsa       = np.load('data/Parsa2016UVLF.npz')
    bhatawdekar = np.load('data/Bhatawdekar2019UVLF.npz')
    atek        = np.load('data/Atek2018UVLF.npz')
    livermore   = np.load('data/Livermore2017UVLF.npz')
    wyder       = np.load('data/Wyder2005UVLF.npz')
    arnouts     = np.load('data/Arnouts2005UVLF.npz')
    reddy       = np.load('data/Reddy2009UVLF.npz')
    oesch2      = np.load('data/Oesch2018UVLF.npz')
    hmfs        = np.load('data/HMF.npz'); hmfs = [hmfs[str(i)] for i in range(20)] 

    ## TURN DATA INTO GROUP OBJECTS, INCLUDING PLOT PARAMETER
    madau       = group(madau,       range(0,5)  ).plot_parameter('black', 'o', 'Cucciati2012')
    duncan      = group(duncan,      range(4,8)  ).plot_parameter('black', 'v', 'Duncan2014')
    bouwens     = group(bouwens,     range(4,9)  ).plot_parameter('black', 's', 'Bouwens2015')
    bouwens2    = group(bouwens2,    range(2,11) ).plot_parameter('black', '^', 'Bouwens2021')
    oesch       = group(oesch,       range(1,3)  ).plot_parameter('black', 'X', 'Oesch2010')
    atek        = group(atek,        range(6,7)  ).plot_parameter('black', 'o', 'Atek2018')
    bhatawdekar = group(bhatawdekar, range(6,10) ).plot_parameter('black', '<', 'Bhatawdekar2019')
    parsa       = group(parsa,       range(2,5)  ).plot_parameter('black', '>', 'Parsa2016')
    livermore   = group(livermore,   range(6,9)  ).plot_parameter('black', 'H', 'Livermore2017')
    wyder       = group(wyder,       range(0,1)  ).plot_parameter('black', '+', 'Wyder2005')
    arnouts     = group(arnouts,     range(0,1)  ).plot_parameter('black', 'd', 'Arnouts2005')
    reddy       = group(reddy,       range(2,4)  ).plot_parameter('black', 'D', 'Reddy2009')
    oesch2      = group(oesch2,      range(10,11)).plot_parameter('black', 'x', 'Oesch2018')
    groups      = [madau, duncan, bouwens, bouwens2, oesch, atek, bhatawdekar, parsa, livermore,
                   wyder, arnouts, reddy, oesch2]
    
    ## DATA SORTED BY REDSHIFT
    lfs = z_ordered_data(groups)
    # undo log for easier fitting, turn magnitudes into luminosities
    raise10 = lambda list_log: [10**list_log[i] for i in range(len(list_log))]
    def lfs_conversion(lfs):
        for lf in lfs:
            lf[:,0] = mag_to_lum(lf[:,0])
            lf[:,1] = raise10(lf[:,1])
        return(lfs)
    lfs  = lfs_conversion(lfs)
    hmfs = raise10(hmfs)
    return(groups, lfs, hmfs)


## DATA CONTAINER CLASS
# class object for group data (all redshifts) = plotting parameter
class group():
     def __init__(self, data, z_range):
        if len(data) != len(z_range):
            raise ValueError('Length of data list does not match assigned\
                              redshift range')
                              
        self._data     = data
        self.redshift  = z_range    
     def plot_parameter(self, color, marker, label):
        self.color     = color
        self.marker    = marker
        self.label     = label
        return(self)
     
     # turn data for specific redshift into dataset object
     def data_at_z(self, redshift):
        ind = self.redshift.index(redshift) # get index for data file for specific redshift
        return(dataset(self._data[str(ind)]))

# object that contains all data for a given dataset (specified group + redshift)
# important: cuts data with number density < 10^6
class dataset():
    def __init__(self, dataset):
        dataset = dataset[dataset[:,1]>-6] # cut unreliable values
        if dataset.shape[1] == 4:
            self.data          = dataset
            self.mag           = dataset[:,0]
            self.lum           = mag_to_lum(dataset[:,0])
            self.phi           = dataset[:,1]
            self.lower_error   = dataset[:,2]
            self.upper_error   = dataset[:,3] 
        if dataset.shape[1] == 2:      
            self.data          = dataset
            self.mag           = dataset[:,0]
            self.lum           = mag_to_lum(dataset[:,0])
            self.phi           = dataset[:,1]
            self.lower_error   = None
            self.upper_error   = None

## HELP FUNCTIONS
# order data from different group to redshift bins and return as list,
def z_ordered_data(groups):
    # create arrays with all datasets and all corresponding redshifts
    # also turn from log
    all_data = []; all_z = []
    for g in groups:
        all_z.append(g.redshift)
        for i in range(len(g.redshift)):
            all_data.append(g._data[str(i)][:,:2]) # remove error data
    all_data = np.array(all_data, dtype=object)
    all_z = np.array([item for sublist in all_z for item in sublist]) # flatten nested list
    
    # sort by redshift
    smfs = []
    for i in range(min(all_z),max(all_z)+1):
        z_mask = (all_z==i)
        data_at_z = all_data[z_mask]
        smfs.append(np.concatenate(data_at_z))
    return(smfs)


# convert between luminosity and abosulte magntidue (luminosity given in 
#  ergs s^-1 Hz^-1)
def lum_to_mag(L_nu):
    d    = 3.086e+19                # 10pc in cm
    flux = L_nu/(4*np.pi*d**2)
    M_uv = -2.5*np.log10(flux)-48.6 # definition in AB magnitude system
    return(M_uv)

def mag_to_lum(M_uv):
    d     = 3.086e+19                # 10pc in cm
    log_L = (M_uv + 48.6)/(-2.5) + np.log10(4*np.pi*d**2) 
    return(np.power(10,log_L))