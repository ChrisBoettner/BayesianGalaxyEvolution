#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 12:10:37 2021

@author: boettner
"""

import numpy as np

## MAIN FUNCTION
def load_data():
    # get z=0,1,2,3,4 for Davidson, z=1,2,3 for Ilbert
    davidson    = np.load('data/SMF/Davidson2017SMF.npz') 
    davidson = {i:davidson[j] for i,j in [['0','1'],['1','2'],['2','4'],['3','6'],['4','8']]}
    ilbert      = np.load('data/SMF/Ilbert2013SMF.npz')   
    ilbert = {i:ilbert[j] for i,j in [['0','0'],['1','2'],['2','4'],['3','6']]}
    duncan      = np.load('data/SMF/Duncan2014SMF.npz')      
    song        = np.load('data/SMF/Song2016SMF.npz')       
    bhatawdekar = np.load('data/SMF/Bhatawdekar2018SMF.npz')
    stefanon    = np.load('data/SMF/Stefanon2021SMF.npz')
    hmfs        = np.load('data/HMF.npz'); hmfs = [hmfs[str(i)] for i in range(20)]   

    ## TURN DATA INTO GROUP OBJECTS, INCLUDING PLOT PARAMETER
    davidson    = group(davidson,    [0,1,2,3,4]   ).plot_parameter('black', 'o', 'Davidson2017')
    ilbert      = group(ilbert,      [0,1,2,3]     ).plot_parameter('black', 'H', 'Ilbert2013')
    duncan      = group(duncan,      [4,5,6,7]   ).plot_parameter('black', 'v', 'Duncan2014')
    song        = group(song,        [6,7,8]     ).plot_parameter('black', 's', 'Song2016')
    bhatawdekar = group(bhatawdekar, [6,7,8,9]   ).plot_parameter('black', '^', 'Bhatawdekar2019')
    stefanon    = group(stefanon,    [6,7,8,9,10]).plot_parameter('black', 'X', 'Stefanon2021')
    groups      = [davidson, ilbert, duncan, song, bhatawdekar, stefanon]

    ## DATA SORTED BY REDSHIFT
    smfs = z_ordered_data(groups)
    # undo log for easier fitting
    raise10 = lambda list_log: [10**list_log[i] for i in range(len(list_log))]
    smfs = raise10(smfs)
    hmfs = raise10(hmfs)
    return(groups, smfs, hmfs)


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
            self.mass          = dataset[:,0]
            self.phi           = dataset[:,1]
            self.lower_error   = dataset[:,2]
            self.upper_error   = dataset[:,3] 
        if dataset.shape[1] == 2:      
            self.data          = dataset
            self.mass          = dataset[:,0]
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


