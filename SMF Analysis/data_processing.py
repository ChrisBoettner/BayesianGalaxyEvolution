#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 12:10:37 2021

@author: boettner
"""

import numpy as np

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


