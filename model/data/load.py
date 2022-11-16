#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 12:41:29 2022

@author: chris
"""
import numpy as np
from scipy.interpolate import interp1d

from model.helper import make_list
################ MAIN FUNCTIONS ###############################################
path = 'model/data/'

def load_hmf_functions(source='ShethTormen', redshift=range(20)):
    '''
    Load HMF data and transform it to callable function by interpolating
    the data. Either load Pratikas data, or self-created Sheth-Tormen HMF using
    hmf module. Get for redshifts 0-20.
    If ShethTormen, also return dicts of turnover mass and total number of 
    halos.
    '''

    # either load or create HMFs (self-created ones have larger range)
    if source == 'ShethTormen':
        hmfs                    = np.load(path + 'HMF/HMF.npz')
        log_turnover_halo_mass  = np.load(path + 'HMF/turnover_mass.npz')
        total_halo_number       = np.load(path + 'HMF/total_halo_number.npz')
    elif source == 'Pratika':
        hmfs = np.load(path + 'HMF/HMF_Pratika.npz')
    else:
        raise NameError('HMF source not known.')

    # create evaluatable function from data
    hmf_functions = []
    for z in redshift:
        h = hmfs[str(z)]
        hmf_functions.append(interp1d(*h.T, bounds_error=False,
                                      fill_value=-np.inf))
    # turn into dictonary
    hmf_functions = {z: hmf_functions[z] for z in redshift}
    if source == 'ShethTormen':
        log_turnover_halo_mass = {z: log_turnover_halo_mass[str(z)] 
                                  for z in redshift}
        total_halo_number      = {z: total_halo_number[str(z)] 
                                  for z in redshift} 
        return(hmf_functions, log_turnover_halo_mass, total_halo_number)
    return(hmf_functions)


def load_ndf_data(quantity_name, cutoff, data_subset=None):
    '''
    Wrapper function to load datasets.

    Can take optional argument that for selecting specific datasets, input dataset
    name or list of names of form AuthorYear. (e.g. 'Song2016' or
    ['Davidson2017', 'Duncan2014'])

    IMPORTANT:  number densities (phi values) below threshold value are cut off
                because they can't be measured reliably (default is 10^(-6) for
                'mstar' and 'Muv', -np.inf for 'Lbol', 'mbh')
    '''
    if quantity_name == 'mstar':
        groups, data = _load_smf_data(cutoff=cutoff, data_subset=data_subset)
    elif quantity_name == 'Muv':
        groups, data = _load_uvlf_data(cutoff=cutoff, data_subset=data_subset)
    elif quantity_name == 'Lbol':
        groups, data = _load_qlf_data(cutoff=cutoff, data_subset=data_subset)
    elif quantity_name == 'mbh':
        groups, data = _load_bhmf_data(cutoff=cutoff, data_subset=data_subset)
    else:
        raise NameError('quantity_name not known.')
    return(groups, data)

def load_data_points(quantity_name):
    '''
    Load specific data sets for individual data that are not ndfs. 
    Currently included are
    
    mstar_mbh : Stellar mass - Black hole mass relation
       Returns numpy array of form [log_stellar_mass, log_bh_mass, index],
       where index contains information about the data source,
       0 : Baron2019
       1 : Reines2015
       2 : Bentz2018
       
    mbh_lbol : Black hole mass - Bolometric luminosity relation
       Returns numpy array of form [log_bh_mass, log_l_bol].
       
    Muv_mstar : UV luminosity - stellar mass relation. 
        Contains data from Bhatawdekar (z=6-9) and Song (z=4-8).
        Returns dictonaries of form {z: [Muv, log_mstar, index]}, where
        where index contains information about the data source,
        0 : Song2016
        1 : Bhatawdekar2019
    '''
    if quantity_name == 'mstar_mbh':
        data = np.load(path + '/BHmass/mstar_mbh_Baron2019.npy')
    elif quantity_name == 'mbh_Lbol':
        data = np.load(path + '/BHmass/mbh_Lbol_Baron2019.npy')
    elif quantity_name == 'Muv_mstar':
        # Bhatawdekar data
        data = np.load(path + '/Mainsequence/Muv_mstar_Song2016.npz')
        data = {z:data[str(z)] for z in range(4,9)}
    else:
        raise NameError('quantity_name not known.')
    return(data)

def load_JWST_UVLF_points():
    '''
    Load JWST UVLF. Currently included are Harrikane2022 (z=12, 17)
    and Donnan2022 (z=13).
    '''
    data = np.load(path + '/UVLF/JWST_UVLF.npz')
    data = {int(z):data[z] for z in ['12','13','17']}
    return(data)

################ NUMBER DENSITY FUNCTION DATASETS #############################

def _load_smf_data(cutoff, data_subset):
    '''
    Load the SMF data. Returns list of group objects that contain the data 
    connected to individual groups, and a direct directory of SMF ordered by 
    redshift (of the form {redshift:data}).
    Remove data with number density below some cutoff limit.
    '''
    davidson = dict(np.load(path + 'SMF/Davidson2017SMF.npz'))
    ilbert = dict(np.load(path + 'SMF/Ilbert2013SMF.npz'))
    duncan = dict(np.load(path + 'SMF/Duncan2014SMF.npz'))
    song = dict(np.load(path + 'SMF/Song2016SMF.npz'))
    bhatawdekar = dict(np.load(path + 'SMF/Bhatawdekar2018SMF.npz'))
    stefanon = dict(np.load(path + 'SMF/Stefanon2021SMF.npz'))
    tomczak = dict(np.load(path + 'SMF/Tomczak2014SMF.npz'))
    baldry = dict(np.load(path + 'SMF/Baldry2012SMF.npz'))
    moustakas = dict(np.load(path + 'SMF/Moustakas2013SMF.npz'))

    # TURN DATA INTO GROUP OBJECTS, INCLUDING PLOT PARAMETER
    davidson = Group(davidson, [0, 1, 2, 3, 4], cutoff).plot_parameter(
        'black', 'o', 'Davidzon2017')
    ilbert = Group(ilbert, [0, 1, 2, 3], cutoff).plot_parameter(
        'black', 'H', 'Ilbert2013')
    duncan = Group(duncan, [4, 5, 6, 7], cutoff).plot_parameter(
        'black', 'v', 'Duncan2014')
    song = Group(song, [4, 5, 6, 7, 8], cutoff).plot_parameter(
        'black', 's', 'Song2016')
    bhatawdekar = Group(bhatawdekar, [6, 7, 8, 9], cutoff).plot_parameter(
        'black', '^', 'Bhatawdekar2019')
    stefanon = Group(stefanon, [6, 7, 8, 9, 10], cutoff).plot_parameter(
        'black', 'X', 'Stefanon2021')
    tomczak = Group(tomczak, [0,1,2,3], cutoff).plot_parameter(
        'black', '+', 'Tomczak2014')
    baldry = Group(baldry, [0], cutoff).plot_parameter(
        'black', 'd', 'Baldry2012')
    moustakas = Group(moustakas, [0,1], cutoff).plot_parameter(
        'black', 'D', 'Moustakas2013')
    groups = [davidson, ilbert, duncan, song, bhatawdekar, stefanon,
              baldry, moustakas, tomczak]

    # choose subselection of data if given when calling the function
    if data_subset:
        data_subset = make_list(data_subset)
        groups = {g.label: g for g in groups}
        try:
            groups = [groups[dataset] for dataset in data_subset]
        except BaseException:
            raise KeyError('dataset not in groups')

    # DATA SORTED BY REDSHIFT
    smfs = z_ordered_data(groups)
    return(groups, smfs)


def _load_uvlf_data(cutoff, data_subset):
    '''
    Load the UVLF data. Returns list of group objects that contain the data
    connected to individual groups, and a direct directory of SMF ordered by 
    redshift (of the form {redshift:data}).
    Remove data with number density below some cutoff limit.
    '''
    # remove redshift 10 for Bouwens2021, since this is same data as Oesch2018,
    # also only load Bouwens2021, since Bouwens2015 contains same data
    cucciati = dict(np.load(path + 'UVLF/Cucciati2012UVLF.npz'))
    duncan = dict(np.load(path + 'UVLF/Duncan2014UVLF.npz'))
    #bouwens = dict(np.load(path + 'UVLF/Bouwens2015UVLF.npz'))
    bouwens2 = dict(np.load(path + 'UVLF/Bouwens2021UVLF.npz'))
    bouwens2 = {str(i): bouwens2[str(i)] for i in range(8)}
    oesch = dict(np.load(path + 'UVLF/Oesch2010UVLF.npz'))
    parsa = dict(np.load(path + 'UVLF/Parsa2016UVLF.npz'))
    bhatawdekar = dict(np.load(path + 'UVLF/Bhatawdekar2019UVLF.npz'))
    atek = dict(np.load(path + 'UVLF/Atek2018UVLF.npz'))
    livermore = dict(np.load(path + 'UVLF/Livermore2017UVLF.npz'))
    wyder = dict(np.load(path + 'UVLF/Wyder2005UVLF.npz'))
    arnouts = dict(np.load(path + 'UVLF/Arnouts2005UVLF.npz'))
    reddy = dict(np.load(path + 'UVLF/Reddy2009UVLF.npz'))
    oesch2 = dict(np.load(path + 'UVLF/Oesch2018UVLF.npz'))

    # TURN DATA INTO GROUP OBJECTS, INCLUDING PLOT PARAMETER
    cucciati = Group(cucciati, range(0, 5), cutoff).plot_parameter(
        'black', 'o', 'Cucciati2012')
    duncan = Group(duncan,range(4, 8), cutoff).plot_parameter(
            'black', 'v', 'Duncan2014')
    #bouwens = Group(bouwens, range(4, 9), cutoff).plot_parameter(
    #        'black', 's', 'Bouwens2015')
    bouwens2 = Group(bouwens2, range(2, 10), cutoff).plot_parameter(
        'black', '^', 'Bouwens2021')
    oesch = Group(oesch, range(1, 3), cutoff).plot_parameter(
            'black', 'X', 'Oesch2010')
    atek = Group(atek, range(6, 7), cutoff).plot_parameter(
            'black', 'o', 'Atek2018')
    bhatawdekar = Group(bhatawdekar, range(6, 10), cutoff).plot_parameter(
        'black', '<', 'Bhatawdekar2019')
    parsa = Group(parsa, range(2, 5), cutoff).plot_parameter(
            'black', '>', 'Parsa2016')
    livermore = Group(livermore, range(6, 9), cutoff).plot_parameter(
        'black', 'H', 'Livermore2017')
    wyder = Group(wyder, range(0, 1), cutoff).plot_parameter(
            'black', '+', 'Wyder2005')
    arnouts = Group(arnouts, range(0, 1), cutoff).plot_parameter(
            'black', 'd', 'Arnouts2005')
    reddy = Group(reddy, range(2, 4), cutoff).plot_parameter(
            'black', 'D', 'Reddy2009')
    oesch2 = Group(oesch2, range(10, 11), cutoff).plot_parameter(
            'black', 'x', 'Oesch2018')
    groups = [cucciati, duncan, bouwens2, oesch, atek, bhatawdekar,
              parsa, livermore, wyder, arnouts, reddy, oesch2]

    # choose subselection of data if given when calling the function
    if data_subset is not None:
        data_subset = make_list(data_subset)
        groups = {g.label: g for g in groups}
        try:
            groups = [groups[dataset] for dataset in data_subset]
        except BaseException:
            raise KeyError('dataset not in groups')

    # DATA SORTED BY REDSHIFT
    lfs = z_ordered_data(groups)
    return(groups, lfs)

def _load_qlf_data(cutoff, data_subset):
    '''
    Load the QLF data. Returns list of group objects that contain the data 
    connected to individual groups, and a direct directory of SMF ordered by 
    redshift (of the form {redshift:data}).
    Remove data with number density below some cutoff limit.
    '''
    # get z=0,1,2,3,4 for Davidson, z=1,2,3 for Ilbert
    shen = dict(np.load(path + 'QLF/Shen2020QLF.npz'))
    
    # TURN DATA INTO GROUP OBJECTS, INCLUDING PLOT PARAMETER
    shen = Group(shen, range(0,8), cutoff).plot_parameter(
        'black', 'o', 'Shen2020')
    groups = [shen]

    # choose subselection of data if given when calling the function
    if data_subset:
        data_subset = make_list(data_subset)
        groups = {g.label: g for g in groups}
        try:
            groups = [groups[dataset] for dataset in data_subset]
        except BaseException:
            raise KeyError('dataset not in groups')

    # DATA SORTED BY REDSHIFT
    qlfs = z_ordered_data(groups)
    return(groups, qlfs)

def _load_bhmf_data(cutoff, data_subset):
    '''
    Load the BHLF data. Returns list of group objects that contain the data 
    connected to individual groups, and a direct directory of SMF ordered by 
    redshift (of the form {redshift:data}).
    Remove data with number density below some cutoff limit.
    '''
    zhang = dict(np.load(path + 'BHMF/Zhang2021BHMF.npz'))
    
    # TURN DATA INTO GROUP OBJECTS, INCLUDING PLOT PARAMETER
    zhang = Group(zhang, range(0,6), cutoff).plot_parameter(
        'black', 'o', 'Zhang2021')
    groups = [zhang]

    # choose subselection of data if given when calling the function
    if data_subset:
        data_subset = make_list(data_subset)
        groups = {g.label: g for g in groups}
        try:
            groups = [groups[dataset] for dataset in data_subset]
        except BaseException:
            raise KeyError('dataset not in groups')

    # DATA SORTED BY REDSHIFT
    bhmfs = z_ordered_data(groups)
    return(groups, bhmfs)

################ CLASSES ######################################################


# class object for group data (all redshifts) + plotting parameter
class Group():
    def __init__(self, data, redshift, cutoff=None):
        if len(data) != len(redshift):
            raise ValueError('Length of data list does not match assigned\
                              redshift range')

        # cutoff data below certain phi threshold
        if cutoff:
            for key in data.keys():
                d = data[key]
                data[key] = d[d[:, 1] > cutoff]

        self._data = data
        self.redshift = redshift

    def plot_parameter(self, color, marker, label):
        self.color = color
        self.marker = marker
        self.label = label
        return(self)

    # turn data for specific redshift into dataset object
    def data_at_z(self, redshift):
        # get index for data file for specific redshift
        ind = self.redshift.index(redshift)
        return(Dataset(self._data[str(ind)]))

# object that contains all data for a given dataset (specified group +
# redshift)


class Dataset():
    def __init__(self, dataset):
        if dataset.shape[1] == 4:
            self.data = dataset
            self.quantity = dataset[:, 0]
            self.phi = dataset[:, 1]
            self.lower_error = dataset[:, 2]
            self.upper_error = dataset[:, 3]
        elif dataset.shape[1] == 2:
            self.data = dataset
            self.quantity = dataset[:, 0]
            self.phi = dataset[:, 1]
            self.lower_error = None
            self.upper_error = None
        else:
            raise ValueError('Don\'t know how to handle Group Dataset shape.')

################ HELP FUNCTIONS ###############################################


def z_ordered_data(groups):
    '''
    Sort data from list of groups by redshift and return dictonary of form
    redshift:data.
    '''
    # get all datasets and associated redshifts
    all_data = []
    all_z = []
    for g in groups:
        all_z.append(g.redshift)
        for i in range(len(g.redshift)):
            all_data.append(g._data[str(i)])
    all_data = np.array(all_data, dtype=object)
    # flatten nested list
    all_z = np.array([item for sublist in all_z for item in sublist])

    # sort by redshift
    smfs = []
    for i in range(min(all_z), max(all_z) + 1):
        z_mask = (all_z == i)
        data_at_z = np.concatenate(all_data[z_mask])
        data_at_z = data_at_z[data_at_z[:, 0].argsort()]  # sort data
        smfs.append(data_at_z)

    # get unique list of redshifts and connect turn into dictonary
    redshift = sorted(list(set(all_z)))
    smfs = {redshift[i]: smfs[i] for i in range(len(redshift))}
    return(smfs)
