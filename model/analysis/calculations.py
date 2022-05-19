#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:11:04 2022

@author: chris
"""
import numpy as np

from model.helper import calculate_percentiles, make_list

################ MAIN FUNCTIONS ###############################################


def calculate_qhmr(ModelResult, log_m_halos, redshifts):
    '''
    Calculates the quantity distribution for an array of input halo masses
    for different redshifts. Returns dictonary of form (redshift:qhmr), where
    the qhmr is an array contains median, 16th and 84th percentile for every
    halo mass.
    '''
    redshifts = make_list(redshifts)
    log_m_halos = make_list(log_m_halos)

    qhmr = {}
    for z in redshifts:
        percentiles_at_z = []
        # calculate percentiles for every halo mass
        for m_h in log_m_halos:
            # calculate distribution
            q_dist = ModelResult.calculate_quantity_distribution(
                m_h, z, int(1e+4))
            # calculate percentiles and add to list
            percentiles_at_z.append(calculate_percentiles(q_dist))
        qhmr[z] = np.array(percentiles_at_z)[:,:,0]
    return(qhmr)


def calculate_best_fit_ndf(ModelResult, redshifts, quantity_range=None):
    '''
    Calculate best fit number density function by passing calculated best fit
    parameter to calculate_ndf method. Returns array of form {redshift:ndf}
    '''

    redshifts = make_list(redshifts)
    best_fit_ndfs = {}
    for z in redshifts:
        quantity, phi = ModelResult.calculate_ndf(
            z, ModelResult.parameter.at_z(z),
            quantity_range=quantity_range)
        best_fit_ndfs[z] = np.array([quantity, phi]).T
    return(best_fit_ndfs)
