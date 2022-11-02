#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 16:16:39 2022

@author: chris
"""

from scipy.stats import norm, cauchy

def scatter_model(loc, scale, scatter_name):
    '''
    Select model distribution for the scatter in the quantity - halo mass 
    relation. Distributions are specified through location and scale parameter.
    Currently all distributions are in log space rather than linear
    space.
    Available distributions are
        normal: 
            Gaussian distribution, loc is the expectation value,scale is the
            standard deviation.
        cauchy:
            Cauchy distribution, characteristic for its heavy tails. Better
            model to include large outliers. Pathological distribution in the
            sense that expectation value and variance are not defined. For
            integral (also those for our model) to converge, some cutoff must
            be set for the distribution (done in the relevant method in the
            ModelResult class). loc is the mode/median, scale describes spread.
    
    '''
    
    if scatter_name == 'normal':
        scatter = norm(loc=loc, scale=scale)
    elif scatter_name == 'cauchy':
        raise NotImplementedError(
                'cauchy doesnt quite work yet because of covergence issues \n'
                'you should probably implement your idea with the switch to '
                'a gaussian. Also introduce that lower halo mass cutoff you '
                'mentioned in the paper.')
        scatter = cauchy(loc=loc, scale=scale)
    else:
        raise NotImplementedError('scatter_name not known')
    return(scatter)