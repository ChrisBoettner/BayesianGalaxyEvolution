#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 13:09:05 2022

@author: chris
"""
import matplotlib.pyplot as plt
from model.interface import load_model, run_model
from model.plotting.plotting import *

def save_plots(quantity, show = False, file_format='pdf'):
    '''
    Make and save all plots for a given quantity. 
    Choose if plots are displayed using show parameter. Default is False.
    Choose file format with file_format. Default is 'pdf'.
    '''
    if show:
        plt.ion() 
    else:
        plt.ioff()
    
    if quantity in ['mstar' , 'Muv']: 
        model = load_model(quantity, 'changing')
        
        Plot_ndf_intervals(model, sigma=[1,2,3]).save(file_format)
        
        if quantity == 'mstar':
            # feedback illustration plot
            redshift = 0
            mstar_no = run_model('mstar', 'none', 
                                 fitting_method='least_squares',
                                 redshift=redshift)
            mstar_st = run_model('mstar', 'stellar', 
                                 fitting_method='least_squares',
                                 redshift=redshift)
            mstar_sb = run_model('mstar', 'stellar_blackhole', 
                                 fitting_method='least_squares',
                                 redshift=redshift)
            Plot_best_fit_ndf([mstar_no, mstar_st, mstar_sb],
                              columns='single').save()
            
            del mstar_no; del mstar_st; del mstar_sb
        del model

    elif quantity == 'mbh':
        model = load_model(quantity, 'quasar')
        
        Plot_ndf_intervals(model, sigma=[1,2,3]).save()
        del model
        
    elif quantity == 'Lbol':
        model = load_model(quantity, 'eddington')       
        
        Plot_ndf_intervals(model, sigma=[1,2,3], num=1000).save(file_format)
        
        # conditional ERDF example
        Plot_conditional_ERDF(model, parameter = [40 ,  2, -2,  2],
                              columns='single').save()
        # black hole mass distribution
        Plot_black_hole_mass_distribution(model, columns='single').save()
        del model
        
    elif quantity == 'mstar_mbh':
        mstar  = load_model('mstar','changing')
        mbh    = load_model('mbh','quasar', prior_name='successive')
        Plot_q1_q2_relation(
                    mstar,mbh,datapoints=True,sigma=[1],
                    scaled_ndf=(mbh, [10,30,100]),
                    quantity_range=np.linspace(8.7,11.9,100)).save(file_format)
        del mstar
        del mbh
        
    elif quantity == 'Lbol_mbh':
        lbol = load_model('Lbol', 'eddington', prior_name='successive')
        mbh  = load_model('mbh','quasar', prior_name='successive')
        Plot_q1_q2_relation(lbol, mbh, scaled_ndf=(mbh, 3),
                    quantity_range=np.linspace(43,48,100)).save(file_format)
        del lbol
        del mbh
        
    else:
        raise ValueError('quantity_name not known.')
        
    if not show:
        plt.close('all')
        plt.ion()
    return

if __name__ == '__main__':
    quantities = ['mstar', 'Muv', 'mbh', 'Lbol', #'mstar_mbh','Lbol_mbh'
                  ]
    [save_plots(quantity) for quantity in quantities]
    