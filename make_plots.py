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
        model = load_model(quantity, 'stellar_blackhole')
        
        Plot_ndf_intervals(model, sigma=[1,2,3]).save(file_format)
        Plot_parameter_sample(model, columns='single',
                              marginalise=(model.
                                           quantity_options
                                           ['feedback_change_z'], 
                                           [1,2])).save(file_format)
        
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
                              columns='single').save(file_format)
            
            # SHMR
            Plot_qhmr(model, redshift=[0,1,2], sigma=2, columns='single').\
                save(file_format, file_name=quantity + '_qhmr_low_z')
            Plot_qhmr(model, redshift=[4,5,6,7,8], sigma=2, columns='single',
                      only_data=True, m_halo_range=np.linspace(10, 13.43, 1000)
                      ).save(file_format, file_name=quantity + '_qhmr_high_z')
            del mstar_no; del mstar_st; del mstar_sb
        del model

    elif quantity == 'mbh':
        model = load_model(quantity, 'quasar')
        
        Plot_ndf_intervals(model, sigma=[1,2,3]).save(file_format)
        Plot_parameter_sample(model, columns='single').save(file_format)
        del model
        
    elif quantity == 'Lbol':
        model = load_model(quantity, 'eddington')       
        
        Plot_ndf_intervals(model, sigma=[1,2,3], num=1000).save(file_format)
        Plot_parameter_sample(model, columns='single').save(file_format)
        
        # conditional ERDF example
        Plot_conditional_ERDF(model, parameter = [40 ,  2, -2,  2],
                              columns='single').save(file_format)
        # black hole mass distribution
        Plot_black_hole_mass_distribution(model, 
                                          columns='single').save(file_format)
        del model
        
    elif quantity == 'Muv_mstar':
        muv    = load_model('Muv','stellar_blackhole', redshift=range(4,9))
        mstar  = load_model('mstar','stellar_blackhole', redshift=range(4,9))
        # plot for multiple redshifts
        for z in range(4,8):
            Plot_q1_q2_relation(muv, mstar, z=z, datapoints=True,
                                quantity_range=np.linspace(-22.24,-17.23,100),
                                y_lims=(7.5,11.9), columns='single',
                                sigma = 2, 
                                ).save(file_format, 
                                       file_name=(quantity + '_relation_z' 
                                                  + str(z)))
        del muv
        del mstar
        
    elif quantity == 'mstar_mbh':
        mstar  = load_model('mstar','stellar_blackhole', redshift=0)
        mbh    = load_model('mbh','quasar', redshift=0)
        Plot_q1_q2_relation(mstar,mbh,datapoints=True,
                            scaled_ndf=(mbh, [10,30,100]),
                            quantity_range=np.linspace(8.7,11.9,100),
                            sigma=2,
                            columns='single').save(file_format)
        del mstar
        del mbh
        
    elif quantity == 'Lbol_mbh':
        lbol = load_model('Lbol', 'eddington', redshift=0)
        mbh  = load_model('mbh','quasar', redshift=0)
        Plot_q1_q2_relation(lbol, mbh, columns='single', color='lightgrey',
                    quantity_range=np.linspace(43,48.3,100),
                    sigma=2).save(file_format)
        del lbol
        del mbh
        
    else:
        raise ValueError('quantity_name not known.')
        
    if not show:
        plt.close('all')
        plt.ion()
    return

if __name__ == '__main__':
    quantities = ['mstar', 'Muv', 'mbh', 'Lbol', 'Muv_mstar',
                  'Lbol_mbh', 'mstar_mbh']
    [save_plots(quantity) for quantity in quantities]
    