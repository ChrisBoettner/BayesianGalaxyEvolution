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
        
        if model=='mstar':
            redshift = np.arange(0, model.quantity_options
                                 ['extrapolation_end']+1)
        else:
            redshift = np.arange(4, model.quantity_options
                                 ['extrapolation_end']+1)
        
        Plot_quantity_density_evolution(model, redshift=redshift,
                        log_q_space = model.quantity_options['density_bounds'],
                        columns='single', datapoints=True, 
                        legend=True).save(file_format)
        
        if quantity == 'mstar':
            quantity_range = np.linspace(6.1,10.13,100)
        else:
            quantity_range = np.linspace(-24.41, -18.01, 100)
        Plot_ndf_predictions(model, quantity_range=quantity_range,
                             y_lim=[-6.95,0]).save(file_format)
        
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
            # influence of scatter
            Plot_scatter_ndf(model, columns='single').save(file_format)
            
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
        Plot_ndf_predictions(model, y_lim=[-14.9,0]).save(file_format)
        Plot_quantity_density_evolution(model, 
                        log_q_space = model.quantity_options['density_bounds'],
                        columns='single').save(file_format)
        del model
        
    elif quantity == 'Lbol':
        model = load_model(quantity, 'eddington')       
        
        Plot_ndf_intervals(model, sigma=[1,2,3], num=1000).save(file_format)
        Plot_parameter_sample(model, columns='single').save(file_format)
        Plot_ndf_predictions(model, y_lim=[-14.9,0], 
                             num=1000).save(file_format)
        Plot_quantity_density_evolution(model,
                        log_q_space = model.quantity_options['density_bounds'],
                        columns='single').save(file_format)
        
        # conditional ERDF example
        Plot_conditional_ERDF(model, parameter = [40 ,  2, -2,  2],
                              columns='single').save(file_format)
        # black hole mass distribution
        Plot_black_hole_mass_distribution(model, 
                                          sigma=2,
                                          columns='single').save(file_format)
        del model
        
    elif quantity == 'Muv_mstar':
        muv    = load_model('Muv','stellar_blackhole')
        mstar  = load_model('mstar','stellar_blackhole')
        # plot main sequence for multiple redshifts
        for z in range(4,8):
            Plot_q1_q2_relation(muv, mstar, z=z, datapoints=True,
                                quantity_range=np.linspace(-22.24,-17.23,100),
                                y_lims=(7.5,11.9), columns='single',
                                sigma = 2, 
                                ).save(file_format, 
                                       file_name=(quantity + '_relation_z' 
                                                  + str(z)))
        # plot stellar mass density
        Plot_stellar_mass_density_evolution(mstar, muv, 
                                            columns='single').save(file_format)
        
        # plot influence of scatter
        Plot_q1_q2_distribution_with_scatter(mstar, muv, log_q2_value=-20,
                            redshift=4, columns='single').save(file_format)
                                       
        del muv
        del mstar
        
    elif quantity == 'mstar_mbh':
        mstar  = load_model('mstar','stellar_blackhole', redshift=0)
        mbh    = load_model('mbh','quasar', redshift=0)
        Plot_q1_q2_relation(mstar,mbh,datapoints=True,
                            scaled_ndf=(mbh, [10,30,100]),
                            scaled_ndf_color = ['grey', 'black', 'grey'],
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
        Plot_black_hole_mass_density_evolution(mbh, lbol, 
                                            columns='single').save('pdf')
        del lbol
        del mbh
        
    else:
        raise ValueError('quantity_name not known.')
        
    if not show:
        plt.close('all')
        plt.ion()
    return

def make_all_plots(show=False, file_format='pdf'):
    ''' 
    Make plots for all available quantities.
    Choose if plots are displayed using show parameter. Default is False.
    Choose file format with file_format. Default is 'pdf'.
    '''
    quantities = ['mstar', 'Muv', 'mbh', 'Lbol', 'Muv_mstar',
                  'Lbol_mbh', 'mstar_mbh']
    [save_plots(quantity, show=show, file_format=file_format) 
     for quantity in quantities]
    return

if __name__ == '__main__':
    make_all_plots()

    
    