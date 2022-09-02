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
        # successive prior
        model = load_model(quantity, 'changing')
        
        Plot_ndf_sample(model).save(file_format)
        Plot_marginal_pdfs(model).save(file_format)
        Plot_parameter_sample(model).save(file_format)
        if quantity == 'mstar':
            Plot_qhmr(model).save()
        del model
        
        # uniform priors
        models = [load_model(quantity,'none'), load_model(quantity,'stellar'), 
                  load_model(quantity,'stellar_blackhole')]
        Plot_best_fit_ndfs(models).save(file_format)
        Plot_marginal_pdfs(models).save(file_format)
        
    elif quantity == 'Lbol':
        # successive prior
        model = load_model(quantity, 'eddington', prior_name='successive')
        
        # added for plotting
        lbol_free = run_model(quantity, 'eddington_free_ERDF')
        
        Plot_best_fit_ndfs([model, lbol_free]).save(file_format)
        Plot_ndf_sample(model).save(file_format)
        Plot_marginal_pdfs(model).save(file_format)
        Plot_parameter_sample(model).save(file_format)
        Plot_conditional_ERDF(model).save(file_format)
        del model
        
        
    elif quantity == 'mbh':
        # successive prior
        model = load_model(quantity, 'quasar', prior_name='successive')
        
        # added for plotting
        mbh_none = run_model(quantity, 'none')
        
        Plot_best_fit_ndfs([model, mbh_none]).save(file_format)
        Plot_ndf_sample(model).save(file_format)
        Plot_marginal_pdfs(model).save(file_format)
        Plot_parameter_sample(model).save(file_format)
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
        
        
    else:
        raise ValueError('quantity_name not known.')
        
    if not show:
        plt.close('all')
        plt.ion()
    return

if __name__ == '__main__':
    quantities = ['mstar', 'Muv', 'mbh', 'Lbol', 'mstar_mbh']
    [save_plots(quantity) for quantity in quantities]
    