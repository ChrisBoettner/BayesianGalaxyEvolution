#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 13:48:39 2022.

@author: chris
"""
import click

from model.interface import save_model


@click.command()
@click.option('--quantity_name', prompt='Physical quantity',
              help='Physical quantity: \'Muv\', \'mstar\', \'Lbol\' or \'mbh\'')
@click.option('--physics_name', prompt='Physics model',
              help='Physics model: \'none\', \'stellar\', \'stellar_blackhole\,\
                  \'changing\', \'quasar\', \'eddington\', \'eddington_free_ERDF\' or custom')
@click.option('--data_subset', default=None,
              help='List of data sets names (of form AuthorYear')
@click.option('--prior_name', default=None,
              help='Prior model: \'uniform\' or \'successive\'')
@click.option('--redshift', default=None,
              help='Choose list of redshift to include.')
@click.option('--parameter_calc', default=True,
              help='Choose if best fit parameter should be calculated. True/False.')
@click.option('--min_chain_length', default=20000,
              help='Length of MCMC chains.')
@click.option('--num_walker', default=250,
              help='Number of MCMC walker in ensemble.')
def run(**kwargs):
    ''''Runs save_model with choosen options'''
    return(save_model(**kwargs))


if __name__ == '__main__':
    model = run()
