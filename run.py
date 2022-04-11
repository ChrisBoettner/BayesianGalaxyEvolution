#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 13:48:39 2022.

@author: chris
"""
import click

from model.api import save_model

@click.command()
@click.option('--quantity', prompt='Physical quantity',
              help='Physical quantity: \'Muv\' or \'mstar\'')
@click.option('--feedback', prompt='Feedback model',
              help='Feedback model: \'none\', \'stellar\' \'stellar_blackhole\' or custom')
def run(quantity_name, feedback_name):
    ''''Runs save_model with choosen options'''
    return(save_model(quantity_name, feedback_name))

if __name__ == '__main__':
    model = run()

