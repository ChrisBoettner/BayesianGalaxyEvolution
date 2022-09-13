#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:41:12 2022

@author: chris
"""

from model.interface import load_model, run_model, save_model
from model.plotting.plotting import *

#mstar  = load_model('mstar','changing')
# muv    = load_model('Muv','changing')
mbh    = load_model('mbh','quasar', prior_name='successive')
lbol   = load_model('Lbol', 'eddington', prior_name='successive')