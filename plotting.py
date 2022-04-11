#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 21:41:18 2022

@author: chris
"""

make plot functions that take the CalibrationResult (or ModelAnalyis object) as
input and return the plot in turn (make calculations needed for plots in seperate function and put in their own files)

make this file the central one to call any of the plot functions (which should be 
in own folder)

also make stuff written in run.py into function, that only takes some inputs 
(quantity_name, save_mode, save_parameter) and has rest predefined (but changable),
this should be only function you call
-> started already in api.py (include data and groups into CalibrationResult to then
                              pass this to plotting functions)

(remember that you changed A to log_A in feedback_model, either do the same for
 feedback model in analysis package, or make analysis package load the one from calibration,
 (which needs to be adapted then though to work with more than one parameter input))
