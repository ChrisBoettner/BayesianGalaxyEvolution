#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:41:12 2022

@author: chris
"""
# %%
from model.interface import load_model, run_model, save_model
from model.plotting.plotting import *

import warnings

warnings.filterwarnings("ignore")

# %%
mstar = load_model("mstar", "stellar_blackhole")
muv = load_model("Muv", "stellar_blackhole")
mbh = load_model("mbh", "quasar")
lbol = load_model("Lbol", "eddington")

# %%
for quantity in [
    (mstar, "mstar", "stellar_blackhole"),
    (muv, "Muv", "stellar_blackhole"),
    (mbh, "mbh", "quasar"),
    # (lbol, "Lbol", "eddington"),
]:

    models = []
    for subset in quantity[0].groups:
        try:
            models.append(
                run_model(
                    quantity[1],
                    quantity[2],
                    fitting_method="annealing",
                    data_subset=subset.label,
                )
            )
        except:
            continue

    Plot_ndf_intervals(
        quantity[0],
        sigma=[1, 2, 3],
        datapoints=True,
        additional_models=models,
        additional_model_color="blue",
    ).save("pdf", file_name=quantity[1] + "_ndf_groups")

# %%
# from model.helper import log_L_bol_to_log_L_band, calculate_percentiles
# import numpy as np

# # mh = np.linspace(8,15,100)

# # z = 1
# # num= 1000

# # l = lbol.calculate_quantity_distribution(mh, z=z, num = num)
# # lx = log_L_bol_to_log_L_band(l)
# # ms = mstar.calculate_quantity_distribution(mh, z=z, num=num)

# # def lam (ms, lx):
# #     l = 25* 10**lx
# #     denom = 1.3e+38 * 0.002 * 10**ms
# #     return(l/denom)

# # la = lam(ms,lx)
# # la = np.median(la, axis=0)

# # plt.loglog(10**np.median(ms,axis=0),la)

# data1 = np.array([[9.925, 0.5, 0.3, 1],
#                  [10.255, 0.71, 0.37, 0.94],
#                  [10.585, 3.8, 1.6, 2.9],
#                  [11, 3.7, 1.4, 2.5]])

# data2 = np.array([[9.925, 2.8, 1.3, 2.8],
#                  [10.255, 4, 1.5, 2.9],
#                  [10.585, 12.9, 4.9, 8.8],
#                  [11, 25, 12, 25]])

# data = {1: data1, 2: data2}

# def calc_lx(ms, z, num=10000):
#     mh = mstar.calculate_halo_mass_distribution(ms, z=z, num=num//100)
#     l = lbol.calculate_quantity_distribution(mh, z=z, num=num).flatten()
#     lx = log_L_bol_to_log_L_band(l) - 42
#     print(calculate_percentiles(10**lx, sigma_equiv=2))
#     #return(lx)

# num=10000
# n =10
# zs= [1,2]
# ms = np.linspace(9,11.5,n)

# ls = {}
# plt.close()
# for z in zs:
#     mh = mstar.calculate_halo_mass_distribution(ms, z=z, num=num//100)
#     l = lbol.calculate_quantity_distribution(mh, z=z, num=num).reshape([num*num//100, n])
#     lx = log_L_bol_to_log_L_band(l)
#     percentiles = calculate_percentiles(lx, sigma_equiv=2).T
#     ls[z] = percentiles

#     #percentiles = 10**(percentiles)

#     plt.plot(ms, percentiles[:,0])
#     plt.fill_between(ms, percentiles[:,1], percentiles[:,2], alpha = 0.3)
#     d = data[z]

#     plt.scatter(d[:,0], np.log10(d[:,1]*1e+42))
#     #plt.errorbar(d[:,0], 42+d[:,1], d[:,2:].T, fmt = 'o')

# # MAYBE FOR PLOT MAKE GRID WITH PROBABILITIES AND OVERLAY DATA
# # (scatter in LX, Shen paper?)
