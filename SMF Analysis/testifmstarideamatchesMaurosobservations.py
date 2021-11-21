# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 17:09:50 2021

@author: boettner
"""

import numpy as np
from scipy.special import hyp2f1

import matplotlib.pyplot as plt

alpha = 1
beta  = 0.4

m_c = 1e+11

m_h = np.logspace(6, 17,1000)

x = m_h/m_c

alphaextra = alpha+1
alphabet   = alpha+beta

b = alphaextra/alphabet

m_star = m_h/alphaextra * x**alpha * hyp2f1(1,b,b+1, - x**alphabet)

plt.loglog(m_h, m_star/m_h)