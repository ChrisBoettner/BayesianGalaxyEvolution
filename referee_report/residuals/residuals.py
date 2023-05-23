#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 13:49:01 2023

@author: chris
"""

import numpy as np
import matplotlib.pyplot as plt

a = -1
a2 = -0.5
b = 1
b2 = -3
num = 500

x     = np.linspace(1,10,num)
noise = np.random.randn(num)
y     = a*x +b + noise
y_l    = a*x + b
y2_l   = a2*x + b2

plt.close()
plt.scatter(x,y, label = f'data variance = {np.var(y):.2f}')
plt.plot(x,y_l, color='black', label = f'residual variance = {np.var(y_l-y):.2f}')
plt.plot(x,y2_l, color='red', label = f'residual variance = {np.var(y2_l-y):.2f}')
plt.xlabel('x')
plt.ylabel('y')
#plt.title('Figure 1')
plt.legend()

plt.savefig('residuals.png')

print(np.var(noise), np.var(y_l-y), np.var(y), np.var(y2_l-y))