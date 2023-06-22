#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import casadi as ca
import casadi.tools as ca_tools
import time as time
from datetime import datetime
from scipy import linalg
from data_collection import handle_data
import matplotlib.pyplot as plt

# test for subplot opt_u & y
deepc_u = np.load('../Data_MPC/deepc_u.npy', allow_pickle= True)
print('saved opt u \n', deepc_u)   # u shape(24,3)
deepc_y = np.load('../Data_MPC/deepc_y.npy', allow_pickle= True)
print('saved opt y \n', deepc_y)   # y shape(25,9)
Tf = deepc_u.shape[0]
T_values = np.arange(Tf)
Ty = np.arange(deepc_y.shape[0])

fig, axs = plt.subplots(3, 2)
axs[0, 0].plot(T_values, deepc_u[:,0])
axs[0, 0].set_title("deepc u position 1")
axs[1, 0].plot(T_values, deepc_u[:,1])
axs[1, 0].set_title("deepc u position 2")
axs[2, 0].plot(T_values, deepc_u[:,2])
axs[2, 0].set_title("deepc u position 3")
axs[0, 1].plot(Ty, deepc_y[:,0])
axs[0, 1].set_title("deepc y position 1")
axs[1, 1].plot(Ty, deepc_y[:,1])
axs[1, 1].set_title("deepc y position 2")
axs[2, 1].plot(Ty, deepc_y[:,2])
axs[2, 1].set_title("deepc y position 3")
fig.tight_layout()
plt.show() 
# plt.hold(True)

# fig, axs = plt.subplots(3, sharex=True, sharey=True)
# fig.suptitle('saved optimal u from first iteration')
# axs[0].plot(T_values, u1)
# axs[1].plot(T_values, u2)
# axs[2].plot(T_values, u3)

# plt.show()

# Tf = 25
# T_values = np.arange(Tf)
# fig, axs = plt.subplots(3, sharex=True, sharey=True)
# fig.suptitle('saved optimal y from first iteration')
# axs[0].plot(T_values, deepc_y[:,0])
# axs[1].plot(T_values, deepc_y[:,1])
# axs[2].plot(T_values, deepc_y[:,2])

# plt.show()

saved_u = np.load('../Data_MPC/MPC_controls.npy', allow_pickle= True)
saved_y = np.load('../Data_MPC/MPC_states.npy', allow_pickle= True)
print("saved u", saved_u.shape)
print("saved y", saved_y.shape)
Tf = saved_u.shape[0]
T_values = np.arange(Tf)
# print("data length", T_values)

fig, axs = plt.subplots(3, 2)
axs[0, 0].plot(T_values, saved_u[:,0])
axs[0, 0].set_title("saved u position 1")
axs[1, 0].plot(T_values, saved_u[:,1])
axs[1, 0].set_title("saved u position 2")
axs[2, 0].plot(T_values, saved_u[:,2])
axs[2, 0].set_title("saved u position 3")
# axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(T_values, saved_y[:,0])
axs[0, 1].set_title("saved y position 1")
axs[1, 1].plot(T_values, saved_y[:,1])
axs[1, 1].set_title("saved y position 2")
axs[2, 1].plot(T_values, saved_y[:,2])
axs[2, 1].set_title("saved y position 3")
fig.tight_layout()
plt.show()