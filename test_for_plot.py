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

# # test for subplot opt_u & y
# deepc_u_ = np.load('../Data_MPC/deepc_u.npy', allow_pickle= True)
# print('saved deepc u \n', deepc_u_)   # u shape(24,3)
# deepc_y_ = np.load('../Data_MPC/deepc_y.npy', allow_pickle= True)
# print('saved deepc y \n', deepc_y_)   # y shape(25,9)
# Tu = np.arange(deepc_u_.shape[0])
# Ty = np.arange(deepc_y_.shape[0])

# fig, axs = plt.subplots(3, 2)
# axs[0, 0].plot(Tu, deepc_u_[:,0])
# axs[0, 0].set_title("deepc u position 1")
# axs[1, 0].plot(Tu, deepc_u_[:,1])
# axs[1, 0].set_title("deepc u position 2")
# axs[2, 0].plot(Tu, deepc_u_[:,2])
# axs[2, 0].set_title("deepc u position 3")
# axs[0, 1].plot(Ty, deepc_y_[:,0])
# axs[0, 1].set_title("deepc y position 1")
# axs[1, 1].plot(Ty, deepc_y_[:,1])
# axs[1, 1].set_title("deepc y position 2")
# axs[2, 1].plot(Ty, deepc_y_[:,2])
# axs[2, 1].set_title("deepc y position 3")
# fig.tight_layout()
# plt.show() 

# # fig, axs = plt.subplots(3, sharex=True, sharey=True)
# # fig.suptitle('saved optimal u from first iteration')
# # axs[0].plot(Tu, u1)
# # axs[1].plot(Tu, u2)
# # axs[2].plot(Tu, u3)

# # plt.show()

saved_u = np.load('../Data_MPC/MPC_controls_v5.npy', allow_pickle= True)
saved_y = np.load('../Data_MPC/MPC_states_v5.npy', allow_pickle= True)
print("saved u", saved_u)
print("saved y", saved_y)
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


## test for casadi_v5
saved_u = np.load('../Data_MPC/MPC_controls_random.npy', allow_pickle= True)
saved_y = np.load('../Data_MPC/MPC_states_random.npy', allow_pickle= True)
print("saved u", saved_u)
print("saved y", saved_y)
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