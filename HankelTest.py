#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import casadi as ca
import casadi.tools as ca_tools
import time as time
from datetime import datetime
from scipy import linalg
from data_collection import handle_data

class M_data:
    def __init__(self, state_dim=9, dt=0.1, N=25):
        self.Ts = dt
        self.horizon = N
        self.g_ = 9.8066

    def hankel(self,data,num_block_rows,num_block_cols): 
    # Create the Hankel matrix
    # num_block_rows(L) = number of block rows wanted in Hankel matrix
        self.num_data_rows = data.shape[0] # 为data中u的行数 = 331
        num_block_cols = self.num_data_rows - num_block_rows + 1
        # H = np.zeros((num_block_rows, num_block_cols)) 
        x = np.concatenate((data[:num_block_rows+num_block_cols-1], data[:num_block_rows+num_block_cols-2:-1]))  # build vector of user data
        ij = (np.arange(num_block_rows)[:,None] + np.arange(num_block_cols))  # Hankel subscripts
        H = x[ij]  # actual data
        if ij.shape[1] == 1:  # preserve shape for a single row
            H = H.T
        return H

    def matrix(self,data):
    # reshape Hankel Matrix to the right form 
        B = []  # e.g. U_p(6,301,3) --> (6*3, 301)
        C = []
        num_rows = data.shape[0] * data.shape[2]
        num_cols = data.shape[1]
        
        # Reshape the data to (num_rows, num_cols)
        B = data.transpose(0, 2, 1)
        C = B.reshape((num_rows, num_cols))
        return C


if __name__ == '__main__':
    Td = 331 # total number of data in Hankel Matrix,data length
    Tf = 25  # Tf = exc_param.Nd  prediction horizon 
    Tini = 6 # time horizon used for initial condition estimation
    L = 31  # number of block rows in Hankel
    n_states = 9 # system order, = A.shape[0]
    n_controls = 3 #number of input = B.shape[1]
    p = 3 # number of output = c.shape[0]L = 83  # Td>=(m+1)L-1
    dt = 0.1
    N = Tf

    # Hankel Matrix
    ## controls to Hankel
    saved_u = np.load('../Data_MPC/MPC_controls.npy', allow_pickle= True)
    # print('saved controls \n', saved_u)
    DeePC = M_data()
    H_u = np.zeros((Tini + Tf, Td-L+1, saved_u.shape[1]))
    for i in range(Td):
        H_u = DeePC.hankel(saved_u[:], L, Td-L+1)
    # print("Hankel_u is:", H_u)
    print("Hankel_u's shape is:", H_u.shape)

    ## states to Hankel
    saved_y = np.load('../Data_MPC/MPC_states.npy', allow_pickle= True)
    H_y = np.zeros((Tini + Tf, Td-L+1, saved_y.shape[1])) 
    for i in range(Td):
        H_y = DeePC.hankel(saved_y[:], L, Td-L+1)
    # print('saved states \n', saved_y)
    print("Hankel_y's shape is:", H_y.shape)

    ## data collection for state and controls
    ### divide HM into two parts: past and future
    U_p = np.zeros((Tini, Td-L+1, saved_u.shape[1]))   
    U_f = np.zeros(())
    U_p = H_u[0:Tini,:,:]
    U_f = H_u[Tini:-1,:,:]

    Y_p = np.zeros((Tini, Td-L+1, saved_y.shape[1]))
    Y_f = np.zeros(())
    Y_p = H_y[0:Tini,:,:]
    Y_f = H_y[Tini:,:,:]
    print("Y_p \n", Y_p.shape)
    print("Y_f \n", Y_f.shape)

    Y_ref = np.array(
        [[0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.67, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.69, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.2, 0.0, 0.73, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.2, 0.0, 0.76, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.2, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.2, 0.0, 0.83, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.2, 0.0, 0.85, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.3, 0.0, 0.88, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.3, 0.0, 0.91, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.4, 0.0, 0.93, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.97, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.7, 0.0, 0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.8, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.9, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ])

    U_ref = np.zeros((n_controls, Tf-1))   # (3,24)
    print("control ref \n", U_ref.shape)
    for i in range(Tf-1):
        U_ref[2,i] = 9.8066

    G = ca.SX.sym('G', Td-L+1, 1)    # (301,1)
    G_ref = ca.SX.sym('G_ref', Td-L+1, 1) #(301,1)
    U_p = DeePC.matrix(U_p)
    U_f = DeePC.matrix(U_f)
    Y_p = DeePC.matrix(Y_p)
    Y_f = DeePC.matrix(Y_f)
    print("U_p \n", U_p)
    print("U_f \n", U_f)
    print("Y_p \n", Y_p.shape)
    print("Y_f \n", Y_f.shape)


    # r(g)
    r_g = []
    lambda_g = 500
    stacked_ur_Tini = ca.repmat(U_ref[:,1], 1, Tini) 
    stacked_yr_Tini = ca.repmat(Y_ref[24,:], 1, Tini)
    stacked_ur_Tf = ca.repmat(U_ref[:,1],1, Tf-1)
    stacked_yr_Tf = ca.repmat(Y_ref[24,:], 1, Tf)
    
    print("stacked1 shape:", stacked_ur_Tini)
    print("stacked2 shape:", stacked_yr_Tini)
    print("stacked3 shape:", stacked_ur_Tf.shape)
    print("stacked4 shape:", stacked_yr_Tf.shape)
    stacked = ca.vertcat(
        ca.reshape(stacked_yr_Tini, -1, 1),    # Tini & Y_ref
        ca.reshape(stacked_yr_Tf, -1, 1),      # Tf & Y_ref
        ca.reshape(stacked_ur_Tini, -1, 1),    # Tini & U_ref
        ca.reshape(stacked_ur_Tf, -1, 1)       # Tf & U_ref
        )

    print("stacked shape:", stacked.shape)
    combined = ca.vertcat(Y_p, Y_f, U_p, U_f)
    print("combined UY shape:", combined.shape)

    start_time = time.time()
    combined_inv = ca.pinv(combined)
    print("combined inv shape:", combined_inv.shape)
    end_time = time.time()
    print('time for inverse \n', end_time-start_time)
    t_ = time.time()
    G_ref = ca.mtimes(combined_inv, stacked)
    print("G_ref shape:", G_ref.shape)
    end_time = time.time()
    print('time for multi \n', end_time-t_)
    t_ = time.time()
    r_g = lambda_g*ca.norm_2(G-G_ref)
    print("r(g) shape:", r_g)
    end_time = time.time()
    print('time for multi \n', end_time-t_)

    # # data collection for saving calculation time
    # data_save = handle_data()
    # data_save.get_r_g(
    #     r_g = r_g) 
    # data_save.save_cul_r_g(
    #     file_name = '../Data_MPC/r_g.npy')
