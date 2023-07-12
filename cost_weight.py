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
    horizon = 25

    # Hankel Matrix
    ## controls to Hankel
    # saved_u = np.load('../Data_MPC/MPC_controls.npy', allow_pickle= True)
    saved_u = np.load('../Data_MPC/MPC_controls_random.npy', allow_pickle= True)
    DeePC = M_data()
    H_u = np.zeros((Tini + Tf, Td-L+1, saved_u.shape[1]))
    for i in range(Td):
        H_u = DeePC.hankel(saved_u[:], L, Td-L+1)
    # print("Hankel_u is:", H_u)
    print("Hankel_u's shape is:", H_u.shape)

    ## states to Hankel
    # saved_y = np.load('../Data_MPC/MPC_states.npy', allow_pickle= True)
    saved_y = np.load('../Data_MPC/MPC_states_random.npy', allow_pickle= True)
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

    # G = ca.SX.sym('G', Td-L+1, 1)    # (301,1)
    # G_ref = ca.SX.sym('G_ref', Td-L+1, 1) #(301,1)
    U_p = DeePC.matrix(U_p)
    U_f = DeePC.matrix(U_f)
    Y_p = DeePC.matrix(Y_p)
    Y_f = DeePC.matrix(Y_f)
    print("U_p \n", U_p.shape)
    print("U_f \n", U_f.shape)
    print("Y_p \n", Y_p.shape)
    print("Y_f \n", Y_f.shape)

    Y_ref = np.array(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.67, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.69, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.2, 0.0, 0.73, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.2, 0.0, 0.76, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.2, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.2, 0.0, 0.83, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.3, 0.0, 0.85, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.3, 0.0, 0.88, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.4, 0.0, 0.91, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.93, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.7, 0.0, 0.97, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.8, 0.0, 0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.9, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ])

    # # soft constraints
    # Y_ref = np.array(
    #     [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 0.67, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 0.69, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 0.73, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 0.76, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 0.83, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 0.85, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 0.88, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 0.91, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 0.93, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 0.97, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #     ])
    
    Y_ref = Y_ref.T   # (9,25)

    U_ref = np.zeros((n_controls, Tf-1))   # (3,24)
    for i in range(Tf-1):
        U_ref[2,i] = 9.8066
    print("control ref \n", U_ref.shape)

    y_ini = np.array([0.0]*n_states*Tini)
    u_ini = np.array([0.0]*n_controls*Tini)   # (3*6,)
    u_ini[2::3] = 9.8066

    rank_Up = np.linalg.matrix_rank(U_p)
    print("Hankel_up's rank is:", rank_Up)
    rank_Uf = np.linalg.matrix_rank(U_f)
    print("Hankel_uf's rank is:", rank_Uf)
    rank_Yp = np.linalg.matrix_rank(Y_p)
    print("Hankel_yp's rank is:", rank_Yp)
    rank_Yf = np.linalg.matrix_rank(Y_f)
    print("Hankel_yf's rank is:", rank_Yf)
    
    # soft constraints
    deepc_u_ = np.load('../Data_MPC/deepc_u.npy', allow_pickle= True)
    # print('saved deepc u \n', deepc_u_)   # deepc u shape(24,3)
    deepc_y_ = np.load('../Data_MPC/deepc_y.npy', allow_pickle= True)
    # print('saved deepc y \n', deepc_y_)   # deepc y shape(25,9)
    deepc_g_ = np.load('../Data_MPC/deepc_g.npy', allow_pickle= True)

    # # hard constraints
    # deepc_u_ = np.load('../Data_MPC/deepc_u_vertical.npy', allow_pickle= True)
    # # print('saved deepc u \n', deepc_u_)   # deepc u shape(24,3)
    # deepc_y_ = np.load('../Data_MPC/deepc_y_vertical.npy', allow_pickle= True)
    # # print('saved deepc y \n', deepc_y_)   # deepc y shape(25,9)
    # deepc_g_ = np.load('../Data_MPC/deepc_g_vertical.npy', allow_pickle= True)

    print('saved deepc g \n', deepc_g_.shape)   # deepc g shape(301,1)
    deepc_u_ = deepc_u_.T
    deepc_y_ = deepc_y_.T
    print('saved deepc u \n', deepc_u_.shape)  # (3,24)
    print('saved deepc y \n', deepc_y_.shape)  # (9,25)

    # additional parameters for cost function
    R_m = np.diag([4.0, 4.0, 160.0]) # roll_ref, pitch_ref, thrust
    Q_m = np.diag([40.0, 40.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0])  
    P_m = np.diag([86.21, 86.21, 120.95, 6.94, 6.94, 11.04])
    P_m[0, 3] = 6.45
    P_m[3, 0] = 6.45
    P_m[1, 4] = 6.45
    P_m[4, 1] = 6.45
    P_m[2, 5] = 10.95
    P_m[5, 2] = 10.95 

    
    # check the weight of the cost function
    # terminal cost
    A = ca.vertcat(deepc_y_[:6, -1] - Y_ref[:6, -1])
    print("A \n", A.shape)
    obj = ca.mtimes([A.T, P_m, A])
    print("terminal cost weight \n", obj)

    # control cost, u_cost = (u-u_ref)*R*(u-u_ref)
    for i in range(horizon-1):
        temp_ = ca.vertcat(deepc_u_[:, i] - U_ref[:, i])
        obj += ca.mtimes([temp_.T, R_m, temp_])
    print("control cost weight \n", obj)
    
    # state cost, y_cost = (y-y_ref)*Q*(y-y_ref)
    for i in range(horizon-1):
        temp_ = ca.vertcat(deepc_y_[:-1, i] - Y_ref[:-1, i+1])    
        obj += ca.mtimes([temp_.T, Q_m, temp_])
    print("stage cost weight \n", obj)

    # constraints cost
    lambda_s = 1e3
    g_norm = ca.norm_2(ca.mtimes([Y_p,deepc_g_])-ca.reshape(y_ini, (-1, 1)))**2   # G from Yp
    obj2 = lambda_s*g_norm
    print("constraints cost weight \n", g_norm)

    # r(g)
    r_g = []
    lambda_g = 500
    stacked_ur_Tini = ca.repmat(U_ref[:,0], 1, Tini) 
    stacked_yr_Tini = ca.repmat(Y_ref[:, 24], 1, Tini)
    stacked_ur_Tf = ca.repmat(U_ref[:,0],1, Tf-1)
    stacked_yr_Tf = ca.repmat(Y_ref[:, 24], 1, Tf)

    stacked = ca.vertcat(
        ca.reshape(stacked_yr_Tini, -1, 1),    # Tini & Y_ref
        ca.reshape(stacked_yr_Tf, -1, 1),      # Tf & Y_ref
        ca.reshape(stacked_ur_Tini, -1, 1),    # Tini & U_ref
        ca.reshape(stacked_ur_Tf, -1, 1)       # Tf & U_ref
        )

    print("stacked shape:", stacked.shape)

    combined_inv = np.load('../Data_MPC/inverse.npy', allow_pickle= True)
    # print('saved inverse \n', saved_inv)

    t_ = time.time()
    G_ref = ca.mtimes(combined_inv, stacked)
    # print("G_ref shape:", G_ref.shape)
    end_time = time.time()
    # print('time for multi \n', end_time-t_)
    t_ = time.time()
    reg = ca.norm_2(deepc_g_-G_ref)
    r_g = lambda_g*ca.norm_2(deepc_g_-G_ref)
    # print("r(g) shape:", r_g.shape)
    end_time = time.time()
    # print('time for multi \n', end_time-t_)
    print("regularization func weight \n", reg)