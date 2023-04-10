#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import casadi as ca
import casadi.tools as ca_tools
import time as time
from datetime import datetime
from scipy import linalg

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class handle_data:
    def __init__(self, state_dim, dt=0.1, N=20):
        self.Ts = dt
        self.horizon = N
        self.g_ = 9.8066

        # declare model variables
        # control parameters
        roll_ref_ = ca.SX.sym('roll_ref_') # 绕y轴
        pitch_ref_ = ca.SX.sym('pitch_ref_') # 绕x轴
        thrust_ref_ = ca.SX.sym('thrust_ref_')  # z向推力
        controls_ = ca.vertcat(*[roll_ref_, pitch_ref_, thrust_ref_])
        num_controls = controls_.size()[0]

        ## control relevant parameters
        self.roll_tau = 0.257
        self.roll_gain = 0.75
        self.pitch_tau = 0.259
        self.pitch_gain = 0.78

        # model states
        x_ = ca.SX.sym('x_')
        y_ = ca.SX.sym('y_')
        z_ = ca.SX.sym('z_')
        vx_ = ca.SX.sym('vx_')
        vy_ = ca.SX.sym('vy_')
        vz_ = ca.SX.sym('vz_')
        roll_ = ca.SX.sym('roll_')
        pitch_ = ca.SX.sym('pitch_')
        yaw_ = ca.SX.sym('yaw_')
        ## define output shape = 9/12?
        states_ = ca.vcat([x_, y_, z_, vx_, vy_, vz_, roll_, pitch_, yaw_])
        num_states = states_.size()[0]
        print("num_states is:", num_states)

        # Vertical trajectory in z-direction
        self.trajectory = np.array(
                [
                [0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                ])
    
        # additional parameters for cost function
        self.R_m = np.diag([160.0, 4.0, 4.0]) # roll_ref, pitch_ref, thrust
        self.Q_m = np.diag([40.0, 40.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 40.0])   
        # need thrust in the vertical direction, Q_m最后一项不为0
        # P_m = ?


        # DeePC
        # states and parameters
        U = ca.SX.sym('U', num_controls, self.horizon-1)  # (3,19)  reshape?
        Y = ca.SX.sym('Y', num_states, self.horizon)  # (9,20)
        U_ref = ca.SX.sym('U_ref', num_controls, self.horizon-1)
        Y_ref = ca.SX.sym('Y_ref', num_states, self.horizon) # Y_ref is from casadi saved data
        # get output reference
        # Y_ref = np.zeros((1, 9))
        # Y_ref[0,0] = 0.5
        # Y_ref[0,1] = 0.5
        # Y_ref[0,2] = 1.5
        # constraints and cost
        
        ## end term
        obj = ca.mtimes([   #mimes表示最近一次文件内容被修改的时间
            (Y[:, -1] - Y_ref[:, -1]).T,    # 取Y第0至9项减去Y_ref后转置
            self.Q_m,                         # cost function?
            Y[:, -1] - Y_ref[:, -1]
        ])

        ## control cost, u_cost = (u-u_ref)*R*(u-u_ref)
        for i in range(self.horizon-1):
            temp_ = ca.vertcat(U[:, i] - U_ref[:, i])
            obj = obj + ca.mtimes([
                temp_.T, self.R_m, temp_
            ])
            
        ## state cost, y_cost = (y-y_ref)*Q*(y-y_ref)
        for i in range(self.horizon-1):
            temp_ = Y[:-1, i] - Y_ref[:-1, i+1]   
            obj = obj + ca.mtimes([temp_.T, self.Q_m, temp_])

        ### constraints
        g = []

        g.append(Y[:, 0]- Y_ref[:, 0])    
        for i in range(self.horizon-1):
            Y_next_ = 1    # Y_next read from np.saved data
            g.append(Y[:, i+1]-Y_next_)

        lambda_s = 7.5e8
        y_norm = lambda_s*(Y_p*g-y_ini)
        obj = obj + np.linalg.norm(y_norm, ord=2)
        
        # Regularization
        opt_variables = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(Y, -1, 1), ca.reshape(y_ini, -1, -1))
        opt_params = ca.vertcat(ca.reshape(U_ref, -1, 1), ca.reshape(Y_ref, -1, 1), lambda_s)
        nlp_prob = {'f': obj, 'x':opt_variables, 'p':opt_params,'g':ca.vertcat(*g)}
        opts_setting = {'ipopt.max_iter':200, 'ipopt.print_level':1, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6, 'ipopt.warm_start_init_point':'no'}

        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

         ### constraints and get optimal g
    # def constraints(self,H,ini):
    # ## Reshape Hankel matrix into a 2D matrix for input into lstsq
    #     H_flat = H.reshape((H.shape[0] * H.shape[1], H.shape[2]))
    # # Compute least-squares solution
    #     g = []
    #     g = np.linalg.lstsq(H_flat, ini.T, rcond=None)
    #     # Reshape g back into original shape
    #     g = g.reshape((H.shape[2], H.shape[1]))
    #     return g

    # def cost_fun(u, x, g_opt, lambda_s):
    #     xu_pred = np.concatenate([x, u], axis=0).dot(g_opt)
    #     c = np.sum(xu_pred**2)
    #     return c
    # def constraints(self,g,lambda_s):
    # g = U_p.I*u_ini

    # # Define the objective function
    # def obj(g, U, y):
    #     return np.linalg.norm(U.dot(g) - y)**2
    
    # # get optimal g
    # def optimizer(xu, g0, lambda_s):
    #     u_0 = np.zeros((Tini, 3))
    #     opt_result = np.minimize(cost_fun, u_0, args=(x, g_opt, lambda_s), method='L-BFGS-B', bounds=bounds, options={'disp': True})
    #     opt_result = np.minimize(cost_fun, g_0, args=(xu, lambda_s), method='BFGS', jac=False, options={'disp': True})
    #     g_opt = opt_result.x
    #     return g_opt
          
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
    
    def vertical_trajectory(self, current_state,):
        if current_state[2] >= self.trajectory[0, 2]:  #第三列数值大于等于第一行第三列预计轨迹时前移，直至轨迹终点
            self.trajectory = np.concatenate((current_state.reshape(1, -1), self.trajectory[2:], self.trajectory[-1:]))
        return self.trajectory

    def circle_trajectory(self, current_state, iter):
        if iter<=30:
            if current_state[2] >= self.trajectory[0, 2]:
                self.trajectory = np.concatenate((current_state.reshape(1, -1), self.trajectory[2:], self.trajectory[-1:]))
        else:
            idx_ = np.array([(iter+i-30)/360.0*np.pi for i in range(19)])
            trajectory_ =  self.trajectory[1:].copy()
            trajectory_[:, :2] = np.concatenate((np.cos(idx_), np.sin(idx_))).reshape(2, -1).T
            self.trajectory = np.concatenate((current_state.reshape(1, -1), trajectory_))
            # print(iter)
            # print(trajectory_[:4])
        return self.trajectory
        


if __name__ == '__main__':
    ## parameters
    Td = 331 # total number of data in Hankel Matrix,data length
    Tf = 25  # Tf = exc_param.Nd  prediction horizon 
    Tini = 6 # time horizon used for initial condition estimation
    L = 31  # number of block rows in Hankel
    n_states = 9 # system order, = A.shape[0]
    n_controls = 3 #number of input = B.shape[1]
    p = 3 # number of output = c.shape[0]L = 83  # Td>=(m+1)L-1
    dt = 0.1
    N = 20
    deepc_obj = handle_data(state_dim=n_states, dt=dt, N=N)
    init_state = np.array([0.0]*n_states)
    current_state = init_state.copy() 
    opt_commands = np.zeros((N-1, n_controls))  # (19,3)
    next_states = np.zeros((N, n_states))  # (20,9)


    # Hankel Matrix
    ## controls to Hankel
    saved_u = np.load('../Data_MPC/MPC_controls.npy', allow_pickle= True)
    # print('saved controls \n', saved_u)
    DeePC = handle_data()
    H_u = np.zeros((Tini + Tf, Td-L+1, saved_u.shape[1]))
    for i in range(Td):
        H_u = DeePC.hankel(saved_u[:], L, Td-L+1)
    # print("Hankel_u is:", H_u)
    # print("Hankel_u's shape is:", H_u.shape)

    ## states to Hankel
    saved_y = np.load('../Data_MPC/MPC_states.npy')
    H_y = np.zeros((Tini + Tf, Td-L+1, saved_y.shape[1])) 
    for i in range(Td):
        H_y = DeePC.hankel(saved_y[:], L, Td-L+1)
    # print('saved states \n', saved_y)
    # print("Hankel_y's shape is:", H_y.shape)

    ## data collection for state and controls
    ### divide HM into two parts: past and future
    U_p = np.zeros((Tini, Td-L+1, saved_u.shape[1]))   
    U_f = np.zeros(())
    U_p = H_u[0:Tini,:,:]
    U_f = H_u[Tini:,:,:]
    print("U_p \n", U_p.shape) #, shape Up(6, 301, 3)
    # print("U_f \n", U_f)
    # Up = U_p[:,:,i]
    # Uf = U_f[:,:,i]

    Y_p = np.zeros((Tini, Td-L+1, saved_y.shape[1]))
    Y_f = np.zeros(())
    Y_p = H_y[0:Tini,:,:]
    Y_f = H_y[Tini:,:,:]
    # print("Y_p \n", Y_p)
    # print("Y_f \n", Y_f)

    # initial condition = Tini most recent past but simulate for the first step
    u_ini = U_p[:, 0, :]  # shape [6,3]
    print("uini is", u_ini)
    
    y_ini = Y_p[:, 0, :].reshape((6,1,9))  # shape[6,9]
    # print("yini is", y_ini)

    # # get output reference
    # y_ref = np.zeros((1, 9))
    # y_ref[0,0] = 0.5
    # y_ref[0,1] = 0.5
    # y_ref[0,2] = 1.5

    # set initial trajectory
    init_trajectory = np.array(
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
                 [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                ])
    
    deepc_obj.trajectory = init_trajectory.copy()
    next_trajectory = init_trajectory.copy()
    
    ## set lb and ub
    lbg = 0.0
    ubg = 0.0
    lbx = []
    ubx = []
    for _ in range(N-1):  # np.deg2rad is x*pi/180
        lbx = lbx + [np.deg2rad(-45), np.deg2rad(-45), 0.5*9.8066]
        ubx = ubx + [np.deg2rad(45), np.deg2rad(45), 1.5*9.8066]
    for _ in range(N):
        lbx = lbx + [-np.inf]*n_states
        ubx = ubx + [np.inf]*n_states

    # start DeePC
    sim_time = 600 # Td?
    deepc_iter = 0
    index_time = []
    start_time = time.time()
    lambda_s = 7.5e8

    while(deepc_iter < Td):
        ## set parameters
        control_params = ca.vertcat(U_f.reshape(-1, 1), Y_f.reshape(-1, 1), lambda_s)
        print("U_f reshape:",  U_f.reshape(-1, 1))
        ## initial guess of the optimization targets
        init_control = ca.vertcat(opt_commands.reshape(-1, 1), next_states.reshape(-1, 1))
        ## solve the problem
        t_ = time.time()
        sol = deepc_obj.solver(x0=init_control, p=control_params, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
        index_time.append(time.time() - t_)
        ## get results
        estimated_opt = sol['x'].full()
        deepc_u_ = estimated_opt[:int(n_controls*(N-1))].reshape(N-1, n_controls)
        deepc_y_ = estimated_opt[int(n_controls*(N-1)):].reshape(N, n_states)
        print('deepc u \n',deepc_u_)
        print('deepc y \n',deepc_y_)
        ## save results
        # u_c.append(mpc_u_[0, :])
        # t_c.append(t0)
        # x_c.append(current_state)
        # x_states.append(mpc_x_)

 