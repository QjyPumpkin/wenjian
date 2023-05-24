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
    def __init__(self, state_dim=9, dt=0.1, N=25):
        self.Ts = dt
        self.horizon = N
        self.g_ = 9.8066
        Td = 331 # total number of data in Hankel Matrix,data length
        Tf = 25
        Tini = 6 # time horizon used for initial condition estimation
        L = 31  # number of block rows in Hankel

        # declare model variables
        # control parameters
        roll_ref_ = ca.SX.sym('roll_ref_') # 绕y轴
        pitch_ref_ = ca.SX.sym('pitch_ref_') # 绕x轴
        thrust_ref_ = ca.SX.sym('thrust_ref_')  # z向推力
        controls_ = ca.vertcat(*[roll_ref_, pitch_ref_, thrust_ref_])
        num_controls = controls_.size()[0]

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
        ## define output shape = 9
        states_ = ca.vcat([x_, y_, z_, vx_, vy_, vz_, roll_, pitch_, yaw_])
        num_states = states_.size()[0]
        # print("num_states is:", num_states)

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
        self.Q_m = np.diag([40.0, 40.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0])  
        self.P_m = np.diag([86.21, 86.21, 120.95, 6.94, 6.94, 11.04])
        self.P_m[0, 3] = 6.45
        self.P_m[3, 0] = 6.45
        self.P_m[1, 4] = 6.45
        self.P_m[4, 1] = 6.45
        self.P_m[2, 5] = 10.95
        self.P_m[5, 2] = 10.95 
        # need thrust in the vertical direction, Q_m最后一项不为0

        # DeePC
        # states and parameters
        U = ca.SX.sym('U', num_controls, self.horizon-1)  # (3,25)  reshape?
        Y = ca.SX.sym('Y', num_states, self.horizon)  # (9,25)
        # G = ca.SX.sym('G', (num_controls+num_states)*(Tini+Tf), 1)  #(372,1)
        G = ca.SX.sym('G', Td-L+1, 1)    # (301,1)
        U_ref = ca.SX.sym('U_ref', num_controls, self.horizon-1) #(3,25)
        Y_ref = ca.SX.sym('Y_ref', num_states, self.horizon) #(9,25)
        G_ref = ca.SX.sym('G_ref', Td-L+1, 1) #(372,1)
        u_ini = ca.SX.sym('u_ini', Tini, num_controls) # (6,3) ->(18,1)
        y_ini = ca.SX.sym('y_ini', Tini, num_states) # (6,9)  ->(54,1)
        U_p = ca.SX.sym('U_p', num_controls*Tini, Td-L+1)  # (18,301)
        U_f = ca.SX.sym('U_f', num_controls*(Tf-1), Td-L+1)  # (3*24,301)
        Y_p = ca.SX.sym('Y_p', num_states*Tini, Td-L+1)  # (9*6,301)
        Y_f = ca.SX.sym('Y_f', num_states*Tf, Td-L+1)  # (9*25,301)
        U2Y = ca.vertcat(Y_f, U_p, U_f)  # (18+75+225,301)
        print("U2Y shape : \n", U2Y.shape)

        # constraints and cost
        # end term: cost function ,let the last y better simulation
        # state cost
        obj = ca.mtimes([
            (Y[:6, -1] - Y_ref[:6, -1]).T,    
            self.P_m,                      
            Y[:6, -1] - Y_ref[:6, -1]
        ])

        ## control cost, u_cost = (u-u_ref)*R*(u-u_ref)
        # obj = []
        for i in range(self.horizon-1):
            temp_ = ca.vertcat(U[:, i] - U_ref[:, i])
            obj = obj + ca.mtimes([
                temp_.T, self.R_m, temp_
            ])
            
        ## state cost, y_cost = (y-y_ref)*Q*(y-y_ref)
        for i in range(self.horizon-1):
            temp_ = Y[:-1, i] - Y_ref[:-1, i+1]   
            obj = obj + ca.mtimes([temp_.T, self.Q_m, temp_])

        # constraints cost
        lambda_s = 7.5e8
        print("G shape is \n", G.shape)
        # print("Yp shape is \n", Y_p.shape)
        # g_norm = np.linalg.norm(Yp_flat@G-y_ini, ord=2)**2  
        g_norm = ca.norm_2(ca.mtimes([Y_p,G])-ca.reshape(y_ini, (-1, 1)))**2   # G from Yp
        # print("g_norm shape is \n", g_norm.shape)
        obj = obj + lambda_s*g_norm
        # for i in range(self.horizon-1):
        #     x_next_ = self.RK_4(X[:, i], U[:, i], F_ext)

        # r(g)
        # r_g = []
        # lambda_g = 500
        # r_g = lambda_g*ca.norm_2(G-G_ref)


        # constraints g
        g = []
        variables = ca.vertcat(ca.reshape(Y, (-1, 1)), ca.reshape(u_ini, (-1, 1)), ca.reshape(U, (-1, 1)))    # (315,1) 
        # print("variables shape is :",variables.shape)    #(315,1) 

        # for i in range(self.horizon-1):  # first Y_next_ read from the first row from saved Y_f file 
            # Y_next_ = ca.mtimes(Y_f[:, i], G) # Y_next_ computed: y = Y_f*G
        U2Y_sx = ca.mtimes(U2Y, G) # convert U2Y to casadi SX variable (,1)
        # print("U2Y_sx shape is \n", U2Y_sx.shape)
        g.append(U2Y_sx - variables)    # shape missmatch
        # print("g is:", g)
        
        # Regularization OCP
        opt_variables = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(Y, -1, 1), ca.reshape(G, -1, 1))
        opt_params = ca.vertcat(ca.reshape(U_ref, -1, 1), ca.reshape(Y_ref, -1, 1), ca.reshape(u_ini, -1, 1), ca.reshape(y_ini, -1, 1), 
                                ca.reshape(U_p, -1, 1), ca.reshape(U_f, -1, 1), ca.reshape(Y_p, -1, 1),ca.reshape(Y_f, -1, 1) )
        nlp_prob = {'f': obj, 'x':opt_variables, 'p':opt_params,'g':ca.vertcat(*g)}
        opts_setting = {'ipopt.max_iter':200, 'ipopt.print_level':1, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6, 'ipopt.warm_start_init_point':'no'}

        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

        # ##### following only for test
        # ## define constraints
        # lbg = 0.0
        # ubg = 0.0
        # lbx = []
        # ubx = []
        # for _ in range(self.horizon-1):
        #     lbx = lbx + [np.radians(-45), np.radians(-45), 0.5*self.g_]
        #     ubx = ubx + [np.radians(45), np.radians(45), 1.5*self.g_]
        # for _ in range(self.horizon):
        #     lbx = lbx + [-np.inf]*state_dim
        #     ubx = ubx + [np.inf]*state_dim
        # u0 = np.zeros((self.horizon-1, num_controls))
        # y0 = np.zeros((self.horizon, num_states))
        # g0 = np.zeros((self.horizon-1, 1))
        # init_control = ca.vertcat(u0.reshape(-1, 1), y0.reshape(-1, 1), g0.reshape(-1, 1))
        # c_p = ca.vertcat(U_ref.reshape(-1, 1), Y_ref.reshape(-1, 1), u_ini.reshape(-1, 1), y_ini.reshape(-1, 1), U2Y.reshape(-1, 1))
        # res = self.solver(x0=init_control, p=c_p, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
        # estimated_opt = res['x'].full()
        # u1 = estimated_opt[:int(num_controls*(self.horizon-1))].reshape(self.horizon-1, num_controls)
        # y1 = estimated_opt[int(num_controls*(self.horizon-1)):int(num_states*(self.horizon-1))].reshape(self.horizon, num_states)
        # g1 = estimated_opt[int(num_states*(self.horizon-1)):].reshape(self.horizon, 1)
        # print("u1: \n", u1)
        # print("y1: \n", y1)
        # print("g1: \n", g1)
    
          
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

    def RK_4(self, s_t_, c_, f_):
        # discretize Runge Kutta 4
        k1 = self.dyn_function(s_t_, c_, f_)
        k2 = self.dyn_function(s_t_+self.Ts/2.0*k1, c_, f_)
        k3 = self.dyn_function(s_t_+self.Ts/2.0*k2, c_, f_)
        k4 = self.dyn_function(s_t_+self.Ts*k3, c_, f_)
        result_ = s_t_ + self.Ts/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)
        #print('result_rk4',result_)
        return result_

    def model_based_movement(self, state, control, ext_F, t0, u_, x_):
        # print('state at t {0} is {1}'.format(t0, state))
        # print('control at t {0} is {1}'.format(t0, control))
        k1 = self.dyn_np_function(state, control, ext_F)
        k2 = self.dyn_np_function(state+self.Ts/2.0*k1.T, control, ext_F)
        k3 = self.dyn_np_function(state+self.Ts/2.0*k2.T, control, ext_F)
        k4 = self.dyn_np_function(state+self.Ts*k3.T, control, ext_F)
        x_next = state + self.Ts/6.0*(k1.T+2.0*k2.T+2.0*k3.T+k4.T)    # current state
        # nt_ = state + self.dyn_np_function(state, control, ext_F)*self.Ts
        # print('nt is {0}'.format(x_next))
        next_cmd_ = np.concatenate((u_[1:], u_[-1:]), axis=0)   # opt command
        next_s_ = np.concatenate((x_[1:], x_[-1:]), axis=0)   # next state
        # print('next_cmd is {0}'.format(next_cmd_))
        # print('next_s is {0}'.format(next_s_))
        return t0+self.Ts, x_next, next_cmd_, next_s_
    
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
    N = Tf
    deepc_obj = handle_data(state_dim=n_states, dt=dt, N=N)
    initial_control = np.zeros((Tini, n_controls))
    initial_states = np.zeros((Tini, n_states))
    init_state = np.array([0.0]*n_states)
    # print("init_state shape is: \n", init_state.shape)
    current_state = init_state.copy() 
    opt_commands = np.zeros((n_controls, Tf-1))  # = U
    next_states = np.zeros((n_states, Tf))  # UYG shape
    # G_guess = np.zeros((Td-L+1, 1))
    G_guess = np.zeros((Td-L+1, 1))
    control_ref = np.zeros((n_controls, Tf-1))
    for i in range(Tf-1):
        control_ref[2,i] = 9.8066
    # control_ref = np.zeros(((N-1)*n_controls, Td-L+1))
    

    # Hankel Matrix
    ## controls to Hankel
    saved_u = np.load('MPC_controls.npy', allow_pickle= True)
    # print('saved controls \n', saved_u)
    DeePC = handle_data()
    H_u = np.zeros((Tini + Tf, Td-L+1, saved_u.shape[1]))
    for i in range(Td):
        H_u = DeePC.hankel(saved_u[:], L, Td-L+1)
    # print("Hankel_u is:", H_u)
    # print("Hankel_u's shape is:", H_u.shape)

    ## states to Hankel
    saved_y = np.load('MPC_states.npy', allow_pickle= True)
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
    U_f = H_u[Tini:-1,:,:]
    # Up_flat = U_p.reshape(n_controls*Tini, Td-L+1)
    # Up_flat = U_p.reshape((Tini*(Td-L+1),saved_u.shape[1]))
    # print("U_p \n", U_p)  # shape Up(6, 301, 3)   (,3)
    # print("Up_flat \n", Up_flat)
    # print("U_f \n", U_f)
    # print("Uf_flat \n", Uf_flat)

    Y_p = np.zeros((Tini, Td-L+1, saved_y.shape[1]))
    Y_f = np.zeros(())
    Y_p = H_y[0:Tini,:,:]
    Y_f = H_y[Tini:,:,:]
    # print("Y_p \n", Y_p)
    # print("Y_f \n", Y_f)

    # initial condition = Tini most recent past but simulate for the first step
    # u_ini = [0.0, 0.0, 9.8066]
    # u_ini = U_p[:, 0, :]  # shape [6,3]
    # print("uini is", u_ini)
    
    # y_ini = [-0.5, -0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    # y_ini = Y_p[:, 0, :].reshape((6,1,9))  # shape[]
    # print("yini is", y_ini)

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
    lbg = np.zeros((Td-L+1, 1))
    ubg = np.zeros((Td-L+1, 1))
    lbx = []
    ubx = []
    for _ in range(N-1):  # np.deg2rad is x*pi/180
        lbx = lbx + [np.deg2rad(-45), np.deg2rad(-45), 0.5*9.8066]
        ubx = ubx + [np.deg2rad(45), np.deg2rad(45), 1.5*9.8066]
    for _ in range(N):
        lbx = lbx + [-np.inf]*n_states
        ubx = ubx + [np.inf]*n_states
    # lbG ubG vector
    lbx = lbx + [-np.inf]*(Td-L+1)
    ubx = ubx + [np.inf]*(Td-L+1)

    # for saving data
    t0 = 0
    x_c = []
    u_c = []
    t_c = []
    x_states = []
    traj_c = []

    # start DeePC
    sim_time = 300 # Td?
    deepc_iter = 0
    index_time = []
    start_time = time.time()

    while(deepc_iter < 1):
        ## set parameters
        control_params = ca.vertcat(control_ref.reshape(-1, 1), deepc_obj.trajectory.reshape(-1, 1), initial_control.reshape(-1, 1), initial_states.reshape(-1, 1),
                                    U_p.reshape(-1,1), U_f.reshape(-1,1), Y_p.reshape(-1,1), Y_f.reshape(-1,1))
        ## initial guess of the optimization targets
        init_control = ca.vertcat(opt_commands.reshape(-1, 1), next_states.reshape(-1, 1), G_guess.reshape(-1, 1))
        ## solve the problem
        t_ = time.time()
        sol = deepc_obj.solver(x0=init_control, p=control_params, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
        index_time.append(time.time() - t_)
        ## get results
        estimated_opt = sol['x'].full()
        print('x_opt: ', estimated_opt)
        # y_opt = estimated_opt(1)   # reshape
        # g_opt = estimated_opt(3)
        deepc_u_ = estimated_opt[:int(n_controls*(N-1))].reshape(N-1, n_controls)
        deepc_y_ = estimated_opt[int(n_controls*(N-1)):int(n_controls*(N-1)+n_states*N)].reshape(N, n_states)
        deepc_g_ = estimated_opt[int(n_controls*(N-1)+n_states*N):].reshape(Td-L+1, 1)   # G shape?
        # print('deepc u \n',deepc_u_.shape)
        # print('deepc y \n',deepc_y_.shape)
        # print('deepc_g \n',deepc_g_.shape)

        # # save results
        # u_c.append(deepc_u_[0, :])
        # t_c.append(t0)
        # x_c.append(current_state)
        # x_states.append(deepc_y_)
        # ## the localization system
        # t0, current_state, opt_commands, next_states = deepc_obj.model_based_movement(current_state, deepc_u_[0, :], t0, deepc_u_, deepc_y_) # return t0+self.Ts, x_next, next_cmd_, next_s_
        # # next_trajectories = deepc_obj.vertical_trajectory(current_state)
        # # print(next_states)
        # next_trajectories = deepc_obj.circle_trajectory(current_state, deepc_iter)
        # traj_c.append(next_trajectories[1])
        # # print(next_trajectories[:3])
        # # print('current {}'.format(current_state))
        # # print('control {}'.format(deepc_u_[0]))
        # deepc_iter += 1

 