#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


class handle_data(object):
    def __init__(self, ):
        pass
    
    def get_dmoc_x_u(self,opt_x,opt_u,Tn,N):
        self.opt_x = opt_x  # shape?
        self.opt_u = opt_u
        self.Tn = Tn
        self.N = N

    #def get_manipulator_states(self, x_state):
       # self.x_states = x_state

    def get_target_trajectory(self, target_trajectory): 
        self.mobile_states = target_trajectory

    def get_computation_time(self, t_comp):
        self.computation_time = t_comp

    def get_r_g(self, r_g):
        self.r_g = r_g

    def save_loaded_data(self,file_name):
        with open(file_name,'wb') as f:
        #    np.save(f, self.Tn)
        #    np.save(f, self.N)
            np.save(f, self.opt_x)
            np.save(f, self.opt_u)
            np.save(f, self.mobile_states)
            print("saving data done")

    def save_loaded_u(self, file_name):
        with open(file_name,'wb') as f:
            np.save(f, self.opt_u)
            print("saving control done")

    def save_loaded_x(self, file_name):
        with open(file_name,'wb') as f:
            np.save(f, self.opt_x)
            print("saving states done")

    def save_cul_r_g(self, file_name):
        with open(file_name,'wb') as f:
            np.save(f, self.r_g)
            print("saving r(g) done")

    def load_saved_data(self,file_name): # filename = where to save the data
        with open(file_name, 'rb') as f:
        #    self.Tn = np.load(f, allow_pickle=True)
        #    self.N = np.load(f, allow_pickle=True)
            self.opt_x = np.load(f, allow_pickle=True)
            self.opt_u = np.load(f, allow_pickle=True)
            ## self.manipulator_states = np.load(f, allow_pickle=True), no need
            self.mobile_states = np.load(f, allow_pickle=True)
            # print(self.opt_x)
            print("reading data accomplished")

    def load_data_u(self,file_name):
        with open(file_name, 'rb') as f:
            self.opt_u = np.load(f, allow_pickle=True)
            return opt_u

    def load_data_x(self,file_name):
        with open(file_name, 'rb') as f:
            self.opt_x = np.load(f, allow_pickle=True)
            return opt_x

    def load_r_g(self,file_name):
        with open(file_name, 'rb') as f:
            self.r_g = np.load(f, allow_pickle=True)
            return r_g
            


