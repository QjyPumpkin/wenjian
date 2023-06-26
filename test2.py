#!/usr/env python
# -*- coding: utf-8 -*-
import numpy as np
import casadi as ca
import time as time
from datetime import datetime

class data_test:
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
    A = np.random.randint(1, 100, (4,5,3))
    # A = np.random.rand(4,5,3)
    print('A \n', A)   
    obj = data_test()
    B = obj.matrix(A)
    print("B \n", B)

# A 
#  [[[50 30  3]
#   [42 64 34]
#   [99 45 99]
#   [84  4 12]
#   [87 41 89]]

#  [[61 34 38]
#   [93 31 95]
#   [81 94 92]
#   [86 75 25]
#   [85 55 24]]

#  [[30 54 76]
#   [10 42 90]
#   [96 92 45]
#   [42 27 15]
#   [33 89 74]]

#  [[96 30 48]
#   [95 34 44]
#   [34 34 11]
#   [29 79 51]
#   [96  3 41]]]
# B 
#  [[50 42 99 84 87]
#  [30 64 45  4 41]
#  [ 3 34 99 12 89]
#  [61 93 81 86 85]
#  [34 31 94 75 55]
#  [38 95 92 25 24]
#  [30 10 96 42 33]
#  [54 42 92 27 89]
#  [76 90 45 15 74]
#  [96 95 34 29 96]
#  [30 34 34 79  3]
#  [48 44 11 51 41]]



    # B = np.reshape(A,(-1,2))
    # print(B)
    # print('B的维度:', B.shape)
    # B = np.reshape(A,(-1,1))
    # print('B \n', B)
    # print('B的维度:', B.shape)

    # C = np.reshape(B,(5,12))
    # print('C \n', C)
    # print('C的维度:', C.shape)

    # start_time = datetime.datetime.now()
    # start_time = time.time()
    combined_inv = ca.pinv(B)
    print("combined inv:", combined_inv)
    print("combined inv shape:", combined_inv.shape)
    # # end_time = datetime.datetime.now()
    # t_ = time.time()
    # print('time for inverse \n', t_-start_time)

    # G_ref = ca.mtimes(combined_inv, stacked)
    # print("G_ref shape:", G_ref.shape)
    # r_g = lambda_g*ca.norm_2(G-G_ref)
    # print("r(g) shape:", r_g.shape)