#!/usr/env python
# -*- coding: utf-8 -*-
import numpy as np
import casadi as ca
import time as time
from datetime import datetime

A = np.random.randint(1, 100, (4,5,3))
# A = np.random.rand(4,5,3)
print('A \n', A)   

# C 
#  [[62 43 63 58 98 69 14 52 86 55 17 84]
#  [12 97 83 12 80 31 31 90 41 11 23 10]
#  [92 19 60 52 45  3 25 66 49 14  4 42]
#  [ 9 97 45 59 10 66 23 35 49  7 73 95]
#  [20 38  8 85 35  1 70 12 63 41 20 81]]

# A 
#  [[[62 43 63]
#   [58 98 69]
#   [14 52 86]
#   [55 17 84]
#   [12 97 83]]

#  [[12 80 31]
#   [31 90 41]
#   [11 23 10]
#   [92 19 60]
#   [52 45  3]]

#  [[25 66 49]
#   [14  4 42]
#   [ 9 97 45]
#   [59 10 66]
#   [23 35 49]]

#  [[ 7 73 95]
#   [20 38  8]
#   [85 35  1]
#   [70 12 63]
#   [41 20 81]]]




# B = np.reshape(A,(-1,2))
# print(B)
# print('B的维度:', B.shape)
B = np.reshape(A,(-1,1))
print('B \n', B)
print('B的维度:', B.shape)

C = np.reshape(B,(5,12))
print('C \n', C)
print('C的维度:', C.shape)



# [[[ 0  1  2]
#   [ 3  4  5]
#   [ 6  7  8]]

#  [[ 9 10 11]
#   [12 13 14]
#   [15 16 17]]]
# [[ 0  1]
#  [ 2  3]
#  [ 4  5]
#  [ 6  7]
#  [ 8  9]
#  [10 11]
#  [12 13]
#  [14 15]
#  [16 17]]
# B的维度: (9, 2)

# start_time = datetime.datetime.now()
start_time = time.time()
combined_inv = ca.pinv(B)
print("combined inv:", combined_inv)
print("combined inv shape:", combined_inv.shape)
# end_time = datetime.datetime.now()
t_ = time.time()
print('time for inverse \n', t_-start_time)

# G_ref = ca.mtimes(combined_inv, stacked)
# print("G_ref shape:", G_ref.shape)
# r_g = lambda_g*ca.norm_2(G-G_ref)
# print("r(g) shape:", r_g.shape)