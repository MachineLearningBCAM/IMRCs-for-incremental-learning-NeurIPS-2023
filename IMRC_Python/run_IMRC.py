#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 

@author: 
"""

import numpy as np
import scipy.io
import os
import forward_backward_learning as fbl
from efficient_learning import optimization

"""
  Input
  -----
  
      The name of dataset file
      

  Output
  ------

      Classification error

"""

# Import data
data = scipy.io.loadmat('dataset.mat')

# Input
K = 2000 # Iterations of the optimization
W = 2 # Window size
n_classes = 2 # Number of classes
backward_steps = 3 # Number of backward steps
batch_size = 10 # Samples per task

# Training and test data per tasks
X_train = data['X_train']
Y_train = data['Y_train']
X_test = data['X_test']
Y_test = data['Y_test']

# Length of the instance vectors
d = len(X_train[0][0][0])
# Calculate the length m of the feature vector
m = n_classes*(d+1);
# number of tasks
n_tasks = len(Y_train[0])


# Initialize
tau_single = np.zeros((m, n_tasks))
s_single = np.zeros((m, n_tasks))
lambda_single = np.zeros((m, n_tasks))
tau_forward = np.zeros((m, n_tasks))
s_forward = np.zeros((m, n_tasks))
lambda_forward = np.zeros((m, n_tasks))
tau_fb = np.zeros((m, n_tasks))
lmb_fb = np.zeros((m, n_tasks))
mu_f = np.zeros((m, n_tasks))
mu_fb = np.zeros((m, n_tasks))
w_n_f = np.zeros((m, n_tasks))
w0_n_f = np.zeros((m, n_tasks))
mu = np.zeros((m, 1))
d_change_single = np.zeros((m, n_tasks))
d_change = np.zeros((m, n_tasks))
d_change_backward = np.zeros((m, n_tasks))
error_backward = np.zeros((n_tasks, n_tasks))
error_forward = np.zeros((1, n_tasks))
error_single = np.zeros((1, n_tasks))
W2 = np.floor(W/2)

for k in range(0, n_tasks):
    
    # Train set
    x = X_train[0, k]
    y = Y_train[0, k]
    # Test set
    x_test = X_test[0, k]
    y_test = Y_test[0, k]

    # Single task learning
    tau_single, s_single, lambda_single, mu_s, d_change_single, x = fbl.single(x, y, n_classes, k, m, tau_single, s_single, lambda_single, K, d_change_single, batch_size)
    error = prediction_test_set(x_test, y_test, mu_s, n_classes)
    error_single[0, k] = error    
    
    # Forward learning
    tau_forward, s_forward, lambda_forward = fbl.forward_learning(k, tau_single, s_single, lambda_single, W, m, tau_forward, s_forward, lambda_forward, d_change_single)
            
    ## Classifier parameter
    z = np.zeros((m, 1))
    if k == 0:
        w_n1 = z
        w0_n1 = z
        mu1 = np.zeros((m, 1))
        mu1[:, 0] = mu_f[:, k]
        w_n1[:, 0] = w_n_f[:, k]
        w0_n1[:, 0] = w0_n_f[:, k]
        mu, M1, h1, w_n, w0_n = optimization(x, n_classes, mu1, tau_forward[:, k], lambda_forward[:, k], [], [], w_n1, w0_n1, K, 'f')
        for i in range(0, m):
            mu_f[i, k] = mu[i]
            w_n_f[i, k] = w_n[i]
            w0_n[i, k] = w0_n[i]
        M = np.zeros((n_tasks, len(M1), m))
        h = np.zeros((n_tasks, len(M1)))
        M[k, :, :] = M1
        for i in range(0, batch_size):
            h[k, i] = h1[i]
    else:
        mu1 = np.zeros((m, 1))
        w_n1 = z
        w0_n1 = z
        mu1[:, 0] = mu_f[:, k-1]
        w_n1[:, 0] = w_n_f[:, k-1]
        w0_n1[:, 0] = w0_n_f[:, k-1]
        mu, M1, h1, w_n, w0_n = optimization(x, n_classes, mu1, tau_forward[:, k], lambda_forward[:, k], [], [], w_n1, w0_n1, K, 'f')
        for i in range(0, m):
            mu_f[i, k] = mu[i]
            w_n_f[i, k] = w_n[i]
            w0_n_f[i, k] = w0_n[i]
        M[k, :, :] = M1
        for i in range(0, len(M1)):
            h[k, i] = h1[i]
            
            
    ## Prediction
    error = prediction_test_set(x_test, y_test, mu_f[:, k], n_classes)
    error_forward[0, k] = error
    error_backward[k, k] = error


    # Bacward learning
    if k - backward_steps>0:
        j_k = k - backward_steps
    else:
        j_k = 0
    if k > 0:
        for i in range(k-1, j_k-1, -1):
            i1 = np.max([0, i-w2]);
            i2 = np.min([k, i+w2]);
            vector_d = d_change_single[:, i1:i2]
            if i2-i1 > 0:
                a = np.mean(vector_d, 1)
                for l in range(0, m):
                    d_change_backward[l, i+1] =  a[l]
            else:
                a = vector_d
                for l in range(0, m):
                    d_change_backward[l, i+1] = a[l]
            t1, l1 = fbl.forward_backward(k, i, tau_forward[:, i], s_forward[:, i], tau_single, s_single, d_change_backward);
            for l in range(0, m):
                tau_fb[l, i] = t1[l]
                lmb_fb[l, i] = l1[l]
                
            ## Classifier parameter
            h1 = np.zeros((len(h[0]), 1))
            h1[:, 0] = h[i, :]
            mu = mu_f[:, i]
            w_n1 = w_n_f[:, i]
            w0_n1 = w0_n_f[:, i]
            mu, F1, h1, w1, w01 = optimization(x, n_classes, mu, tau_fb[:, i], lmb_fb[:, i], M[i, :, :], h1, w_n1, w0_n1, K, 'b')
            for l in range(0, m):
                mu_fb[l, i] = mu[l]
                
            ## Prediction
            x_test = X_test[0, i]
            y_test = Y_test[0, i]
            error = prediction_test_set(x_test, y_test, mu_fb[:, i], n_classes)
            error_backward[i, k] = error
    
error_b_tasks = np.zeros((backward_steps, n_tasks))
for i in range(0, backward_steps):
    for k in range(0, n_tasks):
        if k > n_tasks - i-1:
            error_b_tasks[i, k] = error_backward[k, -1]
        else:
            error_b_tasks[i, k] = error_backward[k, k+i]
            
# Print mistakes rate
print("The classification error of IMRC method is ",error_backward) 

# Save results as .mat file
mdic = {"error_single": error_single, "error_forward": error_forward, "error_backward": error_backward}
results_file_name = os.path.join(filename + "_results"  + suffix)
scipy.io.savemat(results_file_name, mdic)
