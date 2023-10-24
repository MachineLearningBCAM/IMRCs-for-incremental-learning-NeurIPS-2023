#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 

@author: 
"""

import numpy as np
from feature_mapping_function import feature_vector 
from efficient_learning import optimization

def single(x, y, n_classes, k, m, tau_single, s_single, lambda_single, K, d_change_single, batch_size):
    """
    Single
    
    This function obtains mean and confidence vectors as well as classifier 
    parameters with single-task learning

     Input
     -----
     
    x : Array
        Input instances
    y : Array
        Input classes
    n_classes : Integer
        Number of classes
    k : Integer
        Step
    m : Integer
        Length of the feature vector
    tau_single : Array
        Mean vectors obtained with single-task learning
    s_single : Array
        MSE vectors obtained with single-task learning
    lambda_single : Array
        Confidence vectors obtained with single-task learning
    K : Integer
        Number of iterations of the optimization
    d_change_single : Array
        Expected squared change between consecutive tasks
    batch_size : Integer
        Samples per task

     Output
     ------
     
    tau_single : Array
        Mean vectors obtained with single-task learning
    s_single : Array
        MSE vectors obtained with single-task learning
    lambda_single : Array
        Confidence vectors obtained with single-task learning
    mu_s : Vector
        Classifier parameter obtained with single-task learning
    d_change_single : Array
        Expected squared change between consecutive tasks
    x : Array
        Instances used

    """
    
    Feature = []
    idx = np.random.randint(0, len(x[:, 0]), size=(batch_size))
    x = x[idx, :]
    y = y[idx]

    for i in range(0, len(x[:, 0])):
        a = feature_vector(x[i, :], y[i][0], n_classes)
        if i == 0:
            Feature = a[0]
        else:
            Feature = np.vstack([Feature, a[0]])

    # Single task learning
    ## Mean vector estimate
    tau_single[:, k] = np.mean(Feature, 0)
    s_single[:, k] = np.var(Feature, 0)/(len(Feature[:, 0]))
    lambda_single[:, k] = np.sqrt(s_single[:, k])
     
    ## Classifier parameter
    z = np.zeros((m, 1))
    F1 = []
    h1 = []
    if k > 0:
        a = (tau_single[:, k] - tau_single[:, k-1])**2
        for l in range(0, m):
            d_change_single[l, k] = a[l]
    mu_s, F1, h1, w1, w01 = optimization(x, n_classes, z, tau_single[:, k], lambda_single[:, k], F1, h1, z, z, K, 'f')
    return tau_single, s_single, lambda_single, mu_s, d_change_single, x

def forward(tau, s, tau_single, s_single, d):
    """
     Forward learning

     This function obtains mean vector estimates and confidence vectors

     Input
     -----

     tau: Vector
         mean vector estimate obtained at forward learning for the preceding task

     s: Vector
         s vector obtained at forward learning for the preceding task

     tau_single: Vector
         mean vector estimate obtained at single task learning for the corresponding task

     s_single: Vector
         s vector obtained at single task learning for the corresponding

     Output
     ------

     tau: Vector
         mean vector estimate 
     
     s: Vector
         s vector 
     
     lmb: Vector
         confidence vector

    """
    m = len(tau)
    lmb = np.zeros((m, 1))
    tau1 = np.zeros((m, 1))
    s1 = np.zeros((m, 1))
    for i in range(0, m):
        if s_single[i] == 0:
            tau1[i] = tau_single[i]
            s1[i] = s_single[i]
        else:
            innovation = tau[i]-tau_single[i]
            tau1[i] = tau_single[i] + (s_single[i]/(s[i] + d[i] + s_single[i]))*innovation
            s1[i] = 1/(1/s_single[i] + 1/(s[i] +d[i]))
            
        lmb[i, 0] = np.sqrt(s1[i])
    return tau1, s1, lmb

def forward_learning(k, tau_single, s_single, lambda_single, W, m, tau_forward, s_forward, lambda_forward, d_change_single):
    # Forward learning
    if k == 0:
        tau_forward[:, 0] = tau_single[:, k]
        s_forward[:, 0] = s_single[:, k]
        lambda_forward[:, 0] = lambda_single[:, k]
    else:
        j = np.max([0, k-W+1])
        vector_d = d_change_single[:, j:(k+1)]
        if k-j>0:
            d_change = np.mean(vector_d, 1)
        else:
            d_change = vector_d
        t1, s1, l1 = forward(tau_forward[:, k-1], s_forward[:, k-1], tau_single[:, k], s_single[:, k], d_change)
        for i in range(0, m):
            tau_forward[i, k] = t1[i]
            s_forward[i, k] = s1[i]
            lambda_forward[i, k] = l1[i]
    return tau_forward, s_forward, lambda_forward

  
def forward_backward(k, j, tau_forward, s_forward, tau_single, s_single, d):
    """
     Forward and backward learning

     This function obtains mean vector estimates and confidence vectors

     Input
     -----

     k: Integer
         task step
     
     j: Integer
         task index
     
     tau_forward: Vector
         mean vector estimate obtained at forward learning

     s_forward: Vector
         s vector obtained at forward learning

     tau_single: Array
         mean vector estimate obtained at single task learning

     s_single: Array
         s vector obtained at single task learning

     Output
     ------

     tau: Vector
         mean vector estimate 
     
     s: Vector
         s vector 
     
     lmb: Vector
         confidence vector

    """

    m = len(tau_single);
    tau_backward = np.zeros((m, k+1))
    s_backward = np.zeros((m, k+1))
    s = np.zeros((m, 1))
    tau = np.zeros((m, 1))
    lmb = np.zeros((m, 1))
    for i in range(k, j, -1):
        for l in range(0, m):
            if i == k:
                tau_backward[l, i] = tau_single[l, i]
                s_backward[l, i] = s_single[l, i]
            else:
                if s_single[l, i] == 0:
                    tau_backward[l, i] = tau_single[l, i]
                    s_backward[l, i] = s_single[l, i]
                else:
                    tau_backward[l, i] = tau_single[l, i] + (1/((1/s_single[l, i])*(s_backward[l, i+1] + d[l, i+1])+1))*(tau_backward[l, i+1]-tau_single[l, i])
                    s_backward[l, i] = 1/((1/s_single[l, i]) + 1/(s_backward[l, i+1] + d[l, i+1]))
    for l in range(0, m):
        if s_forward[l] == 0:
            s[l] = s_forward[l]
            tau[l] = tau_forward[l]
        else:
            s[l] = 1/(1/s_forward[l] + 1/(s_backward[l, j+1]+d[l, j+1]))
            tau[l] = tau_forward[l] + (s[l]/(s_backward[l, j+1]+d[l, j+1]))*(tau_backward[l, j+1] - tau_forward[l])
        lmb[l] = np.sqrt(s[l]);
    return tau, lmb
