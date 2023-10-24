#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 

@author: 
"""

import itertools
import numpy as np
from feature_mapping_function import feature_vector 

def optimization(x, n_classes, mu, tau, lmb, F, h, w, w0, K, tp):
    """
    Optimization

    This function solves the optimization using an algorithm based on
    Nesterov's method

   Input
   -----

    k: Integer
        step

    x: Array
        instances

    mu: Vector
        classifier parameter for the preceding task

    tau: Vector
        mean vector

    lmb: Vector
        confidence vector

    F: Array
        matrix of subgradients 
    
    h: Vector
        vector of subgradients
        
    w, w0: Vectors
        vectors required to solve the optimization
        
    K: Integer
        number of iterations

    tp: String
        's' single task learning
        'f' forward learning
        'b' forward and backward learning

   Output
   ------
    
    mu: Vector
        classifier parameter

    F, h,, w, w0: Array and vectors
        parameters required to solve the optimization

    """
    theta = 1
    theta0 = 1
    d = len(x)
    muaux = mu
    R_Ut = 0
    m = len(mu)
    if tp == 'f':
        F = np.zeros((1, m))
        h = np.zeros((1))
        for i in range (0,len(x[:, 0])):
            M = np.zeros((n_classes, len(mu)))
            for j in range(0, n_classes):
                M[j, :] = feature_vector(x[i, :], j, n_classes)
            for j in range(0, n_classes):
                aux = list(itertools.combinations([*range(0, n_classes, 1)], j+1))
                for kk in range(0, np.size(aux, 0)):
                    idx = np.zeros((1, n_classes))
                    a = aux[kk]
                    for mm in range(0, len(a)):
                        idx[0, a[mm]] = 1
                    a = (np.dot(idx, M))/(j+1)
                    b = -1/(j+1)
                    F = np.vstack([F, a])
                    h = np.vstack([h, b])
        F = np.delete(F, 0, 0)
        h = np.delete(h, 0, 0)
    elif tp == 'b':
        F = F
        muaux = np.zeros((m, 1))
        w1 = w
        w01 = w0
        w = np.zeros((m, 1))
        w0 = np.zeros((m, 1))
        for i in range(0, m):
            muaux[i, 0] = mu[i]
            w[i, 0] = w1[i]
            w0[i, 0] = w01[i]
        
    else:
        print('Error')

    v = np.dot(F, muaux) + h
    varphi = max(v)[0]
    a = lmb*np.transpose(abs(muaux))
    regularization = sum(a[0])
    R_Ut_best_value = 1  - np.dot(np.transpose(tau), muaux)[0] + varphi + regularization
    for i in range(0, K):
        muaux = w + theta*((1/theta0) - 1)*(w-w0)
        v = np.dot(F, muaux) + h
        varphi = max(v)[0]
        idx_mv = np.where(v == varphi)
        if len(idx_mv[0])>1:
            fi = F[[idx_mv[0][0]], :]
        else:
            fi = F[idx_mv[0], :]
        subgradient_regularization = lmb*np.transpose(np.sign(muaux))
        a = lmb*np.transpose(abs(muaux))
        regularization = sum(a[0])
        g = - tau + fi + subgradient_regularization[0]
        theta0 = theta
        theta = 2/(i+2)
        alpha = 1/((i+2)**(3/2))
        w0 = w
        w = muaux - alpha*np.transpose(g)
        R_Ut = 1 - np.dot(np.transpose(tau), muaux)[0] + varphi + regularization
        if R_Ut < R_Ut_best_value:
            R_Ut_best_value = R_Ut
            mu = muaux
    v = np.dot(F, muaux) + h
    varphi = max(v)[0]
    a = lmb*np.transpose(abs(w))
    regularization = sum(a[0])
    R_Ut = 1 - np.dot(np.transpose(tau), w)[0] + varphi + regularization
    if R_Ut < R_Ut_best_value:
        R_Ut_best_value = R_Ut
        mu = w;
    return mu, F, h, w, w0
