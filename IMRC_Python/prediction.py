#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 

@author: 
"""

import numpy as np
from feature_mapping_function import feature_vector 

def prediction_test_set(x, y, mu, n_classes):
    """
    Predict

    This function quantifies the classification error

    Input
    ----------

    x: Array
        Test instances

    y: Vector
        Test labels

    mu: Vector
        Classifier parameter

    n_classes: Integer
        Number of classes

    Output
    ------

    error: Integer
        Classification error

    """
    mistakes = np.zeros((len(x[:, 0])))
    for j in range(0, len(x[:, 0])):
        hat_y = predict_label(x[j, :], mu, n_classes);
        if hat_y != y[j, 0]:
            mistakes[j] = 1
        else:
            mistakes[j] = 0
    error = np.mean(mistakes)
    return error


def predict_label(x, mu, n_classes):
    """
    Predict

    This function assigns labels to instances

    Input
    ----------

    x: Vector
        instances

    mu: Vector
        classifier parameter

    n_classes: Integer
        number of classes

    Output
    ------

    y_pred: Integer
        predicted label

    """
    M = np.zeros((n_classes, len(mu)))
    c = np.zeros((n_classes, 1))
    for j in range(0, n_classes):
        M[j, :] = feature_vector(x, j, n_classes)
        c[j, 0] = np.dot(M[j, :], mu)
    y_pred = np.argmax(c)
    return y_pred
