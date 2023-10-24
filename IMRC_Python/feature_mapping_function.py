#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 

@author: 
"""

import numpy as np

def feature_vector(x, y, n_classes):
    """
    Feature mappings

    This function obtains feature vectors

    Input
    ----------

    x: new instance

    y: new label

    n_classes: number of classes

    Output
    ------

    phi: feature vector

    """
    x_phi = np.append(1, x)
    e = np.zeros((1, n_classes))
    e[0, y] = 1
    phi = np.kron(e, x_phi)
    return phi
