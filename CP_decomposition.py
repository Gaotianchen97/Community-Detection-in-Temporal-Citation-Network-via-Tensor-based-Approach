# -*- coding: utf-8 -*-
"""
Paper: Community Detection in Temporal Citation Network via Tensor-based Approach

Author: Tianchen Gao
"""

#Load the required packages
import numpy as np                           # Basic 
from tensortools.operations import unfold as tt_unfold, khatri_rao  # Calculate khatri_rao
import tensorly as tl                        # Tensor Decomposition
import warnings                              # Basic
warnings.filterwarnings("ignore")



# Algorithm 1: CP decomposition with ALS

def decompose_three_way(tensor, rank, max_iter=501, verbose=False):
    '''
    input: 
        tensor: a third-order tensor;
        rank: the number of compontents.
        
    output: three factor matrices
    '''
    #a = np.random.random((rank, tensor.shape[0]))   
    b = np.random.random((rank, tensor.shape[1]))   # initialization matrix
    c = np.random.random((rank, tensor.shape[2]))   # initialization matrix
    for epoch in range(max_iter):                   # iteration
        # optimize a
        input_a = khatri_rao([b.T, c.T])            # fix b and c to solve a
        target_a = tl.unfold(tensor, mode=0).T
        a = np.linalg.solve(input_a.T.dot(input_a), input_a.T.dot(target_a))
        # optimize b
        input_b = khatri_rao([a.T, c.T])            # fix a and c to solve b 
        target_b = tl.unfold(tensor, mode=1).T
        b = np.linalg.solve(input_b.T.dot(input_b), input_b.T.dot(target_b))
        # optimize c
        input_c = khatri_rao([a.T, b.T])            # fix a and b to solve c 
        target_c = tl.unfold(tensor, mode=2).T
        c = np.linalg.solve(input_c.T.dot(input_c), input_c.T.dot(target_c))
        if verbose and epoch % int(max_iter * .2) == 0:    # convergence criterion
            res_a = np.square(input_a.dot(a) - target_a)
            res_b = np.square(input_b.dot(b) - target_b)
            res_c = np.square(input_c.dot(c) - target_c)
            print("Epoch:", epoch, "| Loss (C):", res_a.mean(), "| Loss (B):", res_b.mean(), "| Loss (C):", res_c.mean())
    return a.T, b.T, c.T