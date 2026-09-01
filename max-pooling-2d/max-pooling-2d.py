import numpy as np
import math

def max_pooling_2d(X: list, pool_size: int) -> list:
    """
    Returns non-overlapping maximum-pooled windows.
    """
    # Write code here
    X = np.asarray(X)
    H_out = len(X) // pool_size
    W_out = len(X[0]) // pool_size
    out = np.empty((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            max = -math.inf
            curr_i = i*pool_size
            curr_j = j*pool_size
            for u in range(pool_size):
                for v in range(pool_size):
                    tmp = X[curr_i+u,curr_j+v]
                    if tmp > max:
                        max = tmp

            out[i,j] = max
    return out.tolist()