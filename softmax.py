import numpy as np

def softmax(Z):
    
    A2 = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    cache = Z
    return A2, cache

def softmax_backward(dA, cache):
    
    Z = cache
    dZ = dA
    assert (dZ.shape == Z.shape)

    return dZ
    
