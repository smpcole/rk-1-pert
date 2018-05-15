import numpy as np
import random

def rand_unit_vector(n):
    """Return a random unit vector uniformly distributed on unit sphere"""
    u = np.array([random.gauss(0, 1) for i in range(n)])
    return u / np.linalg.norm(u)

def rand_diag(n, dmin, dmax, normalize = False):
    """Generate a diagonal matrix with diagonal entries uniformly distributed in the the specified range.
    If normalize == True, the matrix is then divided by its trace so that its trace is 1.
    The entries are always in nonincreasing order."""
    d = [random.uniform(dmin, dmax) for i in range(n)]
    d.sort()
    d.reverse()
    D = np.diag(d)
    if normalize:
        D /= np.trace(D)
    return D

def rand_rk1_proj(n):
    """Return a random matrix of the form uu^T, where u is a random unit vector"""
    u = rand_unit_vector(n)
    return np.outer(u, u)

def pencil(A0, A1):
    """Return a funtion A(z) = A0 + z * A1"""
    return lambda z: A0 + z * A1
