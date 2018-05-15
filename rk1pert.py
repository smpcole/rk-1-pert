import numpy as np
import random
import matplotlib.pyplot as plt
import sys

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

def plot_eigs(A, tmax, tstep, filename):
    """Plot the eigenvalues of the Hermitian pencil A(t) for t in [0, tmax]"""
    x = np.arange(0, tmax, tstep)
    y = {t: np.linalg.eigvalsh(A(t)) for t in x}
    n = A(0).shape[0]
    for i in range(n):
        plt.plot(x, [y[t][i] for t in x])
    plt.xlabel(r"$t$")
    plt.ylabel(r"Eigenvalues of $A(t)$")
    plt.savefig(filename)

def test_eigs(n, dmin, dmax, tmax, tstep):
    """Plot the eigenvalues of a randomly generated Hermitian pencil with given parameters"""
    D = rand_diag(n, dmin, dmax)
    U = rand_rk1_proj(n)
    A = pencil(D, U)
    plot_eigs(A, tmax, tstep, "output.pdf")



if __name__ == "__main__":

    try:
        n = int(sys.argv[1])
        (dmin, dmax, tmax, tstep) = (float(a) for a in sys.argv[2:])
    except(ValueError, TypeError):
        print("ERROR ERROR ERROR!!!")
        print("Usage: python rk1pert.py n dmin dmax tmax tstep")
        exit()

    test_eigs(n, dmin, dmax, tmax, tstep)
