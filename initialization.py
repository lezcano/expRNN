import numpy as np
import scipy.linalg as la

def henaff_init(n):
    # without uniformization || with uniformization
    # with (without terrible results)
    # with converges to 0 loss stably (without converged to 0 loss)
    M = np.random.normal(0., 1., size=(n, n))
    Q, R = la.qr(M)

    # Uniformisation
    d = np.diag(R, 0)
    ph = np.sign(d)
    Q *= ph

    if la.det(Q) < 0.:
        # Go bijectively from O^-(n) to O^+(n) \iso SO(n)
        Q[0] *= -1.

    A = la.logm(Q).real
    A = .5 * (A - A.T)
    eig = la.eigvals(A).imag
    eig = eig[::2]
    if n % 2 == 1:
        eig = eig[:-1]
    return create_diag(eig, n)


def normal_init_squeeze(n):
    # Epoch XXX timit 8.01 then exploded to 100
    # Epoch 70 98.22
    # Stuck at 0.009

    # Initialization of skew-symmetric matrix
    s = np.random.normal(0, np.pi/8., size=int(np.floor(n / 2.)))
    s = np.fmod(s, np.pi)
    return create_diag(s, n)

def normal_init(n):
    # Initialization of skew-symmetric matrix
    s = np.random.normal(0, 1., size=int(np.floor(n / 2.)))
    s = np.fmod(s, np.pi)
    return create_diag(s, n)


def henaff_init_(n):
    # Initialization of skew-symmetric matrix
    s = np.random.uniform(-np.pi, np.pi, size=int(np.floor(n / 2.)))
    return create_diag(s, n)

def cayley_init(n):
    s = np.random.uniform(0, np.pi/2., size=int(np.floor(n / 2.)))
    s = -np.sqrt((1.0 - np.cos(s))/(1.0 + np.cos(s)))
    return create_diag(s, n)

def create_diag(s, n):
    diag = np.zeros(n-1)
    diag[::2] = s
    A_init = np.diag(diag, k=1)
    A_init = A_init - A_init.T
    return A_init.astype(np.float32)
