import numpy as np

def henaff_init(n):
    # Initialization of skew-symmetric matrix
    s = np.random.uniform(-np.pi, 0., size=int(np.floor(n / 2.)))
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
