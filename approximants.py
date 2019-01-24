import torch
import numpy as np


def exp_pade(X):
    """
    Degree 7 Padé approximant
    """
    n = X.size(0)
    Id = torch.eye(n, dtype=X.dtype, device=X.device)
    X1 = torch.mm(X, X)
    X2 = torch.mm(X1, X1)
    X3 = torch.mm(X1, X2)
    P1 = 17297280. * Id + 1995840. * X1 + 25200. * X2 + 56. * X3
    P2 = torch.mm(X,
                  8648640. * Id + 277200. * X1 + 1512. * X2 + X3)
    p7pos = P1 + P2
    p7neg = P1 - P2
    return torch.gesv(p7pos, p7neg)[0]


def cayley(X):
    n = X.size(0)
    Id = torch.eye(n, dtype=X.dtype, device=X.device)
    return torch.gesv(Id - .5 * X, Id + .5 * X)[0]


def taylor(X, n):
    n = X.size(0)
    Id = torch.eye(n, dtype=X.dtype, device=X.device)
    coeff = [Id, X]
    for i in range(2, n):
        coeff.append(coeff[-1].mm(X) / i)
    return sum(coeff)


def scale_square(X, exp):
    """
    Scale-squaring trick
    """
    norm = X.norm()
    if norm < .5:
        return X

    k = int(np.ceil(np.log2(float(norm)))) + 1
    B = X * (2.**-k)
    E = exp(B)
    for _ in range(k):
        E = torch.mm(E, E)
    return E
