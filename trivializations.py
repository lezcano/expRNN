import torch

from expm32 import expm32, expm_frechet

def cayley_map(X):
    n = X.size(0)
    Id = torch.eye(n, dtype=X.dtype, device=X.device)
    return torch.solve(Id - X, Id + X)[0]

class expm_class(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        ctx.save_for_backward(A)
        return expm32(A)

    @staticmethod
    def backward(ctx, G):
        (A,) = ctx.saved_tensors
        return expm_frechet(A.t(), G, expm32)

expm = expm_class.apply
