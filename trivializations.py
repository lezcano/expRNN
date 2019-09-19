import torch

from expm import expm, expm_frechet

def cayley_map(X):
    n = X.size(0)
    Id = torch.eye(n, dtype=X.dtype, device=X.device)
    return torch.solve(Id - X, Id + X)[0]

class expm_skew_class(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        B = expm(A)
        ctx.save_for_backward(A, B)
        return B

    @staticmethod
    def backward(ctx, G):
        def skew(X):
            return .5 * (X - X.t())
        # print(G)
        A, B = ctx.saved_tensors
        grad = skew(B.t().matmul(G))
        out = B.matmul(expm_frechet(-A, grad))
        # correct precission errors
        return skew(out)

expm_skew = expm_skew_class.apply
