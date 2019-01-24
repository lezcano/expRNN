import torch
import torch.nn as nn

from exp_numpy import expm, expm_frechet
from initialization import henaff_init


class Orthogonal(nn.Module):
    """
    Implements a non-square linear with orthogonal colums
    """
    def __init__(self, input_size, output_size):
        super(Orthogonal, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.max_size = max(self.input_size, self.output_size)
        self.log_orthogonal_kernel = nn.Parameter(torch.Tensor(self.max_size, self.max_size))
        self.orthogonal_kernel = torch.empty(self.max_size, self.max_size, requires_grad=True)
        self.skew_initializer = henaff_init

        self.log_orthogonal_kernel.data = \
            torch.as_tensor(self.skew_initializer(self.max_size),
                            dtype=self.log_orthogonal_kernel.dtype,
                            device=self.log_orthogonal_kernel.device)
        self.orthogonal_kernel.data = self._B

    @property
    def _A(self):
        A = self.log_orthogonal_kernel.data
        A = A.triu(diagonal=1)
        return A - A.t()

    @property
    def _B(self):
        return expm(self._A)

    def orthogonal_step(self, optim):
        A = self._A
        B = self.orthogonal_kernel.data
        G = self.orthogonal_kernel.grad.data
        BtG = B.t().mm(G)
        grad = 0.5*(BtG - BtG.t())
        frechet_deriv = B.mm(expm_frechet(-A, grad))

        self.log_orthogonal_kernel.grad = (frechet_deriv - frechet_deriv.t()).triu(diagonal=1)

        optim.step()
        self.orthogonal_kernel.data = self._B
        self.orthogonal_kernel.grad.data.zero_()

    def forward(self, input):
        return input.matmul(self.orthogonal_kernel[:self.input_size, :self.output_size])
