import torch
import torch.nn as nn

from exp_numpy import expm, expm_frechet
from initialization import henaff_init


def cayley(A):
    n = A.size(-1)
    Id = torch.eye(n, dtype=A.dtype, device=A.device)
    return torch.gesv(Id - A, Id + A)[0]


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



class Orthogonal(nn.Module):
    """
    Implements a non-square linear with orthogonal colums

    is_exp = True uses the exponential parametrization (slower but more stable)
    is_exp = False uses the cayley parametrization (faster but less stable)
    """
    def __init__(self, input_size, output_size, is_exp=True):
        super(Orthogonal, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.max_size = max(self.input_size, self.output_size)
        self.log_orthogonal_kernel = nn.Parameter(torch.Tensor(self.max_size, self.max_size))
        self.orthogonal_kernel = nn.Parameter(torch.Tensor(self.max_size, self.max_size))
        self.is_exp = is_exp
        if self.is_exp:
            self.exp = expm
        else:
            self.exp = cayley

        if self.is_exp:
            self.skew_initializer = henaff_init
        else:
            self.skew_initializer = cayley_init

        self.log_orthogonal_kernel.data = \
            torch.as_tensor(self.skew_initializer(self.max_size),
                            dtype=self.log_orthogonal_kernel.dtype,
                            device=self.log_orthogonal_kernel.device)
        self.orthogonal_kernel.data = self._B(gradients=False)

    def _A(self, gradients):
        A = self.log_orthogonal_kernel
        if not gradients:
            A = A.data
        A = A.triu(diagonal=1)
        return A - A.t()

    def _B(self, gradients=False):
        return self.exp(self._A(gradients))

    def orthogonal_step(self, optim):
        if self.is_exp:
            with torch.no_grad():
                A = self._A(gradients=False)
                B = self.orthogonal_kernel.data
                G = self.orthogonal_kernel.grad.data
                BtG = B.t().mm(G)
                grad = 0.5*(BtG - BtG.t())
                frechet_deriv = B.mm(expm_frechet(-A, grad))

                self.log_orthogonal_kernel.grad = (frechet_deriv - frechet_deriv.t()).triu(diagonal=1)
        else:
            orth_param = self._B(gradients=True)
            self.log_orthogonal_kernel.grad = \
                torch.autograd.grad([orth_param], self.log_orthogonal_kernel,
grad_outputs=(self.orthogonal_kernel.grad,))[0]

        optim.step()
        self.orthogonal_kernel.data = self._B(gradients=False)
        self.orthogonal_kernel.grad.data.zero_()

    def forward(self, input):
        orth = self.orthogonal_kernel[:self.input_size, :self.output_size]
        return input.matmul(orth)
