import torch
import torch.nn as nn

from exp_numpy import expm, expm_frechet
from initialization import henaff_init


class modrelu(nn.Module):
    def __init__(self, features):
        # For now we just support square layers
        super(modrelu, self).__init__()
        self.features = features
        self.b = nn.Parameter(torch.Tensor(self.features))
        self.reset_parameters()

    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.01)

    def forward(self, inputs):
        norm = torch.abs(inputs)
        biased_norm = norm + self.b
        magnitude = nn.functional.relu(biased_norm)
        phase = torch.sign(inputs)

        return phase * magnitude


class ExpRNN(nn.Module):
    def __init__(self, input_size, hidden_size, nonlinearity=modrelu, exponential="exact", skew_initializer=henaff_init):
        super(ExpRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Can be optimised for size but it's not really worth it in most applications
        self.log_recurrent_kernel = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.recurrent_kernel = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.input_kernel = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.skew_initializer = skew_initializer

        self.exact = exponential == "exact"
        if self.exact:
            self.exp = expm
        else:
            self.exp = exponential
        if nonlinearity:
            try:
                self.nonlinearity = nonlinearity(hidden_size)
            except TypeError:
                self.nonlinearity = nonlinearity
        else:
            self.nonlinearity = None
        self.reset_parameters()

    def _A(self, gradients):
        A = self.log_recurrent_kernel
        if not gradients:
            A = A.data
        A = A.triu(diagonal=1)
        return A - A.t()

    def _B(self, gradients):
        return self.exp(self._A(gradients))

    def reset_parameters(self):
        self.log_recurrent_kernel.data = \
                torch.as_tensor(self.skew_initializer(self.hidden_size),
                        dtype=self.log_recurrent_kernel.dtype,
                        device=self.log_recurrent_kernel.device)
        self.recurrent_kernel.data = self._B(gradients=False)
        nn.init.kaiming_normal_(self.input_kernel.weight.data, nonlinearity="relu")

    def orthogonal_step(self, optim):
        if self.exact:
            A = self._A(gradients=False)
            B = self.recurrent_kernel.data
            G = self.recurrent_kernel.grad.data
            BtG = B.t().mm(G)
            grad = 0.5*(BtG - BtG.t())
            frechet_deriv = B.mm(expm_frechet(-A, grad))

            # Account for the triangular representation of the skew-symmetric matrix
            # The gradient with respect to the parameter x_i,j is the upper part of the triangular matrix
            # minus the lower part of the triangular matrix.
            # The gradient needn't be triangular, but we make it triangular for consistency
            self.log_recurrent_kernel.grad = (frechet_deriv - frechet_deriv.t()).triu(diagonal=1)
        else:
            orth_param = self._B(gradients=True)
            self.log_recurrent_kernel.grad = \
                torch.autograd.grad([orth_param], self.log_recurrent_kernel,
                                    grad_outputs=(self.recurrent_kernel.grad,))[0]
        optim.step()
        self.recurrent_kernel.data = self._B(gradients=False)
        self.recurrent_kernel.grad.data.zero_()

    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)

        input = self.input_kernel(input)

        hidden = hidden.matmul(self.recurrent_kernel)

        out = input + hidden

        if self.nonlinearity:
            return self.nonlinearity(out)
        else:
            return out

    def extra_repr(self):
        return 'input_size={}, hidden_size={}'.format(self.input_size, self.hidden_size)
