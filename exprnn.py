import torch
import torch.nn as nn

from expm import expm, expm_frechet, expm_skew


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


def orthogonal_step(orthogonal_optimizer):
    def _orthogonal_step(mod):
        if isinstance(mod, Orthogonal):
            mod.orthogonal_step(orthogonal_optimizer)
    return _orthogonal_step


def get_parameters(model):
    """We get the orthogonal params first and then the others"""

    orth_param = []
    log_orth_param = []

    def get_orth_params(mod):
        nonlocal orth_param
        nonlocal log_orth_param
        if isinstance(mod, Orthogonal):
            orth_param.append(mod.orthogonal_kernel)
            log_orth_param.append(mod.log_orthogonal_kernel)

    def not_in(elem, l):
        return all(elem is not x for x in l)

    # Note that orthogonal params are not included in any of the sets, as they are not optimized
    # directly, but just through the log_orth_params
    model.apply(get_orth_params)
    non_orth_param = (param for param in model.parameters() if not_in(param, orth_param) and not_in(param, log_orth_param))
    return non_orth_param, log_orth_param


class Orthogonal(nn.Module):
    """
    Implements a non-square linear with orthogonal colums
    """
    def __init__(self, input_size, output_size, skew_initializer):
        super(Orthogonal, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.max_size = max(self.input_size, self.output_size)
        self.log_orthogonal_kernel = nn.Parameter(torch.Tensor(self.max_size, self.max_size))
        self.orthogonal_kernel = nn.Parameter(torch.Tensor(self.max_size, self.max_size))
        self.skew_initializer = skew_initializer

        self.reset_parameters()

    def reset_parameters(self):
        self.log_orthogonal_kernel.data = \
                torch.as_tensor(self.skew_initializer(self.max_size),
                        dtype=self.log_orthogonal_kernel.dtype,
                        device=self.log_orthogonal_kernel.device)
        self.orthogonal_kernel.data = self._B(gradients=False).data

    def _A(self, gradients):
        A = self.log_orthogonal_kernel
        if not gradients:
            A = A.data
        A = A.triu(diagonal=1)
        return A - A.t()

    def _B(self, gradients):
        # This could be the Cayley approximant or other parametrization
        # when implementing other parametrizations of SO(n)
        return expm_skew(self._A(gradients))

    def orthogonal_step(self, optim):
        orth_param = self._B(gradients=True)
        self.log_orthogonal_kernel.grad = \
            torch.autograd.grad([orth_param], self.log_orthogonal_kernel,
grad_outputs=(self.orthogonal_kernel.grad,))[0]

        optim.step()
        self.orthogonal_kernel.data = self._B(gradients=False)
        self.orthogonal_kernel.grad.data.zero_()

    def forward(self, input):
        B = self.orthogonal_kernel
        if self.input_size != self.output_size:
            B = B[:self.input_size, :self.output_size]
        out = input.matmul(B)
        return out



class ExpRNN(torch.jit.ScriptModule):
    def __init__(self, input_size, hidden_size, skew_initializer):
        super(ExpRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_kernel = Orthogonal(hidden_size, hidden_size, skew_initializer)
        self.input_kernel = nn.Linear(in_features=self.input_size, out_features=self.hidden_size, bias=False)
        self.nonlinearity = modrelu(hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.input_kernel.weight.data, nonlinearity="relu")

    def default_hidden(self, input):
        return input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)

    @torch.jit.script_method
    def forward(self, input, hidden):
        input = self.input_kernel(input)
        hidden = self.recurrent_kernel(hidden)
        out = self.nonlinearity(input + hidden)

        return out, out

    def extra_repr(self):
        return 'input_size={}, hidden_size={}'.format(self.input_size, self.hidden_size)
