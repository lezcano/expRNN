import torch
import torch.nn as nn
import gc

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


def get_parameters(model):
    """We get the orthogonal params first and then the others"""

    log_orth_param = []

    def get_log_orth_params(mod):
        nonlocal log_orth_param
        if isinstance(mod, Orthogonal):
            log_orth_param.append(mod.log_orthogonal_kernel)

    def not_in(elem, l):
        return all(elem is not x for x in l)

    model.apply(get_log_orth_params)
    non_orth_param = (param for param in model.parameters() if not_in(param, log_orth_param))
    return non_orth_param, log_orth_param


def parametrization_trick(model, loss):
    backward = loss.backward
    def new_backward():
        backward(retain_graph=True)
        backwards_param(model)
    loss.backward = new_backward
    return loss


def backwards_param(model):
    def _orthogonal_step(mod):
        if isinstance(mod, Orthogonal):
            mod.backwards_param()
    model.apply(_orthogonal_step)


class Orthogonal(nn.Module):
    """
    Implements a non-square linear with orthogonal colums
    """
    def __init__(self, input_size, output_size, skew_initializer):
        super(Orthogonal, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.max_size = max(self.input_size, self.output_size)
        self.log_orthogonal_kernel = nn.Parameter(torch.empty(self.max_size, self.max_size))
        self.register_buffer("_orthogonal_kernel", torch.empty(self.max_size, self.max_size, requires_grad=True))
        self.skew_initializer = skew_initializer
        self.orthogonal_updated = False
        self.graph_computed = False

        self.reset_parameters()

    def reset_parameters(self):
        self.log_orthogonal_kernel.data = \
                torch.as_tensor(self.skew_initializer(self.max_size),
                        dtype=self.log_orthogonal_kernel.dtype,
                        device=self.log_orthogonal_kernel.device)

    @property
    def orthogonal_kernel(self):
        """
        We compute the parametrization once per iteration
        This is the forward part of the paramtrization trick
        """
        if not self.orthogonal_updated or (torch.is_grad_enabled() and not self.graph_computed):
            # Clean gradients from last iteration, as the variable is not managed by an optimizer
            if self._orthogonal_kernel.grad is not None:
                # Free the computation graph.
                del self._orthogonal_kernel
                # Calling the gc manually is necessary here.
                gc.collect()
            # Computes the skew_symmetric matrix stored in log_orthogonal_kernel
            A = self.log_orthogonal_kernel
            A = A.triu(diagonal=1)
            A = A - A.t()
            # Computes the exponential of A
            # This could be the Cayley approximant or other parametrization
            # when implementing other parametrizations of SO(n)
            self._orthogonal_kernel = expm_skew(A)
            # Now it's not a leaf tensor, but we convert it into a leaf
            self._orthogonal_kernel.retain_grad()
            # Update the "clean" flags
            self.orthogonal_updated = True
            self.graph_computed = torch.is_grad_enabled()
        return self._orthogonal_kernel

    def backwards_param(self):
        self.log_orthogonal_kernel.grad = \
            torch.autograd.grad([self._orthogonal_kernel], self.log_orthogonal_kernel,
grad_outputs=(self._orthogonal_kernel.grad,))[0]
        self.orthogonal_updated=False

    def forward(self, input):
        B = self.orthogonal_kernel
        if self.input_size != self.output_size:
            B = B[:self.input_size, :self.output_size]
        return input.matmul(B)


class ExpRNN(nn.Module):
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

    def forward(self, input, hidden):
        input = self.input_kernel(input)
        hidden = self.recurrent_kernel(hidden)
        out = self.nonlinearity(input + hidden)

        return out, out

    def extra_repr(self):
        return 'input_size={}, hidden_size={}'.format(self.input_size, self.hidden_size)
