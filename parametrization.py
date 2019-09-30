import torch
import torch.nn as nn
import gc


def get_parameters(model):
    parametrized_params = []

    def get_parametrized_params(mod):
        nonlocal parametrized_params
        if isinstance(mod, Parametrization):
            parametrized_params.append(mod._A)

    def not_in(elem, l):
        return all(elem is not x for x in l)

    model.apply(get_parametrized_params)
    normal_params = (param for param in model.parameters() if not_in(param, parametrized_params))
    return normal_params, parametrized_params


def parametrization_trick(model, loss):
    """ Monkey patching """
    backward = loss.backward
    def new_backward(*args, **kwargs):
        kwargs["retain_graph"] = True
        backward(*args, **kwargs)

        # Apply the backwards function to every Parametrized layer after applying loss.backward()
        def _backwards_param(mod):
            if isinstance(mod, Parametrization):
                mod.backwards_param()
        model.apply(_backwards_param)
    loss.backward = new_backward
    return loss


class Parametrization(nn.Module):
    """
    Implements the parametrization of a manifold in terms of a Euclidean space

    To use it, subclass it implement the method "retraction" (and optionally "project") when subclassing it.
    You can find an example below where we implement the Orthogonal class to optimize over the Stiefel manifold using an arbitrary retraction

    def retraction(self, raw_A, base):
        # raw_A: Square matrix of dimensions max(input_size, output_size) x max(input_size, output_size)
        # base: Matrix of dimensions output_size x input_size
        # It returns the retraction that we are using
        # It usually involves projection raw_A into the tangent space at base, and then computing the retraction
        # When dealing with Lie groups, raw_A is always projected into the Lie algebra, as an optimization.

    def project(self, base):
        # This method is OPTIONAL
        # base: Matrix of dimensions output_size x input_size
        # It returns the projected base back into the manifold

    """
    def __init__(self, input_size, output_size, initializer, mode):
        """
        initializer: (Tensor) -> Tensor. Initializes inplace the given tensor. It also returns it. Compatible with the initializers in torch.nn.init

        mode: "static" or a tuple such that:
                mode[0] == "dynamic"
                mode[1]: int, K, the number of steps after which we should change the basis of the dyn triv
                mode[2]: int, M, the number of changes of basis after which we should project back onto the manifold the basis. This is particularly helpful for small values of K.
        """
        super(Parametrization, self).__init__()
        assert mode == "static" or (isinstance(mode, tuple) and len(mode) == 3 and mode[0] == "dynamic")

        self.input_size = input_size
        self.output_size = output_size
        self.max_size = max(self.input_size, self.output_size)
        self._A = nn.Parameter(torch.empty(self.max_size, self.max_size))
        self.register_buffer("_B", torch.empty(self.input_size, self.output_size, requires_grad=True))
        self.register_buffer('base', torch.eye(self.max_size))
        self.initializer = initializer

        self.B_updated = False
        self.graph_computed = False

        if mode == "static":
            self.mode = mode
        else:
            self.mode = mode[0]
            self.K = mode[1]
            self.M = mode[2]
            self.k = 0
            self.m = 0

        self.first_base_update = self.mode == "static"
        self.reset_parameters()

    def reset_parameters(self):
        self.initializer(self._A)


    def rebase(self):
        with torch.no_grad():
            self.base = self.retraction(self._A, self.base).data
            self._A.data.zero_()

    def B(self):
        """
        Forward part of the paramtrization trick
        """

        # The first time we enter B, if it's a dynamic parametrization, we update the base
        # This line is the difference between static and dynamic with K = infty
        if not self.first_base_update and  self.mode == "dynamic":
            self.rebase()
            self.first_base_update = True

        # We compute the parametrization once per iteration, i.e., if:
            # We haven't updated it yet (the B_updated is a "dirty" flag)
            # We have computed it, but it was during test time (within a with torch.no_grad() clause)
                # In this case, we recompute it to have the graph of its derivative
        if not self.B_updated or (torch.is_grad_enabled() and not self.graph_computed):
            # Clean gradients from last iteration, as the variable is not managed by an optimizer
            # This is necessary becuase we are using retain_graph in the backwards function for efficiency
            if self._B.grad is not None:
                # Free the computation graph.
                del self._B
                # Calling the gc manually is necessary here to clean the graph.
                gc.collect()
            # Compute the parametrization B on the manifold and its derivative graph
            self._B = self.retraction(self._A, self.base)
            # We increment the dynamic trivialization counter whenever we compute B for the first time
            # after a gradient update, this is, whenever self.B_updated == False
            if self.mode == "dynamic" and not self.B_updated:
                if self.K != "infty":
                    # Change the basis after K optimization steps
                    self.k = (self.k + 1) % self.K
                    if self.k == 0:
                        self.rebase()
                        # Project the basis back to the manifold every M changes of basis
                        self.m = (self.m + 1) % self.M
                        if self.m == 0:
                            # It's optional to implement this method
                            if hasattr(self, "project"):
                                with torch.no_grad():
                                    self.base.copy_(self.project(self.base))
            # Now it's not a leaf tensor, but we convert it into a leaf
            self._B.retain_grad()
            # Update the "clean" flags
            self.B_updated = True
            self.graph_computed = torch.is_grad_enabled()

        return self._B

    def backwards_param(self):
        """ Computes the gradients with respect to the parametrization """
        self._A.grad = torch.autograd.grad([self._B], self._A, grad_outputs=(self._B.grad,))[0]
        self.B_updated=False

    def forward(self, input):
        return input.matmul(self.B())
