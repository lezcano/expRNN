import torch
import torch.nn as nn


def get_parameters(model):
    parametrized_params = []

    def get_parametrized_params(mod):
        nonlocal parametrized_params
        if isinstance(mod, Parametrization):
            parametrized_params.append(mod.A)

    def not_in(elem, l):
        return all(elem is not x for x in l)

    model.apply(get_parametrized_params)
    unconstrained_params = (param for param in model.parameters() if not_in(param, parametrized_params))
    return unconstrained_params, parametrized_params


def parameters_updated(model):
    def del_b(mod):
        if isinstance(mod, Parametrization):
            del mod._B
    model.apply(del_b)


class Parametrization(nn.Module):
    """
    Implements the parametrization of a manifold in terms of a Euclidean space

    It gives the parametrized matrix through the attribute `B`

    To use it, subclass it and implement the method `retraction` and the method `forward` (and optionally `project`). See the documentation in these methods for details

    You can find an example in the file `orthogonal.py` where we implement the Orthogonal class to optimize over the Stiefel manifold using an arbitrary retraction
    """

    def __init__(self, A, base, init_A, init_base, mode):
        """
        initializer: (Tensor) -> Tensor. Initializes inplace the given tensor. It also returns it. Compatible with the initializers in torch.nn.init

        mode: "static" or a tuple such that:
                mode[0] == "dynamic"
                mode[1]: int, K, the number of steps after which we should change the basis of the dyn triv
                mode[2]: int, M, the number of changes of basis after which we should project back onto the manifold the basis. This is particularly helpful for small values of K.
        """
        super(Parametrization, self).__init__()
        assert mode == "static" or (isinstance(mode, tuple) and len(mode) == 3 and mode[0] == "dynamic")

        self.A = nn.Parameter(A)
        self.register_buffer("_B", torch.empty(A.size(), dtype=A.dtype, requires_grad=True))
        self.register_buffer('base', base)
        self.init_A = init_A
        self.init_base = init_base

        if mode == "static":
            self.mode = mode
        else:
            self.mode = mode[0]
            self.K = mode[1]
            self.M = mode[2]
            self.k = 0
            self.m = 0


    def reset_parameters(self):
        self.init_base(self.base)
        self.init_A(self.A)
        self._B = self.compute_B()
        if self.mode == "dynamic":
            self.rebase()
        # We have to do this, because apparently retain_gradient does not work during initialisation
        # self._B will be recomputed the first time that self.B is called
        del self._B


    def rebase(self):
        with torch.no_grad():
            self.base.data.copy_(self._B.data)
            self.A.data.zero_()

    def compute_B(self):
        # Compute the parametrization B on the manifold and its forward graph
        ret = self.retraction(self.A, self.base)
        # Now it's not a leaf tensor at the moment, so we convert it into a leaf
        ret.retain_grad()
        return ret

    @property
    def B(self):
        if not hasattr(self, "_B") or (not self._B.grad_fn and torch.is_grad_enabled()):
            has_B = hasattr(self, "_B")
            self._B = self.compute_B()

            # Increment the counters
            if self.mode == "dynamic" and not has_B:
                if self.K != "infty":
                    # Change the basis after K optimization steps
                    self.k = (self.k + 1) % self.K
                    if self.k == 0:
                        self.rebase()
                        # Project the base back to the manifold every M changes of base
                        self.m = (self.m + 1) % self.M
                        if self.m == 0:
                            # It's optional to implement this method
                            if hasattr(self, "project"):
                                with torch.no_grad():
                                    self.base.copy_(self.project(self.base))

        return self._B

    def retraction(self, A, base):
        """
        It computes r_{base}(A).
        Notice that A will not always be in the tangent space of our manifold
          For this reason, we first have to use A to parametrize the tangent space,
          and then compute the retraction
        When dealing with Lie groups, raw_A is always projected into the Lie algebra, as an optimization (cf. Section E in the paper)
        """
        raise NotImplementedError

    def project(self, base):
        """
        This method is OPTIONAL
        It returns the projected base back into the manifold
        """
        raise NotImplementedError

    def forward(self, input):
        """
        It uses the attribute self.B to implement the layer itself (e.g. Linear, CNN, ...)
        """
        raise NotImplementedError
