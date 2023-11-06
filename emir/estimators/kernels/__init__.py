from typing import Literal, Tuple
from .gaussian_kernel import GaussianMargKernel, GaussianCondKernel
from .discrete_kernels import DiscreteCondKernel, DiscreteMargKernel
from .kernels import BaseMargKernel, BaseCondKernel


class KernelFactory:
    def __new__(
        cls,
        kernel: Literal["gaussian", "discrete"],
        **kwargs,
    ) -> Tuple[BaseMargKernel, BaseCondKernel]:
        if kernel == "gaussian":
            return GaussianMargKernel(**kwargs), GaussianCondKernel(**kwargs)
        elif kernel == "discrete":
            return DiscreteMargKernel(**kwargs), DiscreteCondKernel(**kwargs)
        else:
            raise ValueError(f"Kernel {kernel} not implemented")
