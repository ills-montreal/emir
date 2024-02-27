from typing import Literal, Tuple

from .kernels import BaseMargKernel, BaseCondKernel

from .gaussian_kernel import GaussianMargKernel, GaussianCondKernel
from .discrete_kernels import DiscreteCondKernel, DiscreteMargKernel
from .tanimoto_kernels import TanimotoCondKernel, TanimotoMargKernel


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
        elif kernel == "tanimoto":
            return TanimotoMargKernel(**kwargs), TanimotoCondKernel(**kwargs)
        else:
            raise ValueError(f"Kernel {kernel} not implemented")
