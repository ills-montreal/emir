"""
Courtesy of https://github.com/g-pichler/knife/blob/master/da_experiments/estimators/knife.py
From https://arxiv.org/abs/2202.06618
"""

import numpy as np
import torch
import torch.nn as nn

from .kernels import KernelFactory


class KNIFE(nn.Module):
    def __init__(
        self,
        args,
        zc_dim,
        zd_dim,
        kernel_type="gaussian",
        init_samples=None,
        precomputed_marg_kernel=None,
    ):
        super(KNIFE, self).__init__()
        self.kernel_type = kernel_type
        self.kernel_marg, self.kernel_cond = KernelFactory(
            kernel=kernel_type,
            args=args,
            zc_dim=zc_dim,
            zd_dim=zd_dim,
            init_samples=init_samples,
        )
        # put precomputed kernels to device
        self.kernel_cond = self.kernel_cond.to(args.device)
        self.kernel_marg = self.kernel_marg.to(args.device)

        if precomputed_marg_kernel is not None:
            self.kernel_marg = precomputed_marg_kernel.to(args.device)

    def run_kernels(self, z_c, z_d):
        marg_ent = self.kernel_marg(z_d)
        cond_ent = self.kernel_cond(z_c, z_d)
        return marg_ent, cond_ent

    def forward(self, z_c, z_d):  # samples have shape [sample_size, dim]
        marg_ent, cond_ent = self.run_kernels(z_c, z_d)
        return marg_ent - cond_ent, marg_ent, cond_ent

    def forward_samples(self, z_c, z_d):  # samples have shape [sample_size, dim]
        marg_ent = -self.kernel_marg.logpdf(z_d)
        cond_ent = -self.kernel_cond.logpdf(z_c, z_d)

        return marg_ent - cond_ent, marg_ent, cond_ent

    def learning_loss(self, z_c, z_d):
        marg_ent, cond_ent = self.run_kernels(z_c, z_d)
        return marg_ent + cond_ent

    def pmi(self, z_c, z_d):
        marg_ent = -self.kernel_marg.logpdf(z_d)
        cond_ent = -self.kernel_cond.logpdf(z_c, z_d)

        return marg_ent - cond_ent

    def I(self, *args, **kwargs):
        return self.forward(*args[:2], **kwargs)[0]

    def freeze_marginal(self):
        self.kernel_marg.requires_grad_(False)

    def unfreeze_marginal(self):
        self.kernel_marg.requires_grad_(True)
