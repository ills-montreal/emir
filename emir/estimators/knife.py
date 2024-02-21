"""
Courtesy of https://github.com/g-pichler/knife/blob/master/da_experiments/estimators/knife.py
From https://arxiv.org/abs/2202.06618
"""

import numpy as np
import torch
import torch.nn as nn

from .kernels import KernelFactory


class KNIFE(nn.Module):
    def __init__(self, args, zc_dim, zd_dim, kernel_type="gaussian", reg_conf=1e1, init_samples=None, precomputed_marg_kernel=None):
        super(KNIFE, self).__init__()
        self.kernel_type = kernel_type
        self.kernel_marg, self.kernel_cond = KernelFactory(
            kernel=kernel_type, args=args, zc_dim=zc_dim, zd_dim=zd_dim, init_samples=init_samples
        )
        if precomputed_marg_kernel is not None:
            self.kernel_marg = precomputed_marg_kernel
        self.reg_conf = reg_conf

    def run_kernels(self, z_c, z_d):
        marg_ent, marg_means = self.kernel_marg(z_d)
        if "delta" in self.kernel_type:
            cond_ent, cond_means = self.kernel_cond(
                z_c,
                z_d,
                (self.kernel_marg.means, self.kernel_marg.logvar, self.kernel_marg.weigh, self.kernel_marg.tri),
            )
        else:
            cond_ent, cond_means = self.kernel_cond(z_c, z_d)
        return marg_ent, cond_ent, marg_means, cond_means

    def forward(self, z_c, z_d):  # samples have shape [sample_size, dim]
        marg_ent, cond_ent, _, _ = self.run_kernels(z_c, z_d)
        return marg_ent - cond_ent, marg_ent, cond_ent

    def learning_loss(self, z_c, z_d):
        marg_ent, cond_ent, marg_means, cond_means = self.run_kernels(z_c, z_d)

        # marg_means_avg_dist = torch.cdist(marg_means, marg_means).mean() / marg_means.shape[1]
        # cond_means_avg_dist = torch.cdist(cond_means, cond_means).mean() / cond_means.shape[1]

        reg_exp = 0  # - self.reg_conf * (marg_means_avg_dist + cond_means_avg_dist)
        return marg_ent + cond_ent, reg_exp, marg_ent, cond_ent

    def I(self, *args, **kwargs):
        return self.forward(*args[:2], **kwargs)[0]

    def freeze_marginal(self):
        self.kernel_marg.requires_grad_(False)

    def unfreeze_marginal(self):
        self.kernel_marg.requires_grad_(True)
