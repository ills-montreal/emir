"""
Courtesy of https://github.com/g-pichler/knife/blob/master/da_experiments/estimators/knife.py
From https://arxiv.org/abs/2202.06618
"""

import numpy as np
import torch
import torch.nn as nn

from .feed_forward import FF
from .kernels import BaseMargKernel, BaseCondKernel

class DiscreteMargKernel(BaseMargKernel):
    """
    Used to compute p(z_d)
    """

    def __init__(self, args, zc_dim, zd_dim, init_samples=None, **kwargs):
        super().__init__(args, zc_dim, zd_dim)
        self.kernel_temp = 10  # HARDCODED
        self.K = args.marg_modes if self.optimize_mu else args.batch_size
        self.init_std = (
            args.init_std * 10
        )  # Hard coded for now to avoid too small std due to sigmoid

        if init_samples is None:
            init_samples = self.init_std * torch.randn((self.K, self.d))
        # self.means = nn.Parameter(torch.rand(self.K, self.d), requires_grad=True)
        if self.optimize_mu:
            self.means = nn.Parameter(init_samples, requires_grad=True)  # [K, db]
        else:
            self.means = nn.Parameter(init_samples, requires_grad=False)

        weigh = torch.ones((1, self.K))
        if args.average == "var":
            self.weigh = nn.Parameter(weigh, requires_grad=True)
        else:
            self.weigh = nn.Parameter(weigh, requires_grad=False)

    def logpdf(self, x):
        assert len(x.shape) == 2 and x.shape[1] == self.d, "x has to have shape [N, d]"
        x = x[:, None, :]
        w = torch.log_softmax(self.weigh, dim=1)
        mu = torch.sigmoid(self.means * self.kernel_temp)
        y = x * torch.log(mu + 1e-8) + (1 - x) * torch.log(1 - mu + 1e-8)  # [N, K, d]
        y = torch.logsumexp(y.sum(dim=-1) + w, dim=-1)
        return y

    def update_parameters(self, z):
        self.means = z



class DiscreteCondKernel(BaseCondKernel):
    """
    Used to compute p(z_d | z_if args.cov_off_diagonal == "var":
            tri = self.init_std * torch.randn((1, self.K, self.d, self.d))
            tri = tri.to(init_samples.dtype)
            self.tri = nn.Parameter(tri, requires_grad=True)
        else:
            self.tri = Nonec)
    """

    def __init__(self, args, zc_dim, zd_dim, layers=1, **kwargs):
        super().__init__(args, zc_dim, zd_dim)
        self.K = args.cond_modes
        self.mu = FF(args, zc_dim, self.ff_hidden_dim, self.K * zd_dim)
        self.weight = FF(args, zc_dim, self.ff_hidden_dim, self.K)
        self.kernel_temp = 10

    def logpdf(self, z_c, z_d):  # H(z_d|z_c)
        z_d = z_d[:, None, :]  # [N, 1, d]
        w = torch.log_softmax(self.weight(z_c), dim=-1)  # [N, K]
        mu = self.mu(z_c)
        mu = mu.reshape(-1, self.K, self.d)
        mu = torch.sigmoid(mu * self.kernel_temp)  # [N, K, d]
        z = z_d * torch.log(mu + 1e-8) + (1 - z_d) * torch.log(
            1 - mu + 1e-8
        )  # [N, K, d]
        z = torch.logsumexp(z.sum(dim=-1) + w, dim=-1)
        return z

