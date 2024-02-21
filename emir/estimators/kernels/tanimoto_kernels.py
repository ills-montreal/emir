"""
Courtesy of https://github.com/g-pichler/knife/blob/master/da_experiments/estimators/knife.py
From https://arxiv.org/abs/2202.06618
"""

import numpy as np
import torch
import torch.nn as nn

from ..distances import TanimotoDistance
from .feed_forward import FF
from .kernels import BaseMargKernel, BaseCondKernel, BaseCondKernelDelta
from line_profiler_pycharm import profile


class TanimotoMargKernel(BaseMargKernel):
    """
    Used to compute p(z_d)
    """

    def __init__(self, args, zc_dim, zd_dim, init_samples=None, **kwargs):
        super().__init__(args, zc_dim, zd_dim)
        self.kernel_temp = 100  # HARDCODED
        self.K = args.marg_modes if self.optimize_mu else args.batch_size
        self.tri = None

        self.init_std = (
            args.init_std
        )  # Hard coded for now to avoid too small std due to sigmoid

        if init_samples is None:
            init_samples = self.init_std * torch.randn((self.K, self.d))
        else:
            init_samples = (
                torch.log(init_samples / (1 - init_samples + 1e-8) + 1e-8)
                / self.kernel_temp
            )
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

        self.logvar = nn.Parameter(
            self.init_std * torch.randn((1, self.K)), requires_grad=True
        )

    @profile
    def logpdf(self, x):
        assert len(x.shape) == 2 and x.shape[1] == self.d, "x has to have shape [N, d]"
        x = x[:, None, :]
        w = torch.log_softmax(self.weigh, dim=1)
        mu = torch.sigmoid(self.means * self.kernel_temp)
        logvar = self.logvar
        if self.use_tanh:
            logvar = logvar.tanh()
        var = logvar.exp()*10

        # compute logpdf of exponential law on the distance
        mult = (x * mu).sum(dim=-1)
        diff = ((x - mu) ** 2).sum(dim=-1)
        dist = 1 - mult / (diff + mult + 1e-8)

        y = -dist * var + torch.log(var + 1e-8) + w
        y = torch.logsumexp(y, dim=-1)

        return y, mu

    def update_parameters(self, z):
        self.means = z


class TanimotoCondKernel(BaseCondKernel):
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
        self.mu = FF(args, zc_dim, self.d, self.K * zd_dim)
        self.weight = FF(args, zc_dim, self.d, self.K)
        self.logvar = FF(args, zc_dim, self.d, self.K)
        self.kernel_temp = 100
        self.distances = TanimotoDistance()
        self.logC = torch.tensor([-self.d / 2 * np.log(2 * np.pi)])

    @profile
    def logpdf(self, z_c, z_d):  # H(z_d|z_c)
        z_d = z_d[:, None, :]  # [N, 1, d]
        w = torch.log_softmax(self.weight(z_c), dim=-1)  # [N, K]
        mu = self.mu(z_c)
        mu = mu.reshape(-1, self.K, self.d)
        mu = torch.sigmoid(mu * self.kernel_temp)  # [N, K, d]
        logvar = self.logvar(z_c)
        if self.use_tanh:
            logvar = logvar.tanh()
        var = logvar.exp().reshape(-1, self.K)*10

        # compute logpdf of exponential law on the distance
        mask = torch.sigmoid(z_d * mu * 1000) * 2 - 1
        dist = 1 - (mask * (z_d + mu)).sum(dim=-1) / ((2 - mask) * (z_d + mu)).sum(
            dim=-1
        )
        y = -dist * var + torch.log(var + 1e-8) + w
        y = torch.logsumexp(y, dim=-1)

        return y, mu


class TanimotoCondKernelDelta(BaseCondKernelDelta):
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
        self.mu = FF(args, zc_dim, self.d, self.K * zd_dim)
        self.weight = FF(args, zc_dim, self.d, self.K)
        self.logvar = FF(args, zc_dim, self.d, self.K)
        self.kernel_temp = 100
        self.distances = TanimotoDistance()
        self.logC = torch.tensor([-self.d / 2 * np.log(2 * np.pi)])

    @profile
    def logpdf(self, z_c, z_d, params_marg):  # H(z_d|z_c)
        mu_marg = params_marg[0]
        logvar_marg = params_marg[1]
        weights_marg = params_marg[2]

        z_d = z_d[:, None, :]  # [N, 1, d]
        dw = self.weight(z_c)
        w = torch.log_softmax(dw, dim=-1)  # [N, K]

        dmu = self.mu(z_c)
        mu = mu_marg + dmu.reshape(-1, self.K, self.d)
        mu = torch.sigmoid(mu * self.kernel_temp)  # [N, K, d]

        dlogvar = self.logvar(z_c)
        logvar =  logvar_marg + dlogvar
        if self.use_tanh:
            logvar = logvar.tanh()
        var = logvar.exp().reshape(-1, self.K)*10

        # compute logpdf of exponential law on the distance
        mult = (z_d * mu).sum(dim=-1)
        diff = ((z_d - mu) ** 2).sum(dim=-1)
        dist = 1 - mult / (diff + mult + 1e-8)

        y = -dist * var + torch.log(var + 1e-8) + w
        y = torch.logsumexp(y, dim=-1)

        return y, mu
