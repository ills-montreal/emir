"""
Courtesy of https://github.com/g-pichler/knife/blob/master/da_experiments/estimators/knife.py
From https://arxiv.org/abs/2202.06618
"""

import numpy as np
import torch
import torch.nn as nn


from .kernels import BaseMargKernel, BaseCondKernel
from .feed_forward import FF


class GaussianMargKernel(BaseMargKernel):
    """
    Used to compute p(z_d)
    """

    def __init__(self, args, zc_dim, zd_dim, init_samples=None, **kwargs):
        super().__init__(args, zc_dim, zd_dim)
        self.K = args.marg_modes if self.optimize_mu else args.batch_size
        self.init_std = args.init_std

        self.logC = (
            torch.tensor([-self.d / 2 * np.log(2 * np.pi)]).to(args.device).float()
        )

        if init_samples is None:
            init_samples = self.init_std * torch.randn((self.K, self.d))
        # self.means = nn.Parameter(torch.rand(self.K, self.d), requires_grad=True)
        if self.optimize_mu:
            self.means = nn.Parameter(init_samples, requires_grad=True)  # [K, db]
        else:
            self.means = nn.Parameter(init_samples, requires_grad=False)

        if args.cov_diagonal == "var":
            diag = self.init_std * torch.randn((1, self.K, self.d))
        else:
            diag = self.init_std * torch.randn((1, 1, self.d))
        self.logvar = nn.Parameter(diag, requires_grad=True)

        if args.cov_off_diagonal == "var":
            tri = self.init_std * torch.randn((1, self.K, self.d, self.d))
            tri = tri.to(init_samples.dtype)
            self.tri = nn.Parameter(tri, requires_grad=True)
        else:
            self.tri = None

        weigh = torch.ones((1, self.K))
        if args.average == "var":
            self.weigh = nn.Parameter(weigh, requires_grad=True)
        else:
            self.weigh = nn.Parameter(weigh, requires_grad=False)

        self.log_softmax = nn.LogSoftmax(dim=1)

    def logpdf(self, x):
        assert len(x.shape) == 2 and x.shape[1] == self.d, "x has to have shape [N, d]"
        x = x[:, None, :]
        w = self.log_softmax(self.weigh)
        y = x - self.means
        logvar = self.logvar
        if self.use_tanh:
            logvar = logvar.tanh()
        var = logvar.exp()
        y = y * var

        if self.tri is not None:
            y = y + torch.squeeze(
                torch.matmul(torch.tril(self.tri, diagonal=-1), y[:, :, :, None]), 3
            )
        y = torch.sum(y**2, dim=2)

        y = -y / 2 + torch.sum(torch.log(torch.abs(var) + 1e-8), dim=-1) + w
        y = torch.logsumexp(y, dim=-1)
        return self.logC + y

    def update_parameters(self, z):
        self.means = z


class GaussianCondKernel(BaseCondKernel):
    """
    Used to compute p(z_d | z_c)
    """

    def __init__(self, args, zc_dim, zd_dim, **kwargs):
        super().__init__(args, zc_dim, zd_dim)
        self.K = args.cond_modes
        self.logC = torch.tensor([-self.d / 2 * np.log(2 * np.pi)]).to(args.device)

        out_ff_dim = self.K * (2 * self.d + 1)  # mu: d, logvar: d, w: 1
        if args.cov_off_diagonal == "var":
            out_ff_dim += self.K * self.d**2
            self.tri = True
        else:
            self.tri = False

        self.ff = FF(args, zc_dim, self.ff_hidden_dim, out_ff_dim)
        self.tanh = nn.Tanh()

    def logpdf(self, z_c, z_d):  # H(z_d|z_c)
        N = z_c.shape[0]
        z_d = z_d.unsqueeze(1)  # [N, 1, d]
        ff_out = self.ff(z_c).view(z_c.shape[0], self.K, -1)  # [N, K*(2*d+1) + tri_dim]

        w = torch.log_softmax(ff_out[:, :, 0].squeeze(-1), dim=-1).reshape(
            N, -1
        )  # [N, K]
        mu = ff_out[:, :, 1 : self.d + 1]  # [N, K * d]
        logvar = ff_out[:, :, self.d + 1 : 2 * self.d + 1]
        if self.use_tanh:
            var = self.tanh(logvar).exp()
        else:
            var = logvar.exp()

        # print(f"Cond : {var.min()} | {var.max()} | {var.mean()}")

        z = z_d - mu  # [N, K, d]
        z = var * z
        if self.tri:
            tri = ff_out[:, :, -self.d**2 :].reshape(-1, self.K, self.d, self.d)
            z = z + torch.squeeze(
                torch.matmul(torch.tril(tri, diagonal=-1), z[:, :, :, None]), 3
            )
        z = torch.sum(z**2, dim=-1)  # [N, K]
        z = -z / 2 + torch.log(torch.abs(var) + 1e-8).sum(-1) + w
        z = torch.logsumexp(z, dim=-1)
        return self.logC + z
