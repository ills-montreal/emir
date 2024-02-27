from torch import nn
import torch

class BaseMargKernel(nn.Module):
    def __init__(self, args, zc_dim, zd_dim):
        super().__init__()
        self.d = zd_dim
        self.use_tanh = args.use_tanh
        self.optimize_mu = args.optimize_mu
        self.zc_dim = zc_dim
        pass

    def logpdf(self, x):
        raise NotImplementedError

    def forward(self,x):
        y = self.logpdf(x)
        return -torch.mean(y)


class BaseCondKernel(nn.Module):
    def __init__(self, args, zc_dim, zd_dim):
        super().__init__()
        self.d = zd_dim
        self.use_tanh = args.use_tanh
        self.optimize_mu = args.optimize_mu
        self.zc_dim = zc_dim
        pass

    def logpdf(self, z_c, z_d):
        raise NotImplementedError

    def forward(self, z_c, z_d):
        z = self.logpdf(z_c, z_d)
        return -torch.mean(z)

