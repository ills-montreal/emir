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
        y, m = self.logpdf(x)
        return -torch.mean(y), m


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
        z, m = self.logpdf(z_c, z_d)
        return -torch.mean(z), m


class BaseCondKernelDelta(nn.Module):
    def __init__(self, args, zc_dim, zd_dim):
        super().__init__()
        self.d = zd_dim
        self.use_tanh = args.use_tanh
        self.optimize_mu = args.optimize_mu
        self.zc_dim = zc_dim
        pass

    def logpdf(self, z_c, z_d):
        raise NotImplementedError

    def forward(self, z_c, z_d, params_marg):
        z, m = self.logpdf(z_c, z_d, params_marg)
        return -torch.mean(z), m