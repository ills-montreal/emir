""" Contains the VariationalAutoencoder class. Which is based on the implementation of the VAE from
Alexander Van de Kleut. The original implementation can be found at: https://avandekleut.github.io/vae/"""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions

from tqdm import trange


@dataclass(frozen=True)
class VAEArgs:
    input_dim: int
    intermediate_dim: int = 256
    latent_dims: int = 256
    device: str = "cuda"
    batch_size: int = 8192
    lr: float = 0.001


class VariationalEncoder(nn.Module):
    def __init__(self, args: VAEArgs):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(
            args.input_dim, args.intermediate_dim, device=args.device
        )
        self.linear2 = nn.Linear(
            args.intermediate_dim, args.latent_dims, device=args.device
        )
        self.linear3 = nn.Linear(
            args.intermediate_dim, args.latent_dims, device=args.device
        )

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x: torch.Tensor):
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1 / 2).mean()
        return z

    def get_embedding(self, x: torch.Tensor):
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        return mu


class Decoder(nn.Module):
    def __init__(self, args: VAEArgs):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(
            args.latent_dims, args.intermediate_dim, device=args.device
        )
        self.linear2 = nn.Linear(
            args.intermediate_dim, args.input_dim, device=args.device
        )

    def forward(self, z: torch.Tensor):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z


class VariationalAutoencoder(nn.Module):
    def __init__(self, args: VAEArgs):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(args)
        self.decoder = Decoder(args)
        self.args = args
        self.loss = []

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return self.decoder(z)

    def train(
        self,
        epochs: int,
        X: torch.Tensor,
    ):
        opt = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        data = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X.float()),
            batch_size=self.args.batch_size,
            shuffle=True,
        )
        for epoch in trange(epochs, desc="VAE training"):
            for x in data:
                x = x[0]
                opt.zero_grad()
                x_hat = self.forward(x)
                loss = ((x - x_hat) ** 2).mean() + self.encoder.kl
                loss.backward()
                opt.step()
                self.loss.append(loss.item())

    @torch.no_grad()
    def get_embeddings(self, X: torch.Tensor):
        data = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X.float()),
            batch_size=self.args.batch_size,
            shuffle=False,
        )
        embeddings = []
        for x in data:
            embeddings.append(self.encoder.get_embedding(x[0]))
        return torch.cat(embeddings)
