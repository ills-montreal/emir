""" Contains the VariationalAutoencoder class. Which is based on the implementation of the VAE from
Alexander Van de Kleut. The original implementation can be found at: https://avandekleut.github.io/vae/"""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
from sklearn.model_selection import train_test_split

from tqdm import trange


@dataclass(frozen=True)
class VAEArgs:
    input_dim: int
    intermediate_dim: int = 256
    n_layers: int = 2
    latent_dims: int = 256
    device: str = "cuda"
    batch_size: int = 8192
    lr: float = 0.001
    dropout_rate: float = 0.1


class FF(nn.Module):
    def __init__(self, n_layers, dim_input, dim_hidden, dim_output, dropout_rate=0.1):
        super(FF, self).__init__()
        self.num_layers = n_layers
        self.stack = nn.ModuleList()

        for l in range(self.num_layers):
            layer = []
            layer.append(nn.Linear(dim_input if l == 0 else dim_hidden, dim_hidden))
            layer.append(nn.ReLU())
            layer.append(nn.Dropout(dropout_rate))

            self.stack.append(nn.Sequential(*layer))

        self.out = nn.Linear(
            dim_input if self.num_layers < 1 else dim_hidden, dim_output
        )

    def forward(self, x):
        x = x.float()
        for layer in self.stack:
            x = layer(x)
        return self.out(x)


class VariationalEncoder(nn.Module):
    def __init__(self, args: VAEArgs):
        super(VariationalEncoder, self).__init__()
        self.linear1 = FF(
            args.n_layers, args.input_dim, args.intermediate_dim, args.intermediate_dim, dropout_rate=args.dropout_rate
        ).to(args.device)

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
        logvar = self.linear3(x)
        z = mu + (logvar / 2).exp() * self.N.sample(mu.shape)
        self.kl = -0.25 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return z

    def get_embedding(self, x: torch.Tensor):
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        return mu


class Decoder(nn.Module):
    def __init__(self, args: VAEArgs):
        super(Decoder, self).__init__()
        self.linear1 = FF(
            args.n_layers,
            args.latent_dims,
            args.intermediate_dim,
            args.intermediate_dim,
            dropout_rate=args.dropout_rate,
        ).to(args.device)
        self.linear2 = nn.Linear(
            args.intermediate_dim, args.input_dim, device=args.device
        )

    def forward(self, z: torch.Tensor):
        z = F.relu(self.linear1(z))
        z = self.linear2(z)
        return z


class VariationalAutoencoder(nn.Module):
    def __init__(self, args: VAEArgs):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(args)
        self.decoder = Decoder(args)
        self.args = args
        self.loss = []
        self.val_loss = []
        self.mse = []
        self.val_mse = []
        self.kl = []
        self.kl_val = []

        self.use_sigmoid = False
        self.reconst_loss_fn = nn.MSELoss(reduction="sum")

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        if self.use_sigmoid:
            return torch.sigmoid(self.decoder(z))
        return torch.tanh(self.decoder(z)) * 3

    def train_model(
        self,
        epochs: int,
        X: torch.Tensor,
        save_dir: str = None,
        prop_val: float = 0.1,
    ):
        if X.min() >= 0 and X.max() <= 1:
            self.use_sigmoid = True
            if (X.unique().cpu() == torch.tensor([0, 1])).all():
                self.reconst_loss_fn = nn.BCELoss(reduction="sum")

        else:
            X = (X - X.mean()) / X.std()

        X_train, X_val = train_test_split(X, test_size=prop_val)
        train_size = X_train.shape[0]
        val_size = X_val.shape[0]

        opt = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        data = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train.float()),
            batch_size=self.args.batch_size,
            shuffle=True,
        )
        data_val = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_val.float()),
            batch_size=self.args.batch_size,
            shuffle=False,
        )
        max_val_loss = float("inf")
        for epoch in trange(epochs, desc="VAE training"):
            epoch_loss = []
            epoch_mse = []
            epoch_kl = []
            self.train()

            for x in data:
                x = x[0]
                opt.zero_grad()
                x_hat = self.forward(x)
                mse = self.reconst_loss_fn(x_hat, x)
                loss = mse + self.encoder.kl
                loss.backward()
                opt.step()
                epoch_mse.append(mse.item())
                epoch_loss.append(loss.item())
                epoch_kl.append(self.encoder.kl.item())
            self.loss.append(sum(epoch_loss) / train_size)
            self.mse.append(sum(epoch_mse) / train_size)
            self.kl.append(sum(epoch_kl) / train_size)

            val_loss = []
            val_mse = []
            val_kl = []
            self.eval()

            for x in data_val:
                x = x[0]
                x_hat = self.forward(x)
                mse = self.reconst_loss_fn(x_hat, x)
                loss = mse + self.encoder.kl
                val_loss.append(loss.item())
                val_mse.append(mse.item())
                val_kl.append(self.encoder.kl.item())
            self.val_loss.append(sum(val_loss) / val_size)
            self.val_mse.append(sum(val_mse) / val_size)
            self.kl_val.append(sum(val_kl) / val_size)

            if save_dir is not None and self.val_loss[-1] < max_val_loss:
                max_val_loss = self.val_loss[-1]
                torch.save(self.state_dict(), f"{save_dir}/model.pt")

    @torch.no_grad()
    def get_embeddings(self, X: torch.Tensor):
        self.eval()
        data = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X.float()),
            batch_size=self.args.batch_size,
            shuffle=False,
        )
        embeddings = []
        for x in data:
            embeddings.append(self.encoder.get_embedding(x[0]))
        return torch.cat(embeddings)
