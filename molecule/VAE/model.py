from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
from sklearn.model_selection import train_test_split

from tqdm import trange


@dataclass(frozen=True)
class AEArgs:
    input_dim: int
    intermediate_dim: int = 256
    n_layers: int = 2
    latent_dims: int = 256
    device: str = "cuda"
    batch_size: int = 8192
    lr: float = 0.001
    dropout_rate: float = 0.1
    norm: str = "batch"
    residual: bool = True
    lmbd_kl: float = 1.0


class FF(nn.Module):
    def __init__(
        self,
        n_layers,
        dim_input,
        dim_hidden,
        dim_output,
        dropout_rate=0.1,
        norm="batch",
        residual=False,
    ):
        super(FF, self).__init__()
        self.num_layers = n_layers
        self.norm = norm
        self.residual = residual
        self.stack = nn.ModuleList()

        self.input = nn.Sequential(*[nn.Linear(dim_input, dim_hidden), nn.ReLU()])

        for l in range(self.num_layers):
            layer = []
            if self.norm == "batch":
                layer.append(nn.BatchNorm1d(dim_hidden))
            elif self.norm == "layer":
                layer.append(nn.LayerNorm(dim_hidden))

            layer.append(nn.Linear(dim_hidden, dim_hidden))
            layer.append(nn.ReLU())
            layer.append(nn.Dropout(dropout_rate))

            self.stack.append(nn.Sequential(*layer))

    def forward(self, x):
        x = self.input(x.float())
        for layer in self.stack:
            x = layer(x) + x if self.residual else layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, args: AEArgs):
        super(Encoder, self).__init__()
        self.linears = FF(
            args.n_layers,
            args.input_dim,
            args.intermediate_dim,
            args.intermediate_dim,
            dropout_rate=args.dropout_rate,
            norm=args.norm,
            residual=args.residual,
        ).to(args.device)

        self.out = nn.Sequential(
            *[
                nn.Linear(args.intermediate_dim, args.latent_dims, device=args.device),
                nn.LeakyReLU(),
            ]
        )

    def forward(self, x: torch.Tensor):
        x = self.linears(x)
        x = self.out(x)
        return x


class Decoder(nn.Module):
    def __init__(self, args: AEArgs):
        super(Decoder, self).__init__()
        self.linears = FF(
            args.n_layers,
            args.latent_dims,
            args.intermediate_dim,
            args.intermediate_dim,
            dropout_rate=args.dropout_rate,
            norm=args.norm,
            residual=args.residual,
        ).to(args.device)

        self.out = nn.Linear(args.intermediate_dim, args.input_dim, device=args.device)

    def forward(self, z: torch.Tensor):
        z = self.linears(z)
        z = self.out(z)
        return z


class AutoEncoder(nn.Module):
    def __init__(self, args: AEArgs):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.args = args
        self.loss = []
        self.val_loss = []

        self.decoder_param_norm = []
        self.encoder_param_norm = []

        self.use_sigmoid = False
        self.reconst_loss_fn = nn.MSELoss()

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
                self.reconst_loss_fn = nn.BCELoss()

        else:
            X = (X - X.mean()) / (X.std() + 1e-3)

        X_train, X_val = train_test_split(X, test_size=prop_val)
        train_size = X_train.shape[0]
        val_size = X_val.shape[0]

        opt = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
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
            epoch_loss = 0
            epoch_encoder_norm = []
            epoch_decoder_norm = []
            self.train()

            for x in data:
                x = x[0]
                opt.zero_grad()
                x_hat = self.forward(x)
                loss = self.reconst_loss_fn(x_hat, x)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * x.shape[0]
                epoch_encoder_norm.append(
                    torch.mean(
                        torch.tensor(
                            [p.grad.data.norm(2) for p in self.encoder.parameters()]
                        )
                    )
                )
                epoch_decoder_norm.append(
                    torch.mean(
                        torch.tensor(
                            [p.grad.data.norm(2) for p in self.decoder.parameters()]
                        )
                    )
                )
            self.loss.append(epoch_loss / train_size)
            self.encoder_param_norm.append(torch.mean(torch.tensor(epoch_encoder_norm)))
            self.decoder_param_norm.append(torch.mean(torch.tensor(epoch_decoder_norm)))

            val_loss = 0
            val_mse = 0
            val_kl = 0

            self.eval()
            with torch.no_grad():
                for x in data_val:
                    x = x[0]
                    x_hat = self.forward(x)
                    loss = self.reconst_loss_fn(x_hat, x)
                    val_loss += loss.item() * x.shape[0]

            self.val_loss.append(val_loss / val_size)

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
