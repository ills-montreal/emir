from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Dict
import torch
from tqdm import trange
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, roc_auc_score, r2_score


def get_dataloaders(
    split_emb: Dict[str, Dict[str, torch.Tensor]],
    batch_size: int = 512,
        test_batch_size: int = 4096,
):
    input_dim = split_emb["train"]["x"].shape[1]

    dataset_train = torch.utils.data.TensorDataset(
        split_emb["train"]["x"].float(), split_emb["train"]["y"]
    )
    dataset_val = torch.utils.data.TensorDataset(
        split_emb["valid"]["x"].float(), split_emb["valid"]["y"]
    )
    dataset_test = torch.utils.data.TensorDataset(
        split_emb["test"]["x"].float(), split_emb["test"]["y"]
    )

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=test_batch_size, shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=test_batch_size, shuffle=False)
    return dataloader_train, dataloader_val, dataloader_test, input_dim


@dataclass(frozen=True)
class FFConfig:
    hidden_dim: int = 128
    n_layers: int = 1
    d_rate: float = 0.3
    norm: str = "batch"
    lr: float = 0.001
    batch_size: int = 512
    test_batch_size: int = 4096
    n_epochs: int = 100
    device: str = "cpu"


class Feed_forward(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        config: FFConfig,
        task: str = "classification",
        out_train_mean: int = 0,
        out_train_std: int = 1,
        task_size: int = 1,
    ):
        super(Feed_forward, self).__init__()
        self.config = config
        self.task_size = task_size

        self.train_loss = []
        self.val_loss = []
        self.val_roc = []
        self.r2_val = []
        self.task = task

        self.out_train_mean = out_train_mean
        self.out_train_std = out_train_std

        if self.task == "classification":
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        elif self.task == "regression":
            self.loss_fn = torch.nn.MSELoss()

        else:
            raise ValueError("task must be either classification or regression")

        self.norm_fn = (
            torch.nn.BatchNorm1d if self.config.norm == "batch" else torch.nn.LayerNorm
        )
        if self.config.n_layers == 0:
            self.inp_layer = torch.nn.Sequential(
                torch.nn.Linear(input_dim, output_dim, device=self.config.device)
            )
        elif self.config.n_layers >= 1:
            self.inp_layer = torch.nn.Sequential(
                torch.nn.Linear(
                    input_dim, self.config.hidden_dim, device=self.config.device
                ),
                torch.nn.Dropout(self.config.d_rate),
                self.norm_fn(self.config.hidden_dim, device=self.config.device),
                torch.nn.ReLU(),
            )
            if self.config.n_layers > 1:
                self.hidden_layers = torch.nn.ModuleList()
                for i in range(self.config.n_layers):
                    self.hidden_layers.append(
                        torch.nn.Linear(
                            self.config.hidden_dim,
                            self.config.hidden_dim,
                            device=self.config.device,
                        )
                    )
                    self.hidden_layers.append(torch.nn.Dropout(self.config.d_rate))
                    self.hidden_layers.append(torch.nn.ReLU())
                    self.hidden_layers.append(
                        self.norm_fn(self.config.hidden_dim, device=self.config.device)
                    )

            self.out_layer = torch.nn.Linear(
                self.config.hidden_dim, output_dim, device=self.config.device
            )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)

    def forward(self, x):
        x = self.inp_layer(x)
        if self.config.n_layers == 1:
            x = self.out_layer(x)
        elif self.config.n_layers > 1:
            for layer in self.hidden_layers:
                x = layer(x)
            x = self.out_layer(x)

        if self.task == "regression":
            x = x + self.out_train_mean
        return x

    def train_epoch(self, dataloader_train):
        loss_epoch = []
        self.train()
        for batch in dataloader_train:
            self.optimizer.zero_grad()
            x, y = batch
            x = x.to(self.config.device)
            y = y.to(self.config.device)
            y_hat = self(x)
            loss_batch = self.loss_fn(y_hat.squeeze(1), y.float())
            loss_batch.backward()
            self.optimizer.step()
            loss_epoch.append(loss_batch.item())
        self.train_loss.append(sum(loss_epoch) / len(loss_epoch))

    @torch.no_grad()
    def evaluate(self, dataloader_val: DataLoader, record: bool = True):
        self.eval()
        y_hat_val = []
        y_true = []
        loss_val = []
        for batch in dataloader_val:
            x, y = batch
            x = x.to(self.config.device)
            y = y.to(self.config.device)
            y_hat = self(x)
            loss_val.append(self.loss_fn(y_hat.squeeze(1), y.float()))
            if self.task == "classification":
                y_hat_val.append(torch.sigmoid(y_hat))
            else:
                y_hat_val.append(y_hat)
            y_true.append(y)

        self.val_loss.append(sum(loss_val).item() / len(loss_val))
        y_hat_val = torch.cat(y_hat_val).squeeze(1)
        y_true = torch.cat(y_true)

        if self.task == "classification":
            y_true = y_true.cpu()
            y_hat_val = y_hat_val.cpu()
            self.val_roc.append(
                roc_auc_score(
                    y_true, y_hat_val
                )
            )
        else:
            self.r2_val.append(
                r2_score(
                    y_true.detach().cpu().numpy(), y_hat_val.detach().cpu().numpy()
                )
            )

    def train_model(self, dataloader_train, dataloader_val, p_bar_name="Epochs"):
        n_epochs = min(int(self.config.n_epochs * 5000 / self.task_size)+1, self.config.n_epochs)
        for _ in trange(n_epochs, desc=p_bar_name, leave=False):
            self.train_epoch(dataloader_train)
            self.evaluate(dataloader_val)


    def plot_loss(self, title=""):
        fig, axess = plt.subplots(2, 2, figsize=(7, 7))
        axes = axess.flatten()

        axes[0].plot(self.train_loss, label="train")
        axes[0].plot(self.val_loss, label="val")
        axes[0].set_title("Loss evolution over the training")
        axes[0].set_ylabel("Loss")
        axes[0].set_xlabel("Epoch")

        axes[0].legend()
        axes[1].plot(self.val_acc, label="val")
        axes[1].set_title("Accuracy evolution over the training")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()


        axes[3].plot(self.val_roc, label="val")
        axes[3].set_title("ROC evolution over the training")
        axes[3].set_ylabel("ROC")
        axes[3].set_xlabel("Epoch")
        axes[3].legend()

        fig.suptitle(title)
        fig.tight_layout()
        plt.show()



class FF_trainer():
    def __init__(self, model):
        self.model = model
        self.best_ckpt = model.state_dict()
        self.best_metric = 0

    def train_model(self, dataloader_train, dataloader_val, p_bar_name="Epochs"):
        n_epochs = min(int(self.model.config.n_epochs * 5000 / self.model.task_size)+1, self.model.config.n_epochs)
        for i in trange(n_epochs, desc=p_bar_name, leave=False):
            self.model.train_epoch(dataloader_train)
            self.model.evaluate(dataloader_val)

            if self.model.task == "classification" and i > 0:
                if self.model.val_roc[-1] > max(self.model.val_roc[:-1]):
                    self.best_ckpt = self.model.state_dict()
                    self.best_metric = self.model.val_roc[-1]
            elif self.model.task == "regression" and i > 0:
                if self.model.r2_val[-1] > max(self.model.r2_val[:-1]):
                    self.best_ckpt = self.model.state_dict()
                    self.best_metric = self.model.r2_val[-1]

    def eval_on_test(self, dataloader_test):
        self.model.load_state_dict(self.best_ckpt)
        self.model.evaluate(dataloader_test, record=False)
        return self.model.val_roc[-1] if self.model.task == "classification" else self.model.r2_val[-1]
