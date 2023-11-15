from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as DataLoader2
from torch.utils.data import Dataset
import datamol as dm
import torch
from tqdm import trange
import matplotlib.pyplot as plt

from moleculenet_encoding import mol_to_graph_data_obj_simple
from utils import get_embeddings_from_model_moleculenet, get_molfeat_descriptors
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def get_features(
    dataloader,
    smiles,
    mols=None,
    model_path="backbone_pretrained_models/GraphLog/Contextual.pth",
    desc_name="rdkit",
    length=512,
):
    if not model_path == "None":
        e = get_embeddings_from_model_moleculenet(
            dataloader=dataloader, smiles=smiles, path=model_path
        )
    if not desc_name == "None":
        fp = get_molfeat_descriptors(
            dataloader, smiles, desc_name, mols=mols, length=length
        )

    if model_path == "None":
        return fp
    if desc_name == "None":
        return e
    return torch.cat([e, fp], dim=1)


def get_dataloaders(
    smiles_train,
    y_train,
    smiles_test,
    y_test,
    mols_train=None,
    mols_test=None,
    desc_name="rdkit",
    model_path="backbone_pretrained_models/GraphLog/Contextual.pth",
    length=512,
    batch_size=128,
):
    if not model_path is "None":
        dataloader_train = DataLoader(
            [
                mol_to_graph_data_obj_simple(dm.to_mol(s_i), y_i, s_i)
                for s_i, y_i in zip(smiles_train, y_train)
            ],
            batch_size=64,
            shuffle=False,
        )
        dataloader_test = DataLoader(
            [
                mol_to_graph_data_obj_simple(
                    dm.to_mol(
                        s_i,
                    ),
                    y_i,
                    s_i,
                )
                for s_i, y_i in zip(smiles_test, y_test)
            ],
            batch_size=64,
            shuffle=False,
        )
    else:
        dataloader_train = None
        dataloader_test = None
    z_train = get_features(
        dataloader_train,
        smiles_train,
        mols=mols_train,
        model_path=model_path,
        desc_name=desc_name,
        length=length,
    )
    z_test = get_features(
        dataloader_test,
        smiles_test,
        mols=mols_test,
        desc_name=desc_name,
        model_path=model_path,
        length=length,
    )
    input_dim = z_train.shape[1]

    dataset_train = torch.utils.data.TensorDataset(
        z_train.float(), torch.tensor(y_train)
    )
    dataset_test = torch.utils.data.TensorDataset(z_test.float(), torch.tensor(y_test))
    dataloader_train = DataLoader2(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader2(dataset_test, batch_size=batch_size, shuffle=False)
    return dataloader_train, dataloader_test, input_dim


class Feed_forward(torch.nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, n_layers, d_rate=0.3, norm="batch"
    ):
        super(Feed_forward, self).__init__()
        self.train_loss = []
        self.test_loss = []
        self.test_acc = []
        self.test_aucpr = []
        self.test_f1 = []
        self.test_roc = []
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        self.norm_fn = torch.nn.BatchNorm1d if norm == "batch" else torch.nn.LayerNorm

        self.inp_layer = torch.nn.Sequential(
            torch.nn.Dropout(d_rate),
            self.norm_fn(input_dim),
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Dropout(d_rate),
            self.norm_fn(hidden_dim),
        )

        self.hidden_layers = torch.nn.ModuleList()
        for i in range(n_layers):
            self.hidden_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.hidden_layers.append(torch.nn.Dropout(d_rate))
            self.hidden_layers.append(torch.nn.ReLU())
            self.hidden_layers.append(self.norm_fn(hidden_dim))

        self.out_layer = torch.nn.Linear(hidden_dim, output_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = self.inp_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.out_layer(x)
        return x

    def train_epoch(self, dataloader_train):
        loss_epoch = []
        self.train()
        for batch in dataloader_train:
            self.optimizer.zero_grad()
            x, y = batch
            y_hat = self(x)
            loss_batch = self.loss_fn(y_hat.squeeze(1), y.float())
            loss_batch.backward()
            self.optimizer.step()
            loss_epoch.append(loss_batch.item())
        self.train_loss.append(sum(loss_epoch) / len(loss_epoch))

    @torch.no_grad()
    def evaluate(self, dataloader_test):
        self.eval()
        y_hat_test = []
        y_true = []
        loss_test = []
        for batch in dataloader_test:
            x, y = batch
            y_hat = self(x)
            loss_test.append(self.loss_fn(y_hat.squeeze(1), y.float()))
            y_hat_test.append(torch.sigmoid(y_hat))
            y_true.append(y)
        self.test_loss.append(sum(loss_test) / len(loss_test))
        y_hat_test = torch.cat(y_hat_test).squeeze(1)
        y_true = torch.cat(y_true)
        self.test_acc.append(
            ((y_hat_test > 0.5).float() == y_true).float().mean().item()
        )
        self.test_aucpr.append(
            average_precision_score(
                y_true.detach().cpu().numpy(), y_hat_test.detach().cpu().numpy()
            )
        )
        self.test_f1.append(
            f1_score(
                y_true.detach().cpu().numpy(),
                (y_hat_test.detach().cpu().numpy() > 0.5).astype(int),
            )
        )
        self.test_roc.append(
            roc_auc_score(
                y_true.detach().cpu().numpy(), y_hat_test.detach().cpu().numpy()
            )
        )

    def train_model(self, dataloader_train, dataloader_test, n_epochs=100):
        for _ in range(n_epochs):
            self.train_epoch(dataloader_train)
            self.evaluate(dataloader_test)

    def plot_loss(self, title=""):
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes[0].plot(self.train_loss, label="train")
        axes[0].plot(self.test_loss, label="test")
        axes[0].set_title("Loss evolution over the training")
        axes[0].set_ylabel("Loss")
        axes[0].set_xlabel("Epoch")

        axes[0].legend()
        axes[1].plot(self.test_acc, label="test")
        axes[1].set_title("Accuracy evolution over the training")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

        fig.suptitle(title)
        plt.show()
