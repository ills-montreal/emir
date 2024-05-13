"""
Train small models that takes embeddings as input for classification tasks.
"""


import argparse
import os
import re
from datetime import datetime
from typing import Tuple, Optional, List
import numpy as np

import pandas as pd

# set wandb to offline mode
os.environ["WANDB_MODE"] = "offline"
# export WANDB_MODE=offline


from pathlib import Path
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import torch

import wandb


class EmbeddingClassificationModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=2):
        super().__init__()

        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)
        self.output_proj = torch.nn.Linear(hidden_dim, output_dim)

        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.output_proj(x)
        return x


class EmbeddingCrossRegressionTask(L.LightningModule):
    def __init__(self, model, output_dir, lr=0.001, device="cuda", **kwargs):
        super().__init__()

        self.model = model
        self.lr = lr
        self.output_dir = output_dir

        self.save_hyperparameters()
        self.validation_results = []
        self.test_results = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y_hat = self.model(x)

        metrics = self._compute_metrics(y_hat, y)

        loss = metrics["l2"].sum(-1).mean()

        self.log("train_loss", loss)

        for m, v in metrics.items():
            self.log(f"train_{m}", v.sum(-1).mean())

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        y_hat = self.model(x)

        metrics = self._compute_metrics(y_hat, y)

        loss = metrics["l2"].sum(-1).mean()

        self.log("val_loss", loss)

        for m, v in metrics.items():
            self.log(f"val_{m}", v.sum(-1).mean())

        # make records of the results
        metrics_names = list(metrics.keys())
        N = len(y)
        for i in range(N):
            record = {name: metrics[name][i].sum().item() for name in metrics_names}
            self.validation_results.append(record)

        return metrics

    def on_validation_start(self) -> None:
        self.validation_results = []

    def on_validation_end(self) -> None:
        # log table
        df = pd.DataFrame(self.validation_results)
        self.logger.log_table(
            key="validation_results", dataframe=df, step=self.global_step
        )

    def on_test_end(self) -> None:
        # log table
        df = pd.DataFrame(self.test_results)
        self.logger.log_table(key="test_results", dataframe=df, step=self.global_step)

        # dump csv for the test results
        df["id"] = wandb.run.id
        df.to_csv(self.output_dir / f"predictions.csv")

    def on_test_start(self) -> None:
        self.test_results = []

    def test_step(self, batch, batch_idx):
        x, y = batch

        x, y = x.to(self.device), y.to(self.device)
        y_hat = self.model(x)

        metrics = self._compute_metrics(y_hat, y)

        loss = metrics["l2"].sum(-1).mean()

        self.log("test_loss", loss)

        for m, v in metrics.items():
            self.log(f"test_{m}", v.sum(-1).mean())

        # make records of the results
        metrics_names = list(metrics.keys())
        N = len(y)
        for i in range(N):
            record = {name: metrics[name][i].sum().item() for name in metrics_names}
            self.test_results.append(record)

        return metrics

    def _compute_metrics(self, y_hat, y):
        return {
            "l2": torch.nn.functional.mse_loss(y_hat, y, reduction="none"),
            "l1": torch.nn.functional.l1_loss(y_hat, y, reduction="none"),
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)


def load_embeddings(
    model_1_path: Path, model_2_path: Path, dataset_filter: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    # get all paths to embeddings
    embeddings_1 = []
    for path in model_1_path.rglob("*.npy"):
        if dataset_filter is None or re.search(dataset_filter, path.name):
            path = str(path)[len(str(model_1_path)) + 1 :]
            embeddings_1.append(path)

    embeddings_2 = []
    for path in model_2_path.rglob("*.npy"):
        if dataset_filter is None or re.search(dataset_filter, path.name):
            path = str(path)[len(str(model_2_path)) + 1 :]
            embeddings_2.append(path)

    # make sure that the embeddings are in the same order and exists in both models
    embeddings_1 = set(sorted(embeddings_1))
    embeddings_2 = set(sorted(embeddings_2))

    embedding_paths = list(embeddings_1.intersection(embeddings_2))

    # load embeddings
    embeddings_1 = []
    embeddings_2 = []

    for emb in embedding_paths:
        try:
            emb1 = torch.tensor(torch.load(model_1_path / emb))
            embeddings_1.append(emb1)
            print(emb1.shape)
        except:
            print(f"Error loading {model_1_path / emb}")

    for emb in embedding_paths:
        try:
            emb2 = torch.tensor(torch.load(model_1_path / emb))
            embeddings_2.append(emb2)
            print(emb2.shape)
        except:
            print(f"Error loading {model_2_path / emb}")

    return torch.cat(embeddings_1), torch.cat(embeddings_2), embedding_paths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_1", type=str, required=True)
    parser.add_argument("--model_2", type=str, required=True)
    parser.add_argument("--embeddings_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--training_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=512)

    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)

    # model architecture
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)

    # test mode
    parser.add_argument("--test", action="store_true", default=False)

    # seed
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    return parser.parse_args()


# python train_eval_embedding_for_cross_embeddings_prediction.py --model_1 infgrad/stella-base-en-v2 --model_2 WhereIsAI/UAE-Large-V1 --embeddings_dir output --n_epochs 5 --lr 1e-5 --hidden_dim 256 --num_layers 2 --device cuda


def main():
    args = parse_args()
    wandb.init(
        project="emir-nlp-cross-embeddings-prediction",
        config=vars(args),
    )
    # make output dir
    args.output_dir.mkdir(exist_ok=True, parents=True)

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_1_path = Path(args.embeddings_dir) / args.model_1
    model_2_path = Path(args.embeddings_dir) / args.model_2

    embeddings_1, embeddings_2, embedding_paths = load_embeddings(
        model_1_path, model_2_path
    )

    # make dataset
    dataset = torch.utils.data.TensorDataset(embeddings_1, embeddings_2)
    # split dataset into train, validation, test randomly
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.training_batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.eval_batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.eval_batch_size
    )

    # create model
    model = EmbeddingClassificationModel(
        input_dim=embeddings_1.shape[1],
        output_dim=embeddings_2.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )

    # create task
    task = EmbeddingCrossRegressionTask(
        model, output_dir=Path(args.output_dir), lr=args.lr, device=args.device
    )

    # create logger
    wandb_logger = WandbLogger()

    # callbacks
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor="val_l2",
        dirpath=f"logs/{wandb.run.id}",
        filename="best_model",
        save_top_k=1,
        mode="min",
    )

    # create trainer
    trainer = L.Trainer(
        max_epochs=args.n_epochs if not args.test else 1,
        logger=wandb_logger,
        log_every_n_steps=1,
        default_root_dir=f"logs/{wandb.run.id}",
        accelerator="cuda" if args.device == "cuda" else "cpu",
        devices=1,
        callbacks=[checkpoint_callback],
    )

    # fit the model
    trainer.fit(task, train_loader, val_loader)

    # test the model
    trainer.test(task, test_loader)

    date = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata = {
        "date": date,
        **vars(args),
    }
    metadata = pd.DataFrame.from_records([metadata])
    metadata.to_csv(Path(args.output_dir) / "metadata.csv")


if __name__ == "__main__":
    main()
