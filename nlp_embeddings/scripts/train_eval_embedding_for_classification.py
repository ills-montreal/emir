"""
Train small models that takes embeddings as input for classification tasks.
"""

import argparse
import os
from typing import Any

import pandas as pd
from lightning.pytorch.utilities.types import STEP_OUTPUT

# set wandb to offline mode
os.environ["WANDB_MODE"] = "offline"
# export WANDB_MODE=offline


from pathlib import Path
from emb_datasets import load_emd_classif_dataset, TASKS_DATASET
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


class EmbeddingClassificationTask(L.LightningModule):
    def __init__(self, model, output_dir, lr=0.001, device="cuda", **kwargs):
        super().__init__()

        self.model = model
        self.lr = lr
        self.output_dir = output_dir

        self.save_hyperparameters()

        self.test_results = []
        self.validation_results = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y_hat = self.model(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        y_hat = self.model(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log("val_loss", loss)

        preds = y_hat.argmax(dim=1)
        success = (preds == y).float()
        accuracy = success.mean()

        self.log("val_accuracy", accuracy)

        preds, y, success, y_hat = (
            preds.detach().cpu().tolist(),
            y.detach().cpu().tolist(),
            success.detach().cpu().tolist(),
            y_hat.detach().cpu().tolist(),
        )

        # zip the results
        for pred, yy, succ, yh in zip(preds, y, success, y_hat):
            self.test_results.append((pred, yy, succ, yh))

        return loss

    def on_train_epoch_end(self) -> None:
        self.logger.log_table(
            key="Validation results",
            columns=["preds", "y", "success", "logits"],
            data=self.validation_results,
            step=self.current_epoch,
        )
        self.validation_results = []

    def test_step(self, batch, batch_idx):
        x, y = batch

        x, y = x.to(self.device), y.to(self.device)

        y_hat = self.model(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)

        self.log("test_loss", loss)

        preds = y_hat.argmax(dim=1)
        success = (preds == y).float()
        accuracy = success.mean()

        self.log("test_accuracy", accuracy)

        preds, y, success, y_hat = (
            preds.detach().cpu().tolist(),
            y.detach().cpu().tolist(),
            success.detach().cpu().tolist(),
            y_hat.detach().cpu().tolist(),
        )

        # zip the results
        for pred, yy, succ, yh in zip(preds, y, success, y_hat):
            self.test_results.append((pred, yy, succ, yh))

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, nargs="+", required=True)
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

    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    return parser.parse_args()


# command line example:
# python train_eval_embedding_for_classification.py --dataset rotten_tomatoes --model infgrad/stella-base-en-v2 --embeddings_dir classification_embeddings --n_epochs 5 --lr 1e-3 --hidden_dim 256 --num_layers 2 --device cuda --test


def main():
    args = parse_args()
    wandb.init(
        project="emir-nlp-embeddings-classification",
        config=vars(args) | {"model_name": args.model},
    )
    unique_id = wandb.run.id
    # make output dir
    args.output_dir.mkdir(exist_ok=True, parents=True)

    _embeddings = []
    for model in args.model:
        base_dir = Path(args.embeddings_dir) / model / args.dataset

        # list directories in base_dir
        splits = [d.name for d in base_dir.iterdir() if d.is_dir()]

        # load embeddings
        embeddings = {
            split: torch.tensor(torch.load(base_dir / split / "embeddings.npy"))
            for split in splits
        }

        _embeddings.append(embeddings)

    embeddings = {}
    for split in splits:
        embeddings[split] = torch.cat(
            [_embeddings[i][split] for i in range(len(args.model))], dim=1
        )

    labels = {
        split: torch.tensor(torch.load(base_dir / split / "labels.npy")).view(-1)
        for split in splits
    }

    n_labels = len(torch.unique(labels["train"]))

    # zip the embeddings and labels
    data_loaders = {
        split: FastTensorDataLoader(
            embeddings[split],
            labels[split],
            batch_size=32,
            shuffle=(True if split == "train" else False),
        )
        for split in splits
    }

    train_loader = data_loaders["train"]

    if "validation" in splits:
        val_loader = data_loaders["validation"]
    else:
        val_loader = None

    if "test" in splits:
        val_loader = data_loaders["test"]
    else:
        val_loader = None

    # create the model
    model = EmbeddingClassificationModel(
        input_dim=embeddings["train"].shape[1],
        output_dim=n_labels,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(args.device)

    # create the task
    task = EmbeddingClassificationTask(
        model, lr=args.lr, device=args.device, output_dir=args.output_dir
    ).to(args.device)

    # logger:
    wandb_logger = WandbLogger()

    # checkpointng
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"logs/{wandb.run.id}",
        filename="best_model",
        save_top_k=1,
        mode="min",
    )

    # create the trainer
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
    trainer.test(task, val_loader)

    # log table with test results
    wandb_logger.log_table(
        key="Classification results",
        columns=["preds", "y", "success", "logits"],
        data=task.test_results,
    )

    # make dataframe with test results
    df = pd.DataFrame(task.test_results, columns=["preds", "y", "success", "logits"])
    df.to_csv(args.output_dir / f"classification_results_{unique_id}.csv")

    # metadata
    metadata = pd.DataFrame.from_records([vars(args)])
    metadata.to_csv(args.output_dir / f"metadata_{unique_id}.csv")


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    torch.TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a Fasttorch.TensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A Fasttorch.TensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i : self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __getitem__(self, item):
        return tuple(t[item] for t in self.tensors)

    def __len__(self):
        return self.n_batches


if __name__ == "__main__":
    main()
