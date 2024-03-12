import torch
import numpy as np
from emir.estimators.knife_estimator import KNIFEEstimator, KNIFEArgs
import argparse
<<<<<<< HEAD
<<<<<<< HEAD
from typing import Tuple, List, Optional
=======
from typing import Tuple, List
>>>>>>> 0edeba6 (merge knife estimator)
=======
from typing import Tuple, List, Optional
>>>>>>> 33a03ab (fix kinfe estimator)

import uuid
import re
from pathlib import Path
import pandas as pd
<<<<<<< HEAD
import wandb
=======
>>>>>>> 0edeba6 (merge knife estimator)


def parse_args():
    parser = argparse.ArgumentParser()
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 33a03ab (fix kinfe estimator)
    parser.add_argument(
        "--embeddings_dir", type=Path, required=True, default="../output"
    )
    parser.add_argument("--model_1", type=str, required=True)
    parser.add_argument("--model_2", type=str, required=True)
<<<<<<< HEAD

    parser.add_argument(
        "--dataset_filter", type=str, default=None, help="regex to filter datasets"
    )
=======
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_1", type=Path, required=True)
    parser.add_argument("--model_2", type=Path, required=True)
=======
>>>>>>> 33a03ab (fix kinfe estimator)

    parser.add_argument(
        "--dataset_filter", type=str, default=None, help="regex to filter datasets"
    )
<<<<<<< HEAD

>>>>>>> 0edeba6 (merge knife estimator)
=======
>>>>>>> 33a03ab (fix kinfe estimator)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--log_dir", type=Path, required=True)

    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 33a03ab (fix kinfe estimator)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=4096)

    # stoping criterion

<<<<<<< HEAD
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--n_epochs_margs", type=int, default=5)

=======
    parser.add_argument("--n_epochs", type=int, default=1000)
>>>>>>> 33a03ab (fix kinfe estimator)
    parser.add_argument(
        "--stopping_criterion", type=str, default="max_epochs"
    )  # "max_epochs" or "early_stopping"
    parser.add_argument("--eps", type=float, default=1e-7)

    parser.add_argument("--lr", type=float, default=0.001)
<<<<<<< HEAD
    parser.add_argument("--margin_lr", type=float, default=0.1)
=======
>>>>>>> 33a03ab (fix kinfe estimator)
    parser.add_argument("--average", type=str, default="")
    parser.add_argument("--cov_diagonal", type=str, default="")
    parser.add_argument("--cov_off_diagonal", type=str, default="")
    parser.add_argument("--optimize_mu", default=False, action="store_true")

    parser.add_argument("--cond_modes", type=int, default=8)
    parser.add_argument("--marg_modes", type=int, default=8)
    parser.add_argument("--use_tanh", default=True, action="store_true")
    parser.add_argument("--init_std", type=float, default=0.01)
    parser.add_argument("--ff_residual_connection", type=bool, default=False)
    parser.add_argument("--ff_activation", type=str, default="relu")
    parser.add_argument("--ff_layer_norm", default=True, action="store_true")
    parser.add_argument("--ff_layers", type=int, default=2)
<<<<<<< HEAD
    parser.add_argument("--ff_dim_hidden", type=int, default=0)

    parser.add_argument("--normalize_embeddings", action="store_true", default=False)
=======
    parser.add_argument("--batch_size", type=int, default=32)
>>>>>>> 0edeba6 (merge knife estimator)
=======

    parser.add_argument("--normalize_embeddings", action="store_true", default=False)
>>>>>>> 33a03ab (fix kinfe estimator)

    return parser.parse_args()


def load_embeddings(
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 33a03ab (fix kinfe estimator)
    model_1_path: Path,
    model_2_path: Path,
    dataset_filter: Optional[str] = None,
    normalize: bool = False,
<<<<<<< HEAD
=======
    model_1_path: Path, model_2_path: Path, dataset_filter: str
>>>>>>> 0edeba6 (merge knife estimator)
=======
>>>>>>> 33a03ab (fix kinfe estimator)
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    # get all paths to embeddings
    embeddings_1 = []
    for path in model_1_path.rglob("*.npy"):
<<<<<<< HEAD
<<<<<<< HEAD
        if dataset_filter is None or re.search(dataset_filter, path.name):
=======
        if re.search(dataset_filter, path.name):
>>>>>>> 0edeba6 (merge knife estimator)
=======
        if dataset_filter is None or re.search(dataset_filter, path.name):
>>>>>>> 33a03ab (fix kinfe estimator)
            path = str(path)[len(str(model_1_path)) + 1 :]
            embeddings_1.append(path)

    embeddings_2 = []
<<<<<<< HEAD
<<<<<<< HEAD
    for path in model_2_path.rglob("*embeddings.npy"):
        if dataset_filter is None or re.search(dataset_filter, path.name):
=======
    for path in model_2_path.rglob("*.npy"):
        if re.search(dataset_filter, path.name):
>>>>>>> 0edeba6 (merge knife estimator)
=======
    for path in model_2_path.rglob("*embeddings.npy"):
        if dataset_filter is None or re.search(dataset_filter, path.name):
>>>>>>> 33a03ab (fix kinfe estimator)
            path = str(path)[len(str(model_2_path)) + 1 :]
            embeddings_2.append(path)

    # make sure that the embeddings are in the same order and exists in both models
    embeddings_1 = set(sorted(embeddings_1))
    embeddings_2 = set(sorted(embeddings_2))

    embedding_paths = list(embeddings_1.intersection(embeddings_2))

    # load embeddings
<<<<<<< HEAD
<<<<<<< HEAD
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

    emb1, emb2 = torch.cat(embeddings_1), torch.cat(embeddings_2)

    if normalize:
        mean_1, std_1 = emb1.mean(0), emb1.std(0)
        emb1 = (emb1 - mean_1) / std_1

        mean_2, std_2 = emb2.mean(0), emb2.std(0)
        emb2 = (emb2 - mean_2) / std_2

    return emb1, emb2, embedding_paths
=======
    embeddings_1 = [np.load(model_1_path / emb) for emb in embedding_paths]
    embeddings_2 = [np.load(model_2_path / emb) for emb in embedding_paths]

    return torch.tensor(embeddings_1), torch.tensor(embeddings_2), embedding_paths
>>>>>>> 0edeba6 (merge knife estimator)
=======
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

    emb1, emb2 = torch.cat(embeddings_1), torch.cat(embeddings_2)

    if normalize:
        mean_1, std_1 = emb1.mean(0), emb1.std(0)
        emb1 = (emb1 - mean_1) / std_1

        mean_2, std_2 = emb2.mean(0), emb2.std(0)
        emb2 = (emb2 - mean_2) / std_2

    return emb1, emb2, embedding_paths
>>>>>>> 33a03ab (fix kinfe estimator)


def main():
    unid_id = uuid.uuid4()
<<<<<<< HEAD
<<<<<<< HEAD
    date = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
=======
    date = pd.Timestamp.now().strftime("%Y-%m-%d")
>>>>>>> 0edeba6 (merge knife estimator)
=======
    date = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
>>>>>>> 33a03ab (fix kinfe estimator)

    args = parse_args()
    device = torch.device(args.device)

<<<<<<< HEAD
<<<<<<< HEAD
    wandb.init(project="emir", config=vars(args))

    model_1_path = args.embeddings_dir / args.model_1
    model_2_path = args.embeddings_dir / args.model_2

    embeddings_1, embeddings_2, embeddings_paths = load_embeddings(
        model_1_path, model_2_path, args.dataset_filter, args.normalize_embeddings
=======
    model_1_path = args.model_1
    model_2_path = args.model_2

    embeddings_1, embeddings_2, embeddings_paths = load_embeddings(
        model_1_path, model_2_path, args.dataset_filter
>>>>>>> 0edeba6 (merge knife estimator)
=======
    model_1_path = args.embeddings_dir / args.model_1
    model_2_path = args.embeddings_dir / args.model_2

    embeddings_1, embeddings_2, embeddings_paths = load_embeddings(
        model_1_path, model_2_path, args.dataset_filter, args.normalize_embeddings
>>>>>>> 33a03ab (fix kinfe estimator)
    )

    knife_args = KNIFEArgs(
        **{k: v for k, v in vars(args).items() if k in KNIFEArgs.__annotations__}
    )
    estimator = KNIFEEstimator(knife_args, embeddings_1.shape[1], embeddings_2.shape[1])
<<<<<<< HEAD
<<<<<<< HEAD
    d_1 = embeddings_1.shape[1]
    d_2 = embeddings_2.shape[1]

    mi, h2, h2h1 = estimator.eval_per_sample(
        embeddings_1, embeddings_2, record_loss=True
    )

    history = estimator.recorded_loss
    margin_history = estimator.recorded_marg_ent
    cond_history = estimator.recorded_cond_ent

    for step, hh in enumerate(margin_history):
        wandb.log(
            {
                "margin_loss": hh,
                "margin_step": step,
            }
        )

    for step, hh in enumerate(cond_history):
        wandb.log(
            {
                "cond_loss": hh,
                "cond_step": step,
            }
        )

    df = pd.DataFrame({"I(X_1->X_2)": mi, "H(X_2)": h2, "H(X_2|X_1)": h2h1})
    df["id"] = unid_id
    df["date"] = date

    args.output_dir.mkdir(parents=True, exist_ok=True)
=======
=======
    d_1 = embeddings_1.shape[1]
    d_2 = embeddings_2.shape[1]
>>>>>>> 33a03ab (fix kinfe estimator)

    mi, h2, h2h1, history = estimator.eval_per_sample(
        embeddings_1, embeddings_2, record_loss=True
    )
    steps = [i for i in range(len(history))]
    history = {"steps": steps, "loss": history}

    history = pd.DataFrame(history)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(args.log_dir / f"history_{unid_id}.csv")

    df = pd.DataFrame({"I(X_1->X_2)": mi, "H(X_2)": h2, "H(X_2|X_1)": h2h1})
    df["id"] = unid_id
    df["date"] = date

<<<<<<< HEAD
>>>>>>> 0edeba6 (merge knife estimator)
=======
    args.output_dir.mkdir(parents=True, exist_ok=True)
>>>>>>> 33a03ab (fix kinfe estimator)
    df.to_csv(args.output_dir / f"results_{unid_id}.csv")

    # metadata file
    knife_args_dict = {k: v for k, v in knife_args.__dict__.items()}

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 33a03ab (fix kinfe estimator)
    metadata = pd.DataFrame.from_records(
        [
            {
                "id": unid_id,
                "date": date,
                "model_1": str(model_1_path),
                "model_2": str(model_2_path),
                "d_1": d_1,
                "d_2": d_2,
                "dataset_filter": args.dataset_filter,
                "datasets": embeddings_paths,
                **knife_args_dict,
            }
        ]
<<<<<<< HEAD
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

=======
    metadata = pd.DataFrame(
        {
            "id": unid_id,
            "date": date,
            "model_1": str(model_1_path),
            "model_2": str(model_2_path),
            "dataset_filter": args.dataset_filter,
            "datasets": embeddings_paths,
            **knife_args_dict,
        }
    )

>>>>>>> 0edeba6 (merge knife estimator)
=======
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

>>>>>>> 33a03ab (fix kinfe estimator)
    metadata.to_csv(args.output_dir / f"metadata_{unid_id}.csv")


if __name__ == "__main__":
    main()
