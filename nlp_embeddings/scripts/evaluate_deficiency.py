import argparse
import re
import uuid
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
import wandb

from emir.estimators.deficiency_estimator import (
    GANTrickedDeficiencyEstimator,
    GANDeficiencyArgs,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings_dir", type=Path, required=True, default="../output"
    )
    parser.add_argument("--model_Y", type=str, required=True)
    parser.add_argument("--model_X", type=str, required=True)

    parser.add_argument(
        "--dataset_filter", type=str, default=None, help="regex to filter datasets"
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--log_dir", type=Path, required=True)

    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    parser.add_argument("--normalize_embeddings", action="store_true", default=False)
    parser.add_argument("--fixed_embeddings", default="Common")

    parser.add_argument("--gan_batch_size", type=int, default=64)
    parser.add_argument("--gan_n_epochs", type=int, default=100)
    parser.add_argument("--critic_repeats", type=int, default=5)

    parser.add_argument("--gen_hidden_dim", type=int, default=16)
    parser.add_argument("--gen_n_layers", type=int, default=5)
    parser.add_argument("--gen_lr", type=float, default=0.0001)

    parser.add_argument("--critic_hidden_dim", type=int, default=16)
    parser.add_argument("--critic_n_layers", type=int, default=3)
    parser.add_argument("--critic_lr", type=float, default=0.001)

    parser.add_argument("--noise_dim", type=int, default=16)
    parser.add_argument("--noise_std", type=float, default=0.1)

    return parser.parse_args()


def find_embedding_paths(
    model_1_path: Path,
    model_2_paths: List[Path],
    dataset_filter: Optional[str] = None,
) -> Optional[List[str]]:
    # get all paths to embeddings
    N_ds = 12

    embeddings_1 = []
    for path in model_1_path.rglob("*embeddings.npy"):
        if dataset_filter is None or re.search(dataset_filter, path.name):
            path = str(path)[len(str(model_1_path)) + 1 :]
            embeddings_1.append(path)

    if len(embeddings_1) == 0:
        print(f"No datasets found for {model_1_path}")
        print(f"Ignore")
        return

    if len(embeddings_1) < N_ds:
        print("Ignoring model 1 because of too few datasets")
        return

    embeddings_2 = []

    for model_2_path in model_2_paths:
        _embeddings_2 = []
        for path in model_2_path.rglob("*embeddings.npy"):
            if dataset_filter is None or re.search(dataset_filter, path.name):
                path = str(path)[len(str(model_2_path)) + 1 :]
                _embeddings_2.append(path)

        if len(_embeddings_2) == 0:
            print(f"No datasets found for {model_2_path}")
            print(f"Ignore")
            continue
        if len(embeddings_2) < N_ds:
            print("Ignoring model 2 because of too few datasets")
            continue

        print(f"Found {len(_embeddings_2)} datasets for {model_2_path}")
        embeddings_2.append(_embeddings_2)

    print(embeddings_1)
    print(embeddings_2)
    common_datasets = set(embeddings_1).intersection(*embeddings_2)

    if len(common_datasets) == 0:
        raise ValueError("No common datasets found")
    if len(common_datasets) < N_ds:
        print("Ignoring because of too few common datasets")
        return

    embedding_paths = list(common_datasets)

    return embedding_paths


def load_embeddings(common_datasets, model_path, normalize) -> torch.Tensor:
    # load embeddings
    embeddings = []

    for emb in common_datasets:
        try:
            emb = torch.tensor(torch.load(model_path / emb))
            embeddings.append(emb)
            print(emb.shape)
        except:
            print(f"Error loading {model_path / emb}")

    embeddings = torch.cat(embeddings)

    # nprmalize embeddings
    if normalize:
        mean, std = embeddings.mean(0), embeddings.std(0)
        embeddings = (embeddings - mean) / std

    return embeddings


def main():
    args = parse_args()
    device = torch.device(args.device)

    wandb.init(project="emir", config=vars(args))

    model_Y_path = args.embeddings_dir / args.model_Y
    model_X_path = args.embeddings_dir / args.model_X

    if args.fixed_embeddings == "Common":
        embeddings_paths = {
            # "dennlinger/wiki-paragraphs/validation/embeddings.npy",
            "mteb/amazon_polarity/test/embeddings.npy",
            "mteb/banking77/test/embeddings.npy",
            "mteb/biosses-sts/test/embeddings.npy",
            "mteb/sickr-sts/test/embeddings.npy",
            "mteb/sts12-sts/test/embeddings.npy",
            "mteb/sts13-sts/test/embeddings.npy",
            "mteb/sts14-sts/test/embeddings.npy",
            "mteb/sts15-sts/test/embeddings.npy",
            "mteb/stsbenchmark-sts/test/embeddings.npy",
            "mteb/stsbenchmark-sts/validation/embeddings.npy",
            "snli/test/embeddings.npy",
            "snli/validation/embeddings.npy",
        }
    elif args.fixed_embeddings == "All":
        embeddings_paths = find_embedding_paths(
            model_Y_path, [model_X_path], args.dataset_filter
        )
    else:
        embeddings_paths = {args.fixed_embeddings}

    print(f"Embeddings paths: {embeddings_paths}, len: {len(embeddings_paths)}")

    if embeddings_paths is None:
        return

    embeddings_X = load_embeddings(
        embeddings_paths, model_X_path, args.normalize_embeddings
    )

    deficiency_args = GANDeficiencyArgs(
        **{
            k: v
            for k, v in vars(args).items()
            if k in GANDeficiencyArgs.__annotations__
        }
    )

    # eval Y, Y and retrieve the precomputed marg_kernel
    estimator = eval_deficiency(
        embeddings_X,
        embeddings_X,
        model_Y_path,
        model_Y_path,
        embeddings_paths,
        args,
        deficiency_args,
    )


def eval_deficiency(
    embeddings_X,
    embeddings_Y,
    model_X_path,
    model_Y_path,
    dataset_list,
    args,
    deficiency_args: GANDeficiencyArgs,
):
    unique_id = uuid.uuid4()
    date = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    device = torch.device(args.device)

    # clean cuda memory
    torch.cuda.empty_cache()

    estimator = GANTrickedDeficiencyEstimator(
        deficiency_args,
        embeddings_X.shape[1],
        embeddings_Y.shape[1],
    )

    deficiency = estimator.eval(embeddings_X, embeddings_Y)

    d_1 = embeddings_X.shape[1]
    d_2 = embeddings_Y.shape[1]

    # metadata file
    deficiency_args_dict = {k: v for k, v in deficiency_args.__dict__.items()}

    metadata = pd.DataFrame.from_records(
        [
            {
                "id": unique_id,
                "date": date,
                "model_1": str(model_X_path),
                "model_2": str(model_Y_path),
                "d_1": d_1,
                "d_2": d_2,
                "dataset_filter": args.dataset_filter,
                "datasets": dataset_list,
                "d(X_1 -> X_2)": deficiency**deficiency_args_dict,
            }
        ]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(args.output_dir / f"metadata_{unique_id}.csv")

    if model_X_path == model_Y_path:
        for step, hh in enumerate(estimator.recorded_gen_loss):
            wandb.log(
                {
                    f"gen_loss": hh,
                }
            )

    for step, hh in enumerate(estimator.recorded_disc_loss):
        wandb.log(
            {
                f"disc_loss": hh,
            }
        )

    return estimator


if __name__ == "__main__":
    main()
