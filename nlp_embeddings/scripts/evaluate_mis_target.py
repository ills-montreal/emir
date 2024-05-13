import argparse
import re
import uuid
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
import wandb
import gc

from tqdm import tqdm

from emir.estimators.knife_estimator import KNIFEEstimator, KNIFEArgs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings_dir", type=Path, required=True, default="../output"
    )
    parser.add_argument("--model_Y", type=str, required=True)
    parser.add_argument("--model_X", type=str, nargs="+", required=True)

    parser.add_argument(
        "--dataset_filter", type=str, default=None, help="regex to filter datasets"
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--log_dir", type=Path, required=True)

    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=4096)

    # stoping criterion

    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--n_epochs_margs", type=int, default=5)

    parser.add_argument(
        "--stopping_criterion", type=str, default="max_epochs"
    )  # "max_epochs" or "early_stopping"
    parser.add_argument("--eps", type=float, default=1e-7)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--margin_lr", type=float, default=0.1)
    parser.add_argument("--average", type=str, default="var")
    parser.add_argument("--cov_diagonal", type=str, default="var")
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
    parser.add_argument("--ff_dim_hidden", type=int, default=0)

    parser.add_argument("--normalize_embeddings", action="store_true", default=False)
    parser.add_argument("--fixed_embeddings", default="Common")

    return parser.parse_args()


def find_embedding_paths(
    model_1_path: Path,
    model_2_paths: List[Path],
    dataset_filter: Optional[str] = None,
) -> List[str]:
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
    model_X_paths = [args.embeddings_dir / model for model in args.model_X]

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
            model_Y_path, model_X_paths, args.dataset_filter
        )
    else:
        embeddings_paths = {args.fixed_embeddings}

    print(f"Embeddings paths: {embeddings_paths}, len: {len(embeddings_paths)}")

    if embeddings_paths is None:
        return

    embeddings_Y = load_embeddings(
        embeddings_paths, model_Y_path, args.normalize_embeddings
    )

    knife_args = KNIFEArgs(
        **{k: v for k, v in vars(args).items() if k in KNIFEArgs.__annotations__}
    )

    # eval Y, Y and retrieve the precomputed marg_kernel
    estimator = eval_mis_target(
        embeddings_Y,
        embeddings_Y,
        model_Y_path,
        model_Y_path,
        embeddings_paths,
        args,
        knife_args,
    )

    if len(model_X_paths) == 1 and model_X_paths[0] == model_Y_path:
        return

    # eval X, Y
    for model_X_path in tqdm(model_X_paths):
        try:
            embeddings_X = load_embeddings(
                embeddings_paths, model_X_path, args.normalize_embeddings
            )
            eval_mis_target(
                embeddings_X,
                embeddings_Y,
                model_X_path,
                model_Y_path,
                embeddings_paths,
                args,
                knife_args,
                precomputed_marg_kernel=estimator.knife.kernel_marg,
            )

            del embeddings_X
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error with {model_X_path}")
            print(e)
            gc.collect()
            torch.cuda.empty_cache()
            continue


def eval_mis_target(
    embeddings_X,
    embeddings_Y,
    model_X_path,
    model_Y_path,
    dataset_list,
    args,
    knife_args: KNIFEArgs,
    precomputed_marg_kernel=None,
):
    unique_id = uuid.uuid4()
    date = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    device = torch.device(args.device)

    # clean cuda memory
    torch.cuda.empty_cache()

    estimator = KNIFEEstimator(
        knife_args,
        embeddings_X.shape[1],
        embeddings_Y.shape[1],
        precomputed_marg_kernel=precomputed_marg_kernel,
    )

    mi, hY, hYX = estimator.eval(embeddings_X, embeddings_Y)

    d_1 = embeddings_X.shape[1]
    d_2 = embeddings_Y.shape[1]

    mi_sample, h2_sample, h2h1_sample = estimator.eval_per_sample(
        embeddings_X, embeddings_Y
    )

    df = pd.DataFrame(
        {"I(X_1->X_2)": mi_sample, "H(X_2)": h2_sample, "H(X_2|X_1)": h2h1_sample}
    )
    df["id"] = unique_id
    df["date"] = date

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_dir / f"results_{unique_id}.csv")

    # metadata file
    knife_args_dict = {k: v for k, v in knife_args.__dict__.items()}

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
                "I(X_1->X_2)": mi,
                "H(X_2)": hY,
                "H(X_2|X_1)": hYX,
                **knife_args_dict,
            }
        ]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(args.output_dir / f"metadata_{unique_id}.csv")

    if model_X_path == model_Y_path:
        for step, hh in enumerate(estimator.recorded_marg_ent):
            wandb.log(
                {
                    f"margin_loss": hh,
                }
            )

    for step, hh in enumerate(estimator.recorded_cond_ent):
        wandb.log(
            {
                f"{model_X_path}_cond_loss": hh,
            }
        )

    return estimator


if __name__ == "__main__":
    main()
