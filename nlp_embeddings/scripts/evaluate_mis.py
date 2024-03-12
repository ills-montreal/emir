import torch
import numpy as np
from emir.estimators.knife_estimator import KNIFEEstimator, KNIFEArgs
import argparse
from typing import Tuple, List

import uuid
import re
from pathlib import Path
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_1", type=Path, required=True)
    parser.add_argument("--model_2", type=Path, required=True)

    parser.add_argument(
        "--dataset_filter", type=str, required=True, help="regex to filter datasets"
    )

    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--log_dir", type=Path, required=True)

    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch_size", type=int, default=32)

    return parser.parse_args()


def load_embeddings(
    model_1_path: Path, model_2_path: Path, dataset_filter: str
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    # get all paths to embeddings
    embeddings_1 = []
    for path in model_1_path.rglob("*.npy"):
        if re.search(dataset_filter, path.name):
            path = str(path)[len(str(model_1_path)) + 1 :]
            embeddings_1.append(path)

    embeddings_2 = []
    for path in model_2_path.rglob("*.npy"):
        if re.search(dataset_filter, path.name):
            path = str(path)[len(str(model_2_path)) + 1 :]
            embeddings_2.append(path)

    # make sure that the embeddings are in the same order and exists in both models
    embeddings_1 = set(sorted(embeddings_1))
    embeddings_2 = set(sorted(embeddings_2))

    embedding_paths = list(embeddings_1.intersection(embeddings_2))

    # load embeddings
    embeddings_1 = [np.load(model_1_path / emb) for emb in embedding_paths]
    embeddings_2 = [np.load(model_2_path / emb) for emb in embedding_paths]

    return torch.tensor(embeddings_1), torch.tensor(embeddings_2), embedding_paths


def main():
    unid_id = uuid.uuid4()
    date = pd.Timestamp.now().strftime("%Y-%m-%d")

    args = parse_args()
    device = torch.device(args.device)

    model_1_path = args.model_1
    model_2_path = args.model_2

    embeddings_1, embeddings_2, embeddings_paths = load_embeddings(
        model_1_path, model_2_path, args.dataset_filter
    )

    knife_args = KNIFEArgs(
        **{k: v for k, v in vars(args).items() if k in KNIFEArgs.__annotations__}
    )
    estimator = KNIFEEstimator(knife_args, embeddings_1.shape[1], embeddings_2.shape[1])

    mi, h1, h1h2, history = estimator.eval_per_sample(
        embeddings_1, embeddings_2, record_loss=True
    )
    steps = [i for i in range(len(history))]
    history = {"steps": steps, "loss": history}

    history = pd.DataFrame(history)
    history.to_csv(args.log_dir / f"history_{unid_id}.csv")

    df = pd.DataFrame({"I(X_1 -> X_2)": mi, "H(X_1)": h1, "H(X_1|X_2)": h1h2})
    df["id"] = unid_id
    df["date"] = date

    df.to_csv(args.output_dir / f"results_{unid_id}.csv")

    # metadata file
    knife_args_dict = {k: v for k, v in knife_args.__dict__.items()}

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

    metadata.to_csv(args.output_dir / f"metadata_{unid_id}.csv")


if __name__ == "__main__":
    main()
