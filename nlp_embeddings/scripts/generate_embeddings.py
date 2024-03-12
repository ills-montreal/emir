<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 33a03ab (fix kinfe estimator)
from emb_datasets import (
    load_emb_dataset,
    AVAILABLE_DATASETS,
    TASKS_DATASET,
    load_emd_classif_dataset,
)
<<<<<<< HEAD
import argparse
import torch
import numpy as np
=======
from emb_datasets import load_emb_dataset, AVAILABLE_DATASETS
import argparse
import torch
>>>>>>> 0edeba6 (merge knife estimator)
=======
import argparse
import torch
import numpy as np
>>>>>>> 33a03ab (fix kinfe estimator)

from pathlib import Path
from cache_models import SMALL_MODELS, LARGE_MODELS

from sentence_transformers import SentenceTransformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
<<<<<<< HEAD
<<<<<<< HEAD
    parser.add_argument("--classification_task", action="store_true", default=False)
    parser.add_argument(
        "--split",
        type=str,
=======
    parser.add_argument(
        "--split",
        type=str,
        required=True,
>>>>>>> 0edeba6 (merge knife estimator)
=======
    parser.add_argument("--classification_task", action="store_true", default=False)
    parser.add_argument(
        "--split",
        type=str,
>>>>>>> 33a03ab (fix kinfe estimator)
        choices=["train", "validation", "test"],
        default="test",
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    return parser.parse_args()


def main():
    args = parse_args()
<<<<<<< HEAD
<<<<<<< HEAD

    if not args.classification_task:
        dataset = load_emb_dataset(
            args.dataset, AVAILABLE_DATASETS[args.dataset], args.split
        )

        model = SentenceTransformer(args.model)

        output_dir = Path(args.output_dir) / args.model / args.dataset / args.split
        output_dir.mkdir(parents=True, exist_ok=True)

        embeddings = model.encode(
            dataset["text"],
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=args.device,
        )

        with open(output_dir / "embeddings.npy", "wb") as f:
            torch.save(embeddings, f)

    else:
        dataset, splits, metadata = load_emd_classif_dataset(args.dataset)

        model = SentenceTransformer(args.model)

        for split in splits:
            _output_dir = Path(args.output_dir) / args.model / args.dataset / split
            _output_dir.mkdir(parents=True, exist_ok=True)
            embeddings = model.encode(
                dataset[split]["text"],
                batch_size=args.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                device=args.device,
            )
            with open(_output_dir / "embeddings.npy", "wb") as f:
                torch.save(embeddings, f)
            # save the labels
            with open(_output_dir / "labels.npy", "wb") as f:
                torch.save(dataset[split]["label"], f)
=======
    dataset = load_emb_dataset(
        args.dataset, AVAILABLE_DATASETS[args.dataset], args.split
    )
=======
>>>>>>> 33a03ab (fix kinfe estimator)

    if not args.classification_task:
        dataset = load_emb_dataset(
            args.dataset, AVAILABLE_DATASETS[args.dataset], args.split
        )

        model = SentenceTransformer(args.model)

        output_dir = Path(args.output_dir) / args.model / args.dataset / args.split
        output_dir.mkdir(parents=True, exist_ok=True)

<<<<<<< HEAD
    with open(output_dir / "embeddings.npy", "wb") as f:
        torch.save(embeddings, f)
>>>>>>> 0edeba6 (merge knife estimator)
=======
        embeddings = model.encode(
            dataset["text"],
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=args.device,
        )

        with open(output_dir / "embeddings.npy", "wb") as f:
            torch.save(embeddings, f)

    else:
        dataset, splits, metadata = load_emd_classif_dataset(args.dataset)

        model = SentenceTransformer(args.model)

        for split in splits:
            _output_dir = Path(args.output_dir) / args.model / args.dataset / split
            _output_dir.mkdir(parents=True, exist_ok=True)
            embeddings = model.encode(
                dataset[split]["text"],
                batch_size=args.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                device=args.device,
            )
            with open(_output_dir / "embeddings.npy", "wb") as f:
                torch.save(embeddings, f)
            # save the labels
            with open(_output_dir / "labels.npy", "wb") as f:
                torch.save(dataset[split]["label"], f)
>>>>>>> 33a03ab (fix kinfe estimator)


if __name__ == "__main__":
    main()
