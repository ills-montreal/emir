import argparse
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from emb_datasets import (
    load_emb_dataset,
    AVAILABLE_DATASETS,
    load_emd_classif_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--classification_task", action="store_true", default=False)
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "validation", "test"],
        default="test",
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)

    return parser.parse_args()


def main():
    args = parse_args()

    print("START")
    if not args.classification_task:
        print("CLASSIFICATION TASK")

        print("LOADING EMBEDDINGS")
        dataset = load_emb_dataset(
            args.dataset, AVAILABLE_DATASETS[args.dataset], args.split
        )
        print("EMBEDDINGS LOADED")

        print("LOADING MODEL")
        model = SentenceTransformer(
            args.model,
        )

        model.tokenizer.pad_token = model.tokenizer.eos_token
        print("MODEL LOADED")

        output_dir = Path(args.output_dir) / args.model / args.dataset / args.split
        output_dir.mkdir(parents=True, exist_ok=True)

        print("GENERATING EMBEDDINGS")
        embeddings = model.encode(
            dataset["text"],
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        print("EMBEDDINGS GENERATED")

        with open(output_dir / "embeddings.npy", "wb") as f:
            torch.save(embeddings, f)

    else:
        print("CLASSIFICATION TASK")

        print("LOADING DATASET")
        dataset, splits, metadata = load_emd_classif_dataset(args.dataset)
        print("DATASET LOADED")

        print("LOADING MODEL")
        model = SentenceTransformer(args.model)
        print("MODEL LOADED")

        model.tokenizer.pad_token = model.tokenizer.eos_token

        for split in splits:
            _output_dir = Path(args.output_dir) / args.model / args.dataset / split
            _output_dir.mkdir(parents=True, exist_ok=True)
            embeddings = model.encode(
                dataset[split]["text"],
                batch_size=args.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            with open(_output_dir / "embeddings.npy", "wb") as f:
                torch.save(embeddings, f)
            # save the labels
            with open(_output_dir / "labels.npy", "wb") as f:
                torch.save(dataset[split]["label"], f)


if __name__ == "__main__":
    import sys

    print(sys.version)
    main()
