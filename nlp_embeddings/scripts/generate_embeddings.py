from emb_datasets import load_emb_dataset, AVAILABLE_DATASETS
import argparse
import torch

from pathlib import Path
from cache_models import SMALL_MODELS, LARGE_MODELS

from sentence_transformers import SentenceTransformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--split",
        type=str,
        required=True,
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


if __name__ == "__main__":
    main()
