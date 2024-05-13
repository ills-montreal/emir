import pandas as pd
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Pack classification results")
    parser.add_argument(
        "--input_dir", type=Path, help="Input file with classification results"
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Output file with packed classification results",
    )
    parser.add_argument(
        "--avg", action="store_true", help="Average the results", default=False
    )

    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_path = args.output_path

    # get all all idea from files in input_dir "metadata_[id].csv"

    metadata_files = [
        file for file in input_dir.iterdir() if file.name.startswith("metadata_")
    ]
    ids = [file.stem.split("_")[1] for file in metadata_files]

    # read pair of classification_results_[id].csv and metadata_[id].csv

    results = []
    for idx in ids:
        metadata = pd.read_csv(input_dir / f"metadata_{idx}.csv")
        classification_results = pd.read_csv(
            input_dir / f"classification_results_{idx}.csv"
        )

        if args.avg:
            classification_results = (
                classification_results["success"].agg(["mean"]).to_frame()
            )

        if "logits" in classification_results.columns:
            classification_results = classification_results.drop("logits", axis=1)

        metadata["id"] = idx
        classification_results["id"] = idx

        # merge metadata and classification_results on id
        merged = pd.merge(metadata, classification_results, on="id")
        results.append(merged)

    # concatenate all results
    results = pd.concat(results)
    results.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
