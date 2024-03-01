"""Script transforming binary fingerprints into continuous ones using the MDS algorithm. (Can be long on large datasets)"""

import argparse
import json
import os

import numpy as np
from utils import MolecularFeatureExtractor

from sklearn.manifold import MDS

from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(
        description="Script transforming binary fingerprints into continuous ones using the MDS algorithm. (Can be long on large datasets)"
    )

    parser.add_argument(
        "--descriptors",
        type=str,
        nargs="+",
        default=[
            # "default",
            "cats",
            "gobbi",
            "pmapper",
            "cats/3D",
            "gobbi/3D",
            "pmapper/3D",
        ],
        help="List of descriptors to compare",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="ClinTox",
        help="Dataset to use",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="data",
        help="Output file",
    )

    parser.add_argument("--fp-length", type=int, default=1024)
    parser.add_argument("--out-dim", type=int, default=64)

    return parser


def main(args):
    with open(f"data/{args.dataset}/smiles.json", "r") as f:
        smiles = json.load(f)

    feature_extractor = MolecularFeatureExtractor(
        device="cpu",
        length=args.fp_length,
        dataset=args.dataset,
        mds_dim=args.out_dim,
    )
    for desc in tqdm(args.descriptors):
        print(f"Processing {desc}...")
        print(f"Loading {args.dataset}...")
        descriptors_embedding = feature_extractor.get_features(
            smiles=smiles,
            name=desc,
            feature_type="descriptor",
        )
        if (
            len(descriptors_embedding.unique()) < 1500
            and not (descriptors_embedding == 0)
            .logical_or(descriptors_embedding == 1)
            .all()
        ):
            # setting to 1 all elements different from 0
            descriptors_embedding = (descriptors_embedding != 0).float()

        if (descriptors_embedding == 0).logical_or(descriptors_embedding == 1).all():
            print(f"Computing Tanimoto similarity...")
            descriptors_embedding = descriptors_embedding.to_sparse()
            intersection = descriptors_embedding @ descriptors_embedding.T
            descriptors_embedding = descriptors_embedding.to_dense()
            tanimoto = 1 - (intersection.to_dense() + 1e-8) / (
                (
                    descriptors_embedding.sum(1)[:, None]
                    + descriptors_embedding.sum(1)[None, :]
                )
                - intersection
                + 1e-8
            )

            print(f"Computing continuous fingerprints...")
            mds = MDS(
                n_components=args.out_dim,
                dissimilarity="precomputed",
                n_init=1,
                n_jobs=1,
                verbose=3,
            )
            continuous_fingerprints = mds.fit_transform(tanimoto)

            print(f"Saving continuous fingerprints...")
            os.makedirs(f"{args.out_dir}/{args.dataset}", exist_ok=True)
            np.save(
                f"""{args.out_dir}/{args.dataset}/{desc.replace("/", "_")}_{args.fp_length}_mds_{args.out_dim}.npy""",
                continuous_fingerprints,
            )
        print("Done.")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
