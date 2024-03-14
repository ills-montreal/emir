import os
import sys

sys.path.append(os.path.dirname(__file__))

import logging

import argparse

import json
import datamol as dm
import pandas as pd
import wandb


from molecule.utils.knife_utils import compute_all_mi
from molecule.parser_mol import (
    add_eval_cli_args,
    add_knife_args,
)



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

handler = logging.StreamHandler()


logger.info("Logger is set.")


def main():
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser = add_eval_cli_args(parser)
    parser = add_knife_args(parser)
    args = parser.parse_args()

    args.out_dir = os.path.join(
        args.out_dir, args.dataset, str(args.fp_length), str(args.mds_dim)
    )
    os.makedirs(args.out_dir, exist_ok=True)

    os.makedirs(os.path.join(args.out_dir, "losses"), exist_ok=True)
    logger.info(f"Saving results in {args.out_dir}")

    # Update the wandb config with the args specified in the command line
    wandb.config.update(args)

    assert os.path.exists(
        f"data/{args.dataset}/smiles.json"
    ), "Please run precompute_molf_descriptors.py first."

    with open(f"data/{args.dataset}/smiles.json", "r") as f:
        smiles = json.load(f)
    if os.path.exists(f"data/{args.dataset}/preprocessed.sdf"):
        mols = dm.read_sdf(f"data/{args.dataset}/preprocessed.sdf")
    else:
        mols = None


    all_results = compute_all_mi(
        args=args,
        smiles=smiles,
        mols=mols,
    )
    return all_results


if __name__ == "__main__":
    wandb.init(project="Emir")
    df = main()
    max_desc = df.groupby("Y")["I(X->Y)"].max()
    df["mi_normed"] = df.apply(lambda x: x["I(X->Y)"] / max_desc[x.Y], axis=1)

    wandb.log(
        {
            "inter_model_std": df.groupby("Y")["I(X->Y)"].std().mean(),
            "intra_model_std": df.groupby("X")["I(X->Y)"].std().mean(),
            "inter_model_std_normalized": df.groupby("Y").mi_normed.std().mean(),
            "intra_model_std_normalized": df.groupby("X").mi_normed.std().mean(),
        }
    )

    wandb.log({"results": wandb.Table(dataframe=df)})
    wandb.finish()
