import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging

import argparse

import json
import datamol as dm
import wandb
import yaml

from molecule.utils.estimator_utils.knife_utils import compute_all_mi
from molecule.utils.parser_mol import (
    add_eval_cli_args,
    add_knife_args,
)


GROUPED_MODELS = {
    "GNN": [
        "ContextPred",
        "GPT-GNN",
        "GraphMVP",
        "GROVER",
        # "EdgePred", # This model is especially bad and makes visualization hard
        "AttributeMask",
        "GraphLog",
        "GraphCL",
        "InfoGraph",
        "Not-trained",
    ],
    "BERT": [
        "MolBert",
        "ChemBertMLM-5M",
        "ChemBertMLM-10M",
        "ChemBertMLM-77M",
        "ChemBertMTR-5M",
        "ChemBertMTR-10M",
        "ChemBertMTR-77M",
    ],
    "GPT": [
        "ChemGPT-1.2B",
        "ChemGPT-19M",
        "ChemGPT-4.7M",
    ],
    "Denoising": [
        "DenoisingPretrainingPQCMv4",
        "FRAD_QM9",
    ],
    "MolR": [
        "MolR_gat",
        "MolR_gcn",
        "MolR_tag",
    ],
    "MoleOOD": [
        "MoleOOD_OGB_GIN",
        "MoleOOD_OGB_GCN",
        "MoleOOD_OGB_SAGE",
    ],
    "ThreeD": [
        "ThreeDInfomax",
    ],
    "Descriptors": [
        "usrcat",
        "electroshape",
        "usr",
        "ecfp",
        "estate",
        "erg",
        "rdkit",
        "topological",
        "avalon",
        "maccs",
        "atompair-count",
        "topological-count",
        "fcfp-count",
        "secfp",
        "pattern",
        "fcfp",
        "scaffoldkeys",
        "cats",
        "default",
        "gobbi",
        "pmapper",
        "cats/3D",
        "gobbi/3D",
        "pmapper/3D",
    ],
}


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
    if args.wandb:
        wandb.init(project="Emir")

    new_X = []
    for embedder in args.X:
        if not embedder in GROUPED_MODELS:
            new_X.append(embedder)
        else:
            new_X += GROUPED_MODELS[embedder]
    args.X = new_X

    new_Y = []
    for embedder in args.Y:
        if not embedder in GROUPED_MODELS:
            new_Y.append(embedder)
        else:
            new_Y += GROUPED_MODELS[embedder]
    args.Y = new_Y

    dir_key = "tmp" if args.name is None else args.name

    args.out_dir = os.path.join(
        args.out_dir,
        args.dataset,
        str(args.fp_length),
        f"{str(args.use_VAE_embs)}_{args.vae_latent_dim}",
        dir_key,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, "args.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    os.makedirs(os.path.join(args.out_dir, "losses"), exist_ok=True)
    logger.info(f"Saving results in {args.out_dir}")

    # Update the wandb config with the args specified in the command line
    if args.wandb:
        wandb.config.update(args)
    smiles_path = f"{args.data_path}/{args.dataset}/smiles.json"
    mol_path = f"{args.data_path}/{args.dataset}/preprocessed.sdf"

    assert os.path.exists(
        smiles_path
    ), f"Please run precompute_molf_descriptors.py first. Missing path : {smiles_path}"

    with open(smiles_path, "r") as f:
        smiles = json.load(f)
    if os.path.exists(mol_path):
        mols = dm.read_sdf(mol_path)
    else:
        mols = None

    all_results = compute_all_mi(
        args=args,
        smiles=smiles,
        mols=mols,
    )
    return all_results, args


if __name__ == "__main__":
    df, args = main()
    max_desc = df.groupby("Y")["I(X->Y)"].max()
    df["mi_normed"] = df.apply(lambda x: x["I(X->Y)"] / max_desc[x.Y], axis=1)
    if args.wandb:
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
