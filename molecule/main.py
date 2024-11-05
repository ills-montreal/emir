import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import logging

import argparse

import json
import datamol as dm
import wandb
import yaml

from molecule.utils.estimator_utils.estimation_utils import MI_EstimatorRunner
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


def update_args(args):
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
    return args


def main():
    parser = argparse.ArgumentParser(
        description="Compute Pairwise reconstruction statistics between embedders",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser = add_eval_cli_args(parser)
    parser.add_argument("--estimator-config", type=str, default="knife.yaml")
    parser.add_argument("--estimator", type=str, default="KNIFE")
    args = parser.parse_args()
    if args.wandb:
        wandb.init(project="Emir")
    args = update_args(args)

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

    estimator_runner = MI_EstimatorRunner(args=args, smiles=smiles, mols=mols)
    all_results = estimator_runner()
    return estimator_runner.metrics, estimator_runner.loss, args


def save_loss(loss, out_dir):
    for model_name in loss.X.unique():
        model_loss = loss[loss.X == model_name]
        model_loss.to_csv(
            os.path.join(out_dir, f"losses/{model_name}.csv"), index=False
        )


def save_metrics(metrics, out_dir):
    for model_name in metrics.X.unique():
        model_metrics = metrics[metrics.X == model_name]
        model_metrics.to_csv(os.path.join(out_dir, f"{model_name}.csv"), index=False)


if __name__ == "__main__":
    metrics, loss, args = main()

    save_loss(loss, args.out_dir)
    save_metrics(metrics, args.out_dir)

    max_desc = metrics.groupby("Y")["I(Y->X)"].max()
    metrics["mi_normed"] = metrics.apply(lambda x: x["I(Y->X)"] / max_desc[x.X], axis=1)
    if args.wandb:
        wandb.log({"results": wandb.Table(dataframe=metrics)})
        wandb.finish()
