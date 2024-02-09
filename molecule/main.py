import os
import logging

from functools import partial
import argparse
from typing import Tuple, List, Dict

import json
import datamol as dm
import torch
import numpy as np
import pandas as pd
import wandb

from tqdm import tqdm

from torch_geometric.data import DataLoader

from moleculenet_encoding import mol_to_graph_data_obj_simple
from utils import get_features
from parser_mol import (
    add_eval_cli_args,
    add_knife_args,
    generate_knife_config_from_args,
)
from emir.estimators import KNIFEEstimator, KNIFEArgs


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

handler = logging.StreamHandler()


logger.info("Logger is set.")


def get_embedders(args: argparse.Namespace):
    MODEL_PATH = "backbone_pretrained_models"
    MODELS = {}
    # For every directory in the folder
    for model_name in os.listdir(MODEL_PATH):
        if model_name in args.models:
            # For every file in the directory
            for file_name in os.listdir(os.path.join(MODEL_PATH, model_name)):
                # If the file is a .pth file
                if file_name.endswith(".pth"):
                    MODELS[model_name] = os.path.join(MODEL_PATH, model_name, file_name)
    MODELS["Not-trained"] = ""
    embeddings_fn = {}
    for model_name, model_path in MODELS.items():
        embeddings_fn[model_name] = partial(
            get_features, path=model_path, feature_type="model", name=model_name
        )

    for method in args.descriptors:
        embeddings_fn[method] = partial(
            get_features,
            name=method,
            length=args.fp_length,
            feature_type="descriptor",
        )
    for model_name in args.models:
        if not model_name in embeddings_fn:
            embeddings_fn[model_name] = partial(
                get_features, name=model_name, feature_type="model"
            )

    return embeddings_fn

def model_profile(
    args: argparse.Namespace,
    model_name: str,
    dataloader: DataLoader,
    smiles: List[str],
    mols: List[dm.Mol] = None,
    p_bar: tqdm = None,
):
    knife_config = generate_knife_config_from_args(args)
    embeddings_fn = get_embedders(args)
    results = {
        "X": [],
        "Y": [],
        "I(Y)": [],
        "I(Y|X)": [],
        "I(X->Y)": [],
        "I(X)": [],  # Not possible yet
        "I(X|Y)": [],  # Not possible yet
        "I(Y->X)": [],  # Not possible yet
        "Y_dim": [],
        "X_dim": [],
        "is_desc_discrete": [],
    }

    model_embedding = embeddings_fn[model_name](
        dataloader, smiles, mols=mols, dataset=args.dataset
    ).to(knife_config.device)
    df_losses_XY = []
    df_losses_YX = []

    for desc in args.descriptors:
        mis = []
        descriptors_embedding = embeddings_fn[desc](
            dataloader, smiles, mols=mols, dataset=args.dataset
        ).to(knife_config.device)

        for i in range(args.n_runs):
            mi, m, c, loss = get_knife_preds(
                descriptors_embedding, model_embedding, knife_config=knife_config
            )
            results["Y"].append(model_name)
            results["X"].append(desc + str(args.fp_length))
            results["I(Y)"].append(m)
            results["I(Y|X)"].append(c)
            results["I(X->Y)"].append(mi)
            results["Y_dim"].append(model_embedding.shape[1])
            results["X_dim"].append(descriptors_embedding.shape[1])
            results["is_desc_discrete"].append(
                (descriptors_embedding == 0).logical_or(descriptors_embedding == 1).all().item()
            )

            # saving the loss evolution in losses as csvs
            df_losses_XY.append(
                pd.DataFrame(
                    {
                        "loss": loss,
                        "epoch": range(len(loss)),
                        "run": i,
                        "X": desc,
                        "Y": model_name,
                        "direction": "X->Y",
                    }
                )
            )
            if args.compute_both_mi:
                mi, m, c, loss = get_knife_preds(
                    model_embedding,
                    descriptors_embedding,
                    knife_config=knife_config,
                )
                results["I(X)"].append(m)
                results["I(X|Y)"].append(c)
                results["I(Y->X)"].append(mi)
                df_losses_YX.append(
                    pd.DataFrame(
                        {
                            "loss": loss,
                            "epoch": range(len(loss)),
                            "run": i,
                            "X": desc,
                            "Y": model_name,
                            "direction": "Y->X",
                        }
                    )
                )

            else:
                results["I(X)"].append(np.nan)
                results["I(X|Y)"].append(np.nan)
                results["I(Y->X)"].append(np.nan)

            if p_bar is not None:
                p_bar.update(1)

    if df_losses_XY != []:
        pd.concat(df_losses_XY).to_csv(
            os.path.join(
                os.path.join(args.out_dir, "losses"),
                f"{args.dataset}_{model_name}_{args.fp_length}_XY.csv",
            ),
            index=False,
        )
    if df_losses_YX != []:
        pd.concat(df_losses_YX).to_csv(
            os.path.join(
                os.path.join(args.out_dir, "losses"),
                f"{args.dataset}_{model_name}_{args.fp_length}_YX.csv",
            ),
            index=False,
        )
    return pd.DataFrame(results)

def get_knife_preds(
    x1: callable,
    x2: callable,
    knife_config: KNIFEArgs = None,
) -> Tuple[float, float, float, List[float]]:
    knife_estimator = KNIFEEstimator(
        knife_config, x1.shape[1], x2.shape[1]
    )  # Learn x2 from x1
    mi, m, c = knife_estimator.eval(x1.float(), x2.float(), record_loss=True)
    return mi, m, c, knife_estimator.recorded_loss


def main():
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser = add_eval_cli_args(parser)
    parser = add_knife_args(parser)
    args = parser.parse_args()

    assert os.path.exists(
        f"data/{args.dataset}/smiles.json"
    ), "Please run precompute_molf_descriptors.py first."

    with open(f"data/{args.dataset}/smiles.json", "r") as f:
        smiles = json.load(f)
    if os.path.exists(f"data/{args.dataset}/preprocessed.sdf"):
        mols = dm.read_sdf(f"data/{args.dataset}/preprocessed.sdf")
    else:
        mols = None

    graph_input = []
    for s in smiles:
        graph_input.append(mol_to_graph_data_obj_simple(dm.to_mol(s)))

    dataloader = DataLoader(
        graph_input,
        batch_size=32,
        shuffle=False,
    )

    p_bar = tqdm(
        total=len(args.models) * len(args.descriptors) * args.n_runs,
        desc="Progression",
        position=0,
    )
    all_results = []
    for model_name in args.models:
        results = model_profile(
            args,
            model_name,
            dataloader,
            smiles,
            mols=mols,
            p_bar=p_bar,
        )

        results.to_csv(
            os.path.join(
                args.out_dir, f"{args.dataset}_{model_name}_{args.fp_length}.csv"
            ),
            index=False,
        )
        all_results.append(results)
    return pd.concat(all_results)


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
