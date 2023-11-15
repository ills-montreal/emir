import os
import logging

from functools import partial
import argparse
from typing import Tuple, List, Dict

from tdc.utils import retrieve_label_name_list
from tdc.single_pred import Tox

import datamol as dm
import torch
import numpy as np
import pandas as pd
import wandb

from tqdm import tqdm

from torch_geometric.data import DataLoader

from moleculenet_encoding import mol_to_graph_data_obj_simple
from utils import (
    get_embeddings_from_model_moleculenet,
    get_embeddings_from_transformers,
    get_molfeat_descriptors,
    get_molfeat_transformer,
)
from parser_mol import (
    add_eval_cli_args,
    add_knife_args,
    generate_knife_config_from_args,
)
from precompute_3d import precompute_3d
from tdc_dataset import get_dataset
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
            get_embeddings_from_model_moleculenet, path=model_path
        )
    for method in args.descriptors:
        embeddings_fn[method] = partial(
            get_molfeat_descriptors, transformer_name=method, length=args.fp_length
        )
    for model_name in args.models:
        if not model_name in embeddings_fn:
            embeddings_fn[model_name] = partial(
                get_embeddings_from_transformers, transformer_name=model_name
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
    }
    for desc in args.descriptors:
        mis = []
        for _ in range(args.n_runs):
            mi, m, c, loss = get_knife_preds(
                embeddings_fn[desc],
                embeddings_fn[model_name],
                dataloader,
                smiles,
                mols,
                knife_config=knife_config,
            )
            results["Y"].append(model_name)
            results["X"].append(desc)
            results["I(Y)"].append(m)
            results["I(Y|X)"].append(c)
            results["I(X->Y)"].append(mi)

            if args.compute_both_mi:
                mi, m, c, loss = get_knife_preds(
                    embeddings_fn[model_name],
                    embeddings_fn[desc],
                    dataloader,
                    smiles,
                    mols,
                    knife_config=knife_config,
                )
                results["I(X)"].append(m)
                results["I(X|Y)"].append(c)
                results["I(Y->X)"].append(mi)
            else:
                results["I(Y)"].append(np.nan)
                results["I(Y|X)"].append(np.nan)
                results["I(X->Y)"].append(np.nan)

        if p_bar is not None:
            p_bar.update(1)
    return pd.DataFrame(results)


def get_knife_preds(
    emb_fn1: callable,
    emb_fn2: callable,
    dataloader: DataLoader,
    smiles: List[str],
    mols: List[dm.Mol] = None,
    knife_config: KNIFEArgs = None,
) -> Tuple[float, float, float, List[float]]:
    x1 = emb_fn1(dataloader, smiles, mols=mols)
    x2 = emb_fn2(dataloader, smiles, mols=mols)
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

    df = get_dataset(args.dataset)

    smiles = df["Drug"].tolist()
    mols = None
    if args.precompute_3d:
        mols, smiles = precompute_3d(smiles, args.dataset)
        # Removing molecules that cannot be featurized
        for t_name in ["usr", "electroshape", "usrcat"]:
            transformer, thrD = get_molfeat_transformer(t_name)
            feat, valid_id = transformer(mols, ignore_errors=True)
            smiles = np.array(smiles)[valid_id]
            mols = np.array(mols)[valid_id]

    logger.info(f"Dataset {args.dataset} loaded, with {len(smiles)} datapoints")
    graph_input = []
    valid_smiles = []
    valid_mols = []
    for i, s in enumerate(tqdm(smiles, desc="Generating graphs")):
        try:
            graph_input.append(mol_to_graph_data_obj_simple(dm.to_mol(s)))
            valid_smiles.append(s)
            valid_mols.append(mols[i])
        except:
            pass
    smiles = valid_smiles
    mols = valid_mols

    dataloader = DataLoader(
        graph_input,
        batch_size=32,
        shuffle=False,
    )

    p_bar = tqdm(
        total=len(args.models) * len(args.descriptors), desc="Models", position=0
    )
    results = []
    for model_name in args.models:
        results.append(
            model_profile(
                args,
                model_name,
                dataloader,
                smiles,
                mols=mols,
                p_bar=p_bar,
            )
        )
    results = pd.concat(results)
    results.to_csv(args.out_file + args.dataset + ".csv", index=False)
    return results


if __name__ == "__main__":
    wandb.init(project="Emir")
    df = main()
    max_desc = df.groupby("Y")["I(X->Y)"].max()
    df["mi_normed"] = df.apply(lambda x: x["I(X->Y)"] / max_desc[x.Y], axis=1)

    wandb.log(
        {
            "inter_model_std": df.groupby("desc2").mi.std().mean(),
            "intra_model_std": df.groupby("desc1").mi.std().mean(),
            "inter_model_std_normalized": df.groupby("desc2").mi_normed.std().mean(),
            "intra_model_std_normalized": df.groupby("desc1").mi_normed.std().mean(),
        }
    )

    wandb.log({"results": wandb.Table(dataframe=df)})
    wandb.finish()
