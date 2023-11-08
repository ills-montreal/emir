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


def precompute_3d(smiles: List[str]):
    mols = [
        dm.conformers.generate(dm.to_mol(s), align_conformers=True, n_confs=5, ignore_failure=True)
        for s in tqdm(smiles, desc="Generating conformers")
    ]
    # Removing molecules that cannot be featurized
    transformer, thrD = get_molfeat_transformer("usr")
    feat, valid_id = transformer(mols, ignore_errors=True)
    smiles = np.array(smiles)[valid_id]
    mols = np.array(mols)[valid_id]
    return mols, smiles


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
        embeddings_fn[model_name] = partial(get_embeddings_from_model, path=model_path)
    for method in args.descriptors:
        embeddings_fn[method] = partial(
            get_molfeat_descriptors, transformer_name=method, length=args.fp_length
        )
    return embeddings_fn


def model_profile(
    embeddings_fn: Dict[str, callable],
    model_name: str,
    descriptors: List[str],
    dataloader: DataLoader,
    smiles: List[str],
    mols: List[dm.Mol] = None,
    p_bar: tqdm = None,
    knife_config: KNIFEArgs = None,
    n_runs: int = 1,
):
    results = {
        "desc1": [],
        "desc2": [],
        "mi": [],
    }
    for desc in descriptors:
        mis = []
        for _ in range(n_runs):
            mi, _, _, loss = get_knife_preds(
                embeddings_fn[desc],
                embeddings_fn[model_name],
                dataloader,
                smiles,
                mols,
                knife_config=knife_config,
            )
            results["desc1"].append(model_name)
            results["desc2"].append(desc)
            results["mi"].append(mi)
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
    try:
        df = Tox(name=args.dataset).get_data()
    except ValueError as e:
        label_list = retrieve_label_name_list(args.dataset)
        df = Tox(name=args.dataset, label_name=label_list[0]).get_data()

    smiles = df["Drug"].tolist()
    mols = None
    if args.precompute_3d:
        mols, smiles = precompute_3d(smiles)

    knife_config = generate_knife_config_from_args(args)

    logger.info(f"Dataset {args.dataset} loaded, with {len(smiles)} datapoints")

    dataloader = DataLoader(
        [mol_to_graph_data_obj_simple(dm.to_mol(s)) for s in smiles],
        batch_size=32,
        shuffle=False,
    )
    embeddings_fn = get_embedders(args)
    results = []
    p_bar = tqdm(
        total=len(args.models) * len(args.descriptors), desc="Models", position=0
    )

    for model_name in args.models:
        results.append(
            model_profile(
                embeddings_fn,
                model_name,
                args.descriptors,
                dataloader,
                smiles,
                mols=mols,
                p_bar=p_bar,
                knife_config=knife_config,
                n_runs=args.n_runs,
            )
        )
    results = pd.concat(results)
    results.to_csv(args.out_file, index=False)
    return results


if __name__ == "__main__":
    wandb.init(project="Emir")
    df = main()
    max_desc = df.groupby("desc2").mi.max()
    df["mi_normed"] = df.apply(lambda x: x.mi / max_desc[x.desc2], axis=1)

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
