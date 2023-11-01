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

from tqdm import tqdm

from torch_geometric.data import DataLoader

from moleculenet_encoding import mol_to_graph_data_obj_simple
from utils import (
    get_embeddings_from_model,
    get_molfeat_descriptors,
    get_molfeat_transformer,
)

from emir.estimators import KNIFEEstimator, KNIFEArgs

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


Knige_config = KNIFEArgs(
    cond_modes=3,
    marg_modes=3,
    lr=0.01,
    batch_size=128,
    device="cpu",
    n_epochs=30,
    ff_layers=2,
    cov_diagonal="var",
    cov_off_diagonal="",
)


def add_eval_cli_args(parser: argparse.ArgumentParser):
    """
    Parser for the eval command line interface.
    Will collect :
        - The list of models to compare
        - The list of descriptors to compare
        - The dataset to use
    :param parser: argparse.ArgumentParser
    :return: argparse.ArgumentParser
    """
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["GraphMVP", "GROVER"],
        help="List of models to compare",
    )

    parser.add_argument(
        "--descriptors",
        type=str,
        nargs="+",
        default=["ecfp", "erg", "topological", "scaffoldkeys", "cats", "default"],
        help="List of descriptors to compare",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="hERG_Karim",
        help="Dataset to use",
    )

    parser.add_argument(
        "--out_file",
        type=str,
        default="results.csv",
        help="Output file",
    )

    return parser


def main():
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser = add_eval_cli_args(parser)
    args = parser.parse_args()
    try:
        df = Tox(name=args.dataset).get_data()
    except ValueError as e:
        label_list = retrieve_label_name_list('Tox21')
        df = Tox(name=args.dataset, label_name=label_list[0]).get_data()

    logger.info(f"Dataset {args.dataset} loaded, with {len(df)} datapoints")
    smiles = df["Drug"].tolist()

    dataloader = DataLoader(
        [mol_to_graph_data_obj_simple(dm.to_mol(s)) for s in smiles],
        batch_size=32,
        shuffle=False,
    )
    embeddings_fn = get_embedders(args)
    results = []
    p_bar = tqdm(total=len(args.models) * len(args.descriptors), desc="Models", position=0)

    for model_name in args.models:
        results.append(
            model_profile(
                embeddings_fn, model_name, args.descriptors, dataloader, smiles, p_bar = p_bar
            )
        )
    results = pd.concat(results)
    results.to_csv(args.out_file, index=False)


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

    embeddings_fn = {}
    for model_name, model_path in MODELS.items():
        embeddings_fn[model_name] = partial(get_embeddings_from_model, path=model_path)
    for method in args.descriptors:
        embeddings_fn[method] = partial(
            get_molfeat_descriptors, transformer_name=method
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
):
    results = {
        "desc1": [],
        "desc2": [],
        "mi": [],
    }
    for desc in descriptors:
        mi, _, _, _ = get_knife_preds(
            embeddings_fn[model_name], embeddings_fn[desc], dataloader, smiles, mols
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
) -> Tuple[float, float, float, List[float]]:
    x1 = emb_fn1(dataloader, smiles, mols=mols)
    x2 = emb_fn2(dataloader, smiles, mols=mols)
    knife_estimator = KNIFEEstimator(Knige_config, x1.shape[1], x2.shape[1])
    mi, m, c = knife_estimator.eval(x1.float(), x2.float(), record_loss=True)
    return mi, m, c, knife_estimator.recorded_loss


if __name__ == "__main__":
    main()
