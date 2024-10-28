import os
import json
from typing import Dict, List, Callable
import pandas as pd
import datamol as dm
import torch
import argparse

from molecule.utils.tdc_dataset import get_dataset_split

from molecule.utils import MolecularFeatureExtractor
from molecule.utils.estimator_utils.estimation_utils import get_embedders
import tqdm as tqdm

import wandb

import logging


DATASETS_GROUP = {
    "TOX": [
        "hERG",
        "hERG_Karim",
        "AMES",
        "DILI",
        "Carcinogens_Lagunin",
        "Skin__Reaction",
        "Tox21",
        "ClinTox",
    ]
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODELS = [
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
    "MolBert",
    "ChemBertMLM-5M",
    "ChemBertMLM-10M",
    "ChemBertMLM-77M",
    "ChemBertMTR-5M",
    "ChemBertMTR-10M",
    "ChemBertMTR-77M",
    "ChemGPT-1.2B",
    "ChemGPT-19M",
    "ChemGPT-4.7M",
    "DenoisingPretrainingPQCMv4",
    "FRAD_QM9",
    "MolR_gat",
    "MolR_gcn",
    "MolR_tag",
    "MoleOOD_OGB_GIN",
    "MoleOOD_OGB_GCN",
    "MoleOOD_OGB_SAGE",
    "ThreeDInfomax",
]


def preprocess_smiles(s):
    mol = dm.to_mol(s)
    return dm.to_smiles(mol, True, False)


class DatasetReader:
    def __init__(self, data_path: str, dataset: str):
        self.data_path = data_path
        self.dataset = dataset
        self.splits = get_dataset_split(dataset)

        self.smiles = None
        with open(os.path.join(data_path, "smiles.json"), "r") as f:
            self.smiles = json.load(f)

        self.mols = None
        mol_path = os.path.join(data_path, "preprocessed.sdf")
        if os.path.exists(mol_path):
            self.mols = dm.read_sdf(mol_path)
        else:
            self.mols = dm.to_mol(smiles)

        self.smiles_preprocessing_correspondancy = {}
        self.smiles_to_idx = {s: i for i, s in enumerate(self.smiles)}

    def preprocess_smiles(self, s):
        if s in self.smiles_preprocessing_correspondancy:
            return self.smiles_preprocessing_correspondancy[s]
        mol = dm.to_mol(s)
        new_s = dm.to_smiles(mol, True, False)
        self.smiles_preprocessing_correspondancy[s] = new_s
        return new_s

    def get_split_idx(self, split):
        for key in split.keys():
            split[key]["prepro_smiles"] = split[key]["Drug"].apply(
                self.preprocess_smiles
            )
        split_idx = {}
        for key in split.keys():
            split_idx[key] = {"x": [], "y": []}
            for smile, y in zip(split[key]["prepro_smiles"], split[key]["Y"]):
                if smile in self.smiles:
                    split_idx[key]["x"].append(self.smiles_to_idx[smile])
                    split_idx[key]["y"].append(y)
        return split_idx, self.smiles, self.mols


def get_split_emb(
    split_idx: Dict[str, List[int]],
    embedders: Dict[str, Callable],
    smiles: List[str],
    mols: List[dm.Mol],
    embedder_name: str = "ecfp",
):
    X = embedders[embedder_name](smiles, mols=mols)
    split_emb = {}
    for key in split_idx.keys():
        split_emb[key] = {
            "x": X[split_idx[key]["x"]].to("cpu"),
            "y": torch.tensor(split_idx[key]["y"]),
        }
    return split_emb


def launch_evaluation(
    dataset: str,
    length: int,
    embedder_name: str,
    device: str,
    split_idx: Dict[str, List[int]],
    smiles: List[str],
    mols: List[dm.Mol],
    embedders: Dict[str, Callable],
    target_id: str,
):
    split_emb = get_split_emb(split_idx, embedders, smiles, mols, embedder_name)

    X = split_emb["train"]["x"].to(device)
    y = split_emb["train"]["y"].to(device)
    y_class = y > y.median()

    results = {
        "embedder": [embedder_name],
        "target": [target_id],
        "dataset": [dataset],
        "length": [length],
    }

    dist = torch.cdist(X, X)
    for n_neighb in [1, 2, 4, 8]:
        clustering_scores = (
            (
                y_class[torch.argsort(dist, dim=1)[:, 1 : n_neighb + 1]]
                == y_class.unsqueeze(1)
            )
            .float()
            .mean()
        )
        results[f"clustering_{n_neighb}"] = [clustering_scores.item()]
        clustering_l2 = (
            (y[torch.argsort(dist, dim=1)[:, 1 : n_neighb + 1]] - y.unsqueeze(1))
            .abs()
            .mean()
        )
        results[f"clustering_l2_{n_neighb}"] = [clustering_l2.item()]

    df_results = pd.DataFrame(results)
    return df_results


def main(args):
    final_res = []

    for dataset in args.datasets:
        data_path = os.path.join(args.data_path, dataset)
        dataset_reader = DatasetReader(data_path, dataset)
        p_bar = tqdm.tqdm(total=len(args.embedders) * len(dataset_reader.splits))
        for i, embedder_name in enumerate(args.embedders):
            for split in dataset_reader.splits:
                split_idx, smiles, mols = dataset_reader.get_split_idx(split)
                # Get all enmbedders
                feature_extractor = MolecularFeatureExtractor(
                    dataset=dataset,
                    length=args.length,
                    device=args.device,
                    data_dir=args.data_path,
                )
                embedders = get_embedders(MODELS, feature_extractor)

                final_res.append(
                    launch_evaluation(
                        dataset=dataset,
                        length=args.length,
                        embedder_name=embedder_name,
                        split_idx=split_idx,
                        device=args.device,
                        smiles=smiles,
                        mols=mols,
                        embedders=embedders,
                        target_id=split["train"]["Target_ID"].unique()[0],
                    )
                )
                p_bar.update(1)

    df = pd.concat(final_res).reset_index(drop=True)
    df.to_csv(f"results/{dataset}_DTI.csv")
    df = wandb.Table(dataframe=df)
    wandb.log({"results_df": df})


if __name__ == "__main__":
    from molecule.utils.parser_mol import add_downstream_args

    parser = argparse.ArgumentParser(
        description="Launch the evaluation of a downstream model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser = add_downstream_args(parser)

    args = parser.parse_args()
    wandb.init(project="Emir-downstream")

    if args.embedders is None:
        args.embedders = MODELS

    wandb.config.update(args)
    main(args)
