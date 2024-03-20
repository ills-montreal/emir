import os
import json
from typing import Dict, List, Callable, Optional, Tuple
import pandas as pd
import datamol as dm
import torch
import argparse
import yaml

from molecule.utils.evaluation import get_dataloaders, Feed_forward, FFConfig
from molecule.tdc_dataset import get_dataset_split

from molecule.utils import MolecularFeatureExtractor
from molecule.utils.knife_utils import get_embedders
from tqdm import tqdm

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
    ],
    "ADME": [
        "PAMPA_NCATS",
        "HIA_Hou",
        "Pgp_Broccatelli",
        "Bioavailability_Ma",
        "BBB_Martins",
        "CYP2C19_Veith",
        "CYP2D6_Veith",
        "CYP3A4_Veith",
        "CYP1A2_Veith",
        "CYP2C9_Veith",
        "CYP2C9_Substrate_CarbonMangels",
        "CYP2D6_Substrate_CarbonMangels",
        "CYP3A4_Substrate_CarbonMangels",
    ],
    "ADME_REG": [
        "Caco2_Wang",
        "Lipophilicity_AstraZeneca",
        "Solubility_AqSolDB",
        "HydrationFreeEnergy_FreeSolv",
        "PPBR_AZ",
        "VDss_Lombardo",
        "Half_Life_Obach",
        "Clearance_Hepatocyte_AZ",
        "Clearance_Microsome_AZ",
    ],
    "TOX_REG": [
        "LD50_Zhu",
    ],
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DESCRIPTORS = [
    "ecfp",
    "estate",
    "fcfp",
    "erg",
    "rdkit",
    "topological",
    "avalon",
    "maccs",
    "secfp",
    "scaffoldkeys",
    "cats",
    "gobbi",
    "pmapper",
    "cats/3D",
    "gobbi/3D",
    "pmapper/3D",
    # "ScatteringWavelet",
]
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


def get_split_idx(
    dataset: str,
    split: Dict[str, pd.DataFrame],
):
    with open(f"data/{dataset}/smiles.json", "r") as f:
        smiles = json.load(f)
    if os.path.exists(f"data/{dataset}/preprocessed.sdf"):
        mols = dm.read_sdf(f"data/{dataset}/preprocessed.sdf")
    else:
        mols = dm.to_mol(smiles)

    smiles_to_idx = {s: i for i, s in enumerate(smiles)}

    for key in split.keys():
        split[key]["prepro_smiles"] = split[key]["Drug"].apply(preprocess_smiles)

    split_idx = {}
    for key in split.keys():
        split_idx[key] = {"x": [], "y": []}
        for i in range(len(split[key])):
            smile = split[key].iloc[i]["prepro_smiles"]
            y = split[key].iloc[i]["Y"]
            if smile in smiles:
                split_idx[key]["x"].append(smiles_to_idx[smile])
                split_idx[key]["y"].append(y)

    return split_idx, smiles, mols


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
    model_config: Dict,
    smiles: List[str],
    mols: List[dm.Mol],
    embedders: Dict[str, Callable],
    plot_loss: bool = False,
    test_run: bool = False,
    run_num: Optional[Tuple[int, int]] = None,
):
    split_emb = get_split_emb(split_idx, embedders, smiles, mols, embedder_name)

    dataloader_train, dataloader_val, dataloader_test, input_dim = get_dataloaders(
        split_emb,
        batch_size=model_config.batch_size,
        test_batch_size=model_config.test_batch_size,
    )
    y_train = split_emb["train"]["y"]
    if len(y_train.shape) == 1:
        output_dim = 1
    else:
        output_dim = split_emb["train"]["y"].shape[1]
    if dataset in DATASETS_GROUP["ADME_REG"] + DATASETS_GROUP["TOX_REG"]:
        model = Feed_forward(
            input_dim,
            output_dim,
            model_config,
            task="regression",
            task_size=y_train.shape[0],
            out_train_mean=y_train.mean(),
            out_train_std=y_train.std(),
        )
    else:
        model = Feed_forward(
            input_dim,
            output_dim,
            model_config,
            task="classification",
            task_size=y_train.shape[0],
        )

    if test_run:
        model.train_model(
            dataloader_train,
            dataloader_test,
            p_bar_name=f"{run_num[0]} / {run_num[1]} : \t{dataset} - {embedder_name} : ",
        )
    else:
        model.train_model(
            dataloader_train,
            dataloader_val,
            p_bar_name=f"{run_num[0]} / {run_num[1]} : \t{dataset} - {embedder_name} : ",
        )

    if plot_loss:
        model.plot_loss(title=embedder_name)
    for val_roc in model.val_roc:
        wandb.log({f"val_roc": val_roc})

    for r2 in model.r2_val:
        wandb.log({f"r2_val": r2})

    if model.val_roc == []:
        metric = model.r2_val
    else:
        metric = model.val_roc
    n_epochs = len(metric)
    df_results = pd.DataFrame(
        {
            "epoch": range(n_epochs),
            "embedder": [embedder_name] * n_epochs,
            "dataset": [dataset] * n_epochs,
            "length": [length] * n_epochs,
            "metric": metric,
        }
    )
    return df_results


def main(args):
    final_res = []

    for dataset in args.datasets:
        if (
            not args.test_run
        ):  # If we are not in test run, we use the config from the parser
            model_config = FFConfig(
                hidden_dim=args.hidden_dim,
                n_layers=args.n_layers,
                d_rate=args.d_rate,
                norm=args.norm,
                lr=args.lr,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                device=args.device,
                test_batch_size=args.test_batch_size,
            )
        else:  # If we are in test run, we use the config file storing the architecture for each dataset
            with open(args.config, "r") as f:
                config = yaml.safe_load(f)
                config = config[dataset]
            model_config = FFConfig(
                hidden_dim=config.hidden_dim,
                n_layers=config.n_layers,
                d_rate=config.d_rate,
                norm=args.norm,
                lr=args.lr,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                device=args.device,
            )
        for random_seed in tqdm(
            range(args.n_runs), desc=f"Dataset {dataset}", position=1, leave=False
        ):
            splits = get_dataset_split(dataset, random_seed=random_seed)

            for i, embedder_name in enumerate(args.embedders):
                for split in splits:
                    split_idx, smiles, mols = get_split_idx(dataset, split)
                    # Get all enmbedders
                    feature_extractor = MolecularFeatureExtractor(
                        dataset=dataset, length=args.length, device=args.device
                    )
                    embedders = get_embedders(MODELS + DESCRIPTORS, feature_extractor)

                    final_res.append(
                        launch_evaluation(
                            dataset=dataset,
                            length=args.length,
                            embedder_name=embedder_name,
                            split_idx=split_idx,
                            device=args.device,
                            model_config=model_config,
                            smiles=smiles,
                            mols=mols,
                            embedders=embedders,
                            plot_loss=args.plot_loss,
                            test_run=args.test_run,
                            run_num=(i, len(args.embedders)),
                        )
                    )

    df = pd.concat(final_res).reset_index(drop=True)
    df_grouped = (
        df.groupby(["epoch", "embedder", "dataset", "length"]).mean().reset_index()
    )

    # Find best epoch for each embedder - dataset
    best_epoch = (
        df_grouped.groupby(["embedder", "dataset"])
        .apply(lambda x: x.loc[x["metric"].idxmax()])
        .reset_index(drop=True)
    )
    df = df[["epoch", "embedder", "dataset", "length"]].join(
        best_epoch.set_index(["epoch", "embedder", "dataset", "length"]),
        on=["epoch", "embedder", "dataset", "length"],
        how="inner",
    )

    wandb.log({"mean_metric": df.groupby("embedder")["metric"].mean().mean()})
    df = wandb.Table(dataframe=df)
    wandb.log({"results_df": df})


if __name__ == "__main__":
    from parser_mol import add_downstream_args, add_FF_downstream_args

    wandb.init(project="Emir-downstream")

    parser = argparse.ArgumentParser(
        description="Launch the evaluation of a downstream model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser = add_downstream_args(parser)
    parser = add_FF_downstream_args(parser)

    args = parser.parse_args()

    if args.hpo_whole_config is not None:
        config = args.hpo_whole_config.split("_")
        args.n_layers = int(config[0])
        args.hidden_dim = int(config[1])
        args.d_rate = float(config[2])

    if args.embedders is None:
        args.embedders = MODELS  # + DESCRIPTORS

    for group in DATASETS_GROUP.keys():
        if group in args.datasets:
            args.datasets.remove(group)
            args.datasets += DATASETS_GROUP[group]

    wandb.config.update(args)
    main(args)
