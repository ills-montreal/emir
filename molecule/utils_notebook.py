import os

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm as tqdm
import numpy as np
import json
import sklearn
from sklearn.decomposition import PCA
import yaml

from utils import MolecularFeatureExtractor
from models.model_paths import get_model_path

from main import GROUPED_MODELS


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

MODELS_PATH = get_model_path(models=MODELS)


def plot_embeddings(
    DATASET,
    LENGTH=1024,
    use_VAE=False,
    VAE_dim=128,
    VAE_n_layers=2,
    VAE_int_dim=256,
    MODELS=MODELS,
    MODELS_PATH=MODELS_PATH,
):
    with open(f"data/{DATASET}/smiles.json", "r") as f:
        smiles = json.load(f)

    if not len(smiles) > 50000:
        import datamol as dm

        mols = dm.read_sdf(f"data/{DATASET}/preprocessed.sdf")

        feature_extractor = MolecularFeatureExtractor(
            dataset=DATASET,
            length=LENGTH,
            use_vae=use_VAE,
            vae_path=f"data/{DATASET}/VAE/latent_dim_{VAE_dim}/n_layers_{VAE_n_layers}/intermediate_dim_{VAE_int_dim}",
            device="cuda",
        )
        # same plots but in 3D

        fig, axes = plt.subplots(
            4, len(MODELS) // 4, figsize=(len(MODELS) // 4 * 5, 4 * 5)
        )
        axes = axes.flatten()
        df_desc = dm.descriptors.batch_compute_many_descriptors(mols)
        print("Descriptors computed")

        for i, model in enumerate(MODELS):
            embeddings = feature_extractor.get_features(
                smiles,
                mols=mols,
                name=model,
                feature_type="model",
                path=MODELS_PATH.get(model, None),
            )
            # nromalize embeddings
            embeddings = (embeddings - embeddings.mean(axis=0)) / (
                embeddings.std(axis=0) + 1e-8
            )
            pca = PCA(n_components=3)
            embeddings_pca = pca.fit_transform(embeddings.cpu())
            df = pd.DataFrame(embeddings_pca, columns=[f"PC{i}" for i in range(1, 4)])
            df["smiles"] = smiles
            # using pyplot
            sns.scatterplot(
                data=df,
                x="PC1",
                y="PC2",
                ax=axes[i],
                hue=df_desc["sas"],
                cmap="viridis",
                legend=False,
                alpha=0.5,
            )
            axes[i].set_title(model)
            axes[i].set_xlabel("PC1")
            axes[i].set_ylabel("PC2")
        plt.tight_layout()
        plt.show()


def get_loss_df(
    DATASET,
    results_dir_list,
    LENGTH=1024,
    use_VAE=True,
    VAE_dim=64,
    args_to_add=[
        "cond_modes",
        "marg_modes",
        "ff_layers",
        "ff_hidden_dim",
        "batch_size",
    ],
):
    full_df_loss_cond = []
    full_df_loss_marg = []

    for results_dir in tqdm(results_dir_list):
        RESULTS_PATH = f"results/{DATASET}/{LENGTH}/{use_VAE}_{VAE_dim}/{results_dir}"

        with open(RESULTS_PATH + "/args.yaml", "r") as f:
            args = yaml.load(f, Loader=yaml.FullLoader)
        dir_path = os.path.join(RESULTS_PATH, "losses")

        for file in os.listdir(dir_path):
            if file.endswith(".csv") and file[:-4].split("_")[0] == DATASET:
                file_split = file[:-4].split("_")
                if file_split[-1] == "marg":
                    df_tmp = pd.read_csv(os.path.join(dir_path, file))
                    for arg in args_to_add:
                        df_tmp[arg] = args[arg]
                    full_df_loss_marg.append(df_tmp)
                else:
                    df_tmp = pd.read_csv(os.path.join(dir_path, file))
                    for arg in args_to_add:
                        df_tmp[arg] = args[arg]
                    full_df_loss_cond.append(df_tmp)

    full_df_loss_marg = pd.concat(full_df_loss_marg)
    full_df_loss_cond = pd.concat(full_df_loss_cond)

    return full_df_loss_marg, full_df_loss_cond


def get_MI_df(
    DATASET,
    results_dir_list,
    LENGTH=1024,
    use_VAE=True,
    VAE_dim=64,
    args_to_add=[
        "cond_modes",
        "marg_modes",
        "ff_layers",
        "ff_hidden_dim",
        "batch_size",
    ],
):
    full_df_MI = []

    for results_dir in tqdm(results_dir_list):
        RESULTS_PATH = f"results/{DATASET}/{LENGTH}/{use_VAE}_{VAE_dim}/{results_dir}"

        with open(RESULTS_PATH + "/args.yaml", "r") as f:
            args = yaml.load(f, Loader=yaml.FullLoader)

        all_df = []
        for file in os.listdir(RESULTS_PATH):
            if file.endswith(".csv"):
                file_split = file[:-4].split("_")
                if file_split[0] == DATASET and file_split[-1] == str(LENGTH):
                    all_df.append(pd.read_csv(os.path.join(RESULTS_PATH, file)))
        all_df = pd.concat(all_df)
        for arg in args_to_add:
            all_df[arg] = args[arg]
        full_df_MI.append(all_df)

    full_df_MI = pd.concat(full_df_MI)

    return full_df_MI